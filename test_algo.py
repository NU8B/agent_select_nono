import os
import json
import torch
from universa.memory.chromadb.persistent_chromadb import ChromaDB
from data.test_data import (
    QUERY_AGENT_MAPPING,
    get_benchmark_metrics,
    get_detailed_results,
)
import time
from typing import List, Dict, Tuple
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
import torch.cuda.amp
import functools
import numpy as np
from universa.utils.agent_compute import agent_cache
from rapidfuzz import fuzz
import hashlib

# Timing dictionaries
step_times = {}
query_times = {}


@functools.lru_cache(maxsize=1)
def load_agents() -> List:
    """Load agent definitions with memory-efficient processing"""
    agents = []
    for filename in os.listdir("data/agents"):
        if filename.endswith(".json"):
            with open(os.path.join("data/agents", filename)) as file:
                data = json.load(file)
                agent = type(
                    "Agent",
                    (),
                    {
                        "name": data.get("name"),
                        "description": data.get("description"),
                        "response_time": data.get("response_time", 0),
                        "average_rating": data.get("average_rating", 0),
                        "rated_responses": data.get("rated_responses", 0),
                    },
                )()
                agents.append(agent)
    return agents


def select_best_agent(  #
    agents: List, query: str, chroma: ChromaDB, max_rating: float = None
) -> Tuple[object, List[Dict]]:
    """Select best agent with optimized processing."""
    result = chroma.query_data(query_text=[query])

    documents = result["documents"][0]
    distances = np.array(result["distances"][0])

    # Use cached values
    cache_values = agent_cache.values
    agent_lookup = cache_values["agent_lookup"]
    max_responses = cache_values["max_responses"]
    agents_list = [agent_lookup[doc] for doc in documents]

    # Vectorized calculations
    rated_responses = np.array([agent.rated_responses for agent in agents_list])
    response_ratios = rated_responses / max_responses

    # Calculate weights following the new pattern
    response_weights = 0.2 + (0.1 * response_ratios)
    semantic_weights = 0.7 - (0.1 * response_ratios)
    lexical_weights = np.full_like(response_weights, 0.1)  # Fixed 10%

    # Calculate lexical scores
    lexical_scores = np.array(
        [compute_lexical_score(query, agent.description) for agent in agents_list]
    )

    # Normalize distances in single operation
    normalized_distances = distances / distances.max()

    # Compute ratings in single operation
    normalized_ratings = np.array(
        [
            (
                agent.average_rating / cache_values["max_rating"]
                if cache_values["max_rating"] > 0
                else 0
            )
            for agent in agents_list
        ]
    )

    # Updated score calculation including lexical component
    combined_scores = (
        (1 - normalized_distances**2) * semantic_weights
        + normalized_ratings * response_weights
        + lexical_scores * lexical_weights
    )

    best_idx = np.argmax(combined_scores)
    best_agent = agents_list[best_idx]

    selection_details = [
        {
            "agent_name": agent.name,
            "distance": float(dist),
            "normalized_distance": float(norm_dist),
            "average_rating": float(agent.average_rating),
            "normalized_rating": float(norm_rating),
            "rating_weight": float(r_weight),
            "semantic_weight": float(s_weight),
            "lexical_weight": float(l_weight),
            "lexical_score": float(lex_score),
            "combined_score": float(score),
            "rated_responses": agent.rated_responses,
        }
        for agent, dist, norm_dist, norm_rating, r_weight, s_weight, l_weight, lex_score, score in zip(
            agents_list,
            distances,
            normalized_distances,
            normalized_ratings,
            response_weights,
            semantic_weights,
            lexical_weights,
            lexical_scores,
            combined_scores,
        )
    ]

    return best_agent, sorted(
        selection_details, key=lambda x: x["combined_score"], reverse=True
    )


def compute_lexical_score(query: str, description: str) -> float:
    """Compute normalized lexical similarity score"""
    return fuzz.token_sort_ratio(query.lower(), description.lower()) / 100.0


def write_results(query: str, selected_agent: object, details: List[Dict]) -> str:
    """Write detailed results for each query"""
    result_text = f"**Selected Agent**: {selected_agent.name}\n"
    result_text += "\n### Top 3 Agent Matches:\n\n"

    sorted_details = sorted(details, key=lambda x: x["combined_score"], reverse=True)[
        :3
    ]
    for detail in sorted_details:
        result_text += f"**Agent**: {detail['agent_name']}\n"
        result_text += f"- **Combined Score**: {detail['combined_score']:.4f}\n"
        result_text += f"- **Distance**: {detail['distance']:.4f}\n"
        result_text += f"- **Lexical Score**: {detail['lexical_score']:.4f}\n"
        result_text += f"- **Average Rating**: {detail['average_rating']:.2f}\n"
        result_text += f"- **Rated Responses**: {detail['rated_responses']}\n"
        result_text += f"- **Distance Weight**: {detail['semantic_weight']:.2f}\n"
        result_text += f"- **Rating Weight**: {detail['rating_weight']:.2f}\n"
        result_text += f"- **Lexical Weight**: {detail['lexical_weight']:.2f}\n\n"

    result_text += "\n---\n"
    return result_text


def write_benchmark_results(predictions: dict, output_dir: str) -> None:
    """Write benchmark results to output files with incorrect predictions highlighted at the top"""
    metrics = get_benchmark_metrics(
        {k: v["predicted_agent"] for k, v in predictions.items()}
    )
    detailed_results = get_detailed_results(
        {k: v["predicted_agent"] for k, v in predictions.items()}
    )

    with open(os.path.join(output_dir, "test_result.md"), "w", encoding="utf-8") as f:
        # Write header and summary
        f.write(f"# Agent Selection Results - Test\n\n")
        f.write("## Benchmark Summary\n\n")
        f.write(f"**Accuracy**: {metrics['accuracy']:.2%}\n")
        f.write(
            f"**Correct Predictions**: {metrics['correct_predictions']}/{metrics['total_queries']}\n\n"
        )

        # Sort results to put incorrect predictions first
        incorrect_results = []
        correct_results = []

        for result in detailed_results:
            query = result["query"]
            result["details"] = predictions[query]["details"]
            if not result["is_correct"]:
                incorrect_results.append(result)
            else:
                correct_results.append(result)

        # Write incorrect predictions section
        if incorrect_results:
            f.write("## ❌ Incorrect Predictions\n\n")
            for result in incorrect_results:
                query = result["query"]
                f.write(f"## Query:{query}\n")
                f.write(
                    f"**Benchmark**: [INCORRECT - Expected: {result['correct_agent']}]**\n\n"
                )
                f.write(result["details"])
                f.write("\n---\n\n")

        # Write correct predictions section
        if correct_results:
            f.write("## ✅ Correct Predictions\n\n")
            for result in correct_results:
                query = result["query"]
                f.write(f"## Query:{query}\n")
                f.write(f"**Benchmark**: [CORRECT]**\n\n")
                f.write(result["details"])
                f.write("\n---\n\n")


def format_memory(bytes_value: float) -> str:
    """Format memory size to human readable string"""
    return f"{bytes_value:.2f} MB"


def initialize_chromadb():
    """Initialize ChromaDB with fixed Stella model"""
    # Load agents to create unique collection name
    agents = load_agents()
    descriptions = [agent.description for agent in agents]
    descriptions_str = "||".join(
        sorted(descriptions)
    )  # Join sorted descriptions with delimiter
    collection_hash = hashlib.sha256(descriptions_str.encode()).hexdigest()[
        :8
    ]  # Use first 8 chars

    chroma = ChromaDB(collection_name=f"agent_descriptions_test_{collection_hash}")

    # Get embedding dimension from a test query
    test_embedding = chroma.embedding_function.create_embeddings(["test"])[0]
    embedding_dim = len(test_embedding)

    console = Console()
    console.print(f"[blue]Model embedding dimension: {embedding_dim}[/blue]")

    return chroma


def main():
    """Main execution function"""
    total_start = time.time()
    total_query_time = 0  # Add this line to track total query time

    console = Console()

    # Initial console output for GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[green]Using GPU: {gpu_name}[/green]")
    else:
        console.print("[yellow]GPU not available, using CPU[/yellow]")

    # Load agents and initialize cache
    step_start = time.time()
    agents = load_agents()
    agent_cache.initialize(agents)
    max_rating = max(agent.average_rating for agent in agents)
    step_times["load_agents"] = time.time() - step_start

    # Initialize ChromaDB
    step_start = time.time()
    chroma = initialize_chromadb()
    step_times["init_chroma"] = time.time() - step_start

    # Check if collection exists and has data
    if chroma.get_count() > 0:
        console.print("[green]Using existing ChromaDB collection[/green]")
        step_times["add_descriptions"] = 0
    else:
        console.print("[green]Initializing new ChromaDB collection[/green]")
        step_start = time.time()
        agent_descriptions = [agent.description for agent in agents]
        chroma.add_data(
            documents=agent_descriptions,
            ids=[agent.name for agent in agents],
        )
        step_times["add_descriptions"] = time.time() - step_start

    # Setup output directory
    output_dir = "output/test"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "test_result.md"), "w") as f:
        f.write("# Agent Selection Results - Test\n\n")

    # Pre-warm model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _ = chroma.query_data(query_text=["warmup query"])

    # Process queries
    predictions = {}
    results_details = {}  # Store details for each query

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            "[cyan]Processing queries...", total=len(QUERY_AGENT_MAPPING)
        )

        for query in QUERY_AGENT_MAPPING:
            query_start = time.time()
            best_agent, selection_details = select_best_agent(
                agents, query, chroma, max_rating
            )
            total_query_time += time.time() - query_start

            # Store both prediction and details
            predictions[query] = {
                "predicted_agent": best_agent.name,
                "details": write_results(query, best_agent, selection_details),
            }

            progress.advance(task)

    # Write the complete results file with all details
    write_benchmark_results(predictions, output_dir)

    # Calculate final metrics
    total_time = time.time() - total_start
    setup_time = sum(step_times.values())
    avg_query_time = total_query_time / len(QUERY_AGENT_MAPPING)

    # Calculate benchmark metrics
    metrics = get_benchmark_metrics(
        {k: v["predicted_agent"] for k, v in predictions.items()}
    )  # Extract just the predicted agent names

    # Print results
    console.print(f"📂 Results saved to: [cyan]{output_dir}[/cyan]")
    console.print(
        f"⏱️  [bold white]Total execution time: {total_time:.2f} seconds[/bold white]"
    )
    console.print(
        f"⚙️  Total setup time: [bold yellow]{setup_time:.2f}[/bold yellow] seconds"
    )
    console.print("   └─ Breakdown:")
    console.print(
        f"      ├─ Loading agents: [dim]{step_times['load_agents']:.2f}[/dim] seconds"
    )
    console.print(
        f"      ├─ Initializing ChromaDB: [dim]{step_times['init_chroma']:.2f}[/dim] seconds"
    )
    console.print(
        f"      └─ Adding descriptions: [dim]{step_times['add_descriptions']:.2f}[/dim] seconds"
    )
    console.print(
        f"🔄 Total query time: [bold yellow]{total_query_time:.2f}[/bold yellow] seconds"
    )
    console.print(f"   ├─ Average per query: [dim]{avg_query_time:.4f}[/dim] seconds")
    console.print(
        f"   └─ Accuracy: [dim]{metrics['correct_predictions']}/{metrics['total_queries']}[/dim] ({metrics['accuracy']:.1%})"
    )

    # Write metrics to files with detailed breakdown
    metrics_text = (
        f"\nPerformance Metrics:\n"
        f"\nTiming Breakdown:\n"
        f"Setup/overhead time: {setup_time:.2f} seconds\n"
        f"Loading agents: {step_times['load_agents']:.2f} seconds\n"
        f"Initializing ChromaDB: {step_times['init_chroma']:.2f} seconds\n"
        f"Adding descriptions: {step_times['add_descriptions']:.2f} seconds\n"
        f"Query processing time: {total_query_time:.2f} seconds\n"
        f"Average query time: {avg_query_time:.4f} seconds\n"
        f"Total execution time: {total_time:.2f} seconds\n"
    )

    with open(os.path.join(output_dir, "test_result.md"), "a") as f:
        f.write(
            f"\n## Performance Metrics\n\n"
            f"- **Total execution time**: {total_time:.2f} seconds\n"
            f"- **Average query time**: {avg_query_time:.4f} seconds\n"
            f"- **Total query processing time**: {total_query_time:.2f} seconds\n"
        )


if __name__ == "__main__":
    main()
