import os
import json
import torch
from universa.memory.chromadb.persistent_chromadb import ChromaDB
from data.benchmark_data import (
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
    response_weights = 0.2 + (0.1 * (rated_responses / max_responses))
    distance_weights = 1 - response_weights

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

    combined_scores = (
        1 - normalized_distances**2
    ) * distance_weights + normalized_ratings * response_weights

    best_idx = np.argmax(combined_scores)
    best_agent = agents_list[best_idx]

    selection_details = [
        {
            "agent_name": agent.name,
            "distance": float(dist),
            "normalized_distance": float(norm_dist),
            "average_rating": float(agent.average_rating),
            "normalized_rating": float(norm_rating),
            "rating_weight": float(weight),
            "combined_score": float(score),
            "rated_responses": agent.rated_responses,
        }
        for agent, dist, norm_dist, norm_rating, weight, score in zip(
            agents_list,
            distances,
            normalized_distances,
            normalized_ratings,
            response_weights,
            combined_scores,
        )
    ]

    return best_agent, sorted(
        selection_details, key=lambda x: x["combined_score"], reverse=True
    )


def write_results(
    query: str,
    best_agent: object,
    selection_details: List[Dict],
    output_dir: str,
) -> None:
    """Write results to files with consistent formatting"""
    # Write to text file
    with open(os.path.join(output_dir, "test_result.txt"), "a") as txt_file:
        txt_file.write(f"Query: {query}\nSelected Agent: {best_agent.name}\n\n")

    # Write to markdown file
    with open(os.path.join(output_dir, "test_result.md"), "a") as md_file:
        md_file.write(f"## Query: {query}\n\n")
        md_file.write(f"**Selected Agent**: {best_agent.name}\n\n")
        md_file.write("### Top 3 Agent Matches:\n\n")

        for detail in selection_details[:3]:  # Only show top 3 matches
            md_file.write(
                f"**Agent**: {detail['agent_name']}\n"
                f"- **Combined Score**: {detail['combined_score']:.4f}\n"
                f"- **Distance**: {detail['distance']:.4f}\n"
                f"- **Average Rating**: {detail['average_rating']:.2f}\n"
                f"- **Rated Responses**: {detail['rated_responses']}\n"
                f"- **Distance Weight**: {1 - detail['rating_weight']:.2f}\n"
                f"- **Rating Weight**: {detail['rating_weight']:.2f}\n\n"
            )
        md_file.write("\n---\n\n")


def write_benchmark_results(predictions: dict, output_dir: str) -> None:
    """Write benchmark results to output files with incorrect predictions highlighted at the top"""
    metrics = get_benchmark_metrics(predictions)
    detailed_results = get_detailed_results(predictions)

    # Sort results to put incorrect predictions first
    incorrect_results = []
    correct_results = []

    with open(os.path.join(output_dir, "test_result.md"), "r", encoding="utf-8") as f:
        content = f.read()
        sections = content.split("## Query:")
        header = sections[0]  # Save the header section

        for section in sections[1:]:  # Skip header section
            query = section.split("\n")[0].strip()
            result = next((r for r in detailed_results if r["query"] == query), None)

            if result:
                if not result["is_correct"]:
                    incorrect_results.append((query, section, result))
                else:
                    correct_results.append((query, section, result))

    # Write to markdown file with incorrect results first
    with open(os.path.join(output_dir, "test_result.md"), "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# Agent Selection Results - Stella\n\n")
        f.write("## Benchmark Summary\n\n")
        f.write(f"**Accuracy**: {metrics['accuracy']:.2%}\n")
        f.write(
            f"**Correct Predictions**: {metrics['correct_predictions']}/{metrics['total_queries']}\n\n"
        )

        # Write incorrect predictions section
        if incorrect_results:
            f.write("## ❌ Incorrect Predictions\n\n")
            for query, section, result in incorrect_results:
                f.write("## Query:" + query + "\n")
                f.write(
                    f"**Benchmark**: [INCORRECT - Expected: {result['correct_agent']}]**\n\n"
                )
                remaining_content = "\n".join(section.split("\n")[1:])
                f.write(remaining_content)
            f.write("\n---\n\n")

        # Write correct predictions section
        if correct_results:
            f.write("## ✅ Correct Predictions\n\n")
            for query, section, result in correct_results:
                f.write("## Query:" + query + "\n")
                f.write("**Benchmark**: [CORRECT]**\n\n")
                remaining_content = "\n".join(section.split("\n")[1:])
                f.write(remaining_content)

        f.write("\n---\n\n")


def format_memory(bytes_value: float) -> str:
    """Format memory size to human readable string"""
    return f"{bytes_value:.2f} MB"


def initialize_chromadb():
    """Initialize ChromaDB with fixed Stella model"""
    chroma = ChromaDB(collection_name="agent_descriptions")

    # Get embedding dimension from a test query
    test_embedding = chroma.embedding_function.create_embeddings(["test"])[0]
    embedding_dim = len(test_embedding)

    console = Console()
    console.print(f"[blue]Model embedding dimension: {embedding_dim}[/blue]")

    return chroma


def main():
    """Main execution function"""
    total_start = time.time()

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

    # Initialize files
    for filename, header in [
        ("test_result.md", "# Agent Selection Results - Stella\n\n"),
        ("test_result.txt", "Agent Selection Results - Stella\n\n"),
    ]:
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(header)

    # Pre-warm model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _ = chroma.query_data(query_text=["warmup query"])

    # Process queries
    predictions = {}
    total_query_time = 0

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

        # Process in batches
        batch_size = 5
        queries = list(QUERY_AGENT_MAPPING.keys())

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]

            for query in batch_queries:
                query_start = time.time()
                best_agent, selection_details = select_best_agent(
                    agents, query, chroma, max_rating
                )
                query_time = time.time() - query_start

                query_times[query] = query_time
                total_query_time += query_time

                predictions[query] = best_agent.name
                write_results(query, best_agent, selection_details, output_dir)
                progress.advance(task)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Calculate final metrics
    total_time = time.time() - total_start
    setup_time = sum(step_times.values())
    avg_query_time = total_query_time / len(QUERY_AGENT_MAPPING)

    # Calculate benchmark metrics
    metrics = get_benchmark_metrics(predictions)

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
        f"\nSetup/overhead time: {setup_time:.2f} seconds\n"
        f"\nLoading agents: {step_times['load_agents']:.2f} seconds\n"
        f"\nInitializing ChromaDB: {step_times['init_chroma']:.2f} seconds\n"
        f"\nAdding descriptions: {step_times['add_descriptions']:.2f} seconds\n"
        f"\nQuery processing time: {total_query_time:.2f} seconds\n"
        f"\nAverage query time: {avg_query_time:.4f} seconds\n"
        f"\nTotal execution time: {total_time:.2f} seconds\n"
    )

    with open(os.path.join(output_dir, "test_result.txt"), "a") as f:
        f.write(metrics_text)

    with open(os.path.join(output_dir, "test_result.md"), "a") as f:
        f.write(metrics_text)

    write_benchmark_results(predictions, output_dir)


if __name__ == "__main__":
    main()
