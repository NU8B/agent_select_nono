import os
import json
from universa.memory.chromadb.chromadb import ChromaDB
from universa.memory.embedding_functions.chromadb_default import ChromaDBDefaultEF
from data.benchmark_data import QUERY_AGENT_MAPPING
import time
import psutil
from typing import List, Dict, Tuple
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
import torch
import torch.cuda.amp  # For mixed precision
import functools  # For caching
from data.benchmark_data import (
    QUERY_AGENT_MAPPING,
    get_benchmark_metrics,
    get_detailed_results,
)


def format_memory(bytes_value: float) -> str:
    """Format memory size to human readable format"""
    for unit in ["MB", "GB"]:
        if bytes_value < 1024:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.2f} TB"


# Cache the agent loading function
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
                        "input_cost": data.get("input_cost", 0),
                        "output_cost": data.get("output_cost", 0),
                        "popularity": data.get("popularity", 0),
                        "average_rating": data.get("average_rating", 0),
                    },
                )()
                agents.append(agent)
    return agents


def select_best_agent(
    agents: List, query: str, chroma: ChromaDB, max_rating: float = None
) -> Tuple[object, List[Dict]]:
    """
    Select best agent with optimized processing.
    Pre-calculates max_rating to avoid repeated calculations.

    Scoring Process:
    1. Semantic Distance (70% weight):
       - Uses ChromaDB to calculate semantic similarity
       - Distances normalized to 0-1 scale
       - Distance squared to penalize less relevant matches
       - Final distance score = (1 - normalized_distance²) * 0.7

    2. Agent Rating (30% weight):
       - Uses agent's average rating
       - Ratings normalized to 0-1 scale
       - Final rating score = normalized_rating * 0.3
    """
    # Pre-compute agent lookup once
    agent_lookup = {agent.description: agent for agent in agents}

    # Batch query processing
    result = chroma.query_data([query])
    documents, distances = result["documents"][0], result["distances"][0]
    max_distance = max(distances)

    # Vectorized calculations
    normalized_distances = [d / max_distance for d in distances]
    normalized_ratings = [
        agent_lookup[doc].average_rating / max_rating for doc in documents
    ]

    # Compute scores in one go
    scores = [
        (1 - nd**2) * 0.7 + nr * 0.3
        for nd, nr in zip(normalized_distances, normalized_ratings)
    ]

    # Find best score index
    best_idx = max(range(len(scores)), key=scores.__getitem__)

    return agent_lookup[documents[best_idx]], sorted(
        [
            {
                "agent_name": agent_lookup[doc].name,
                "distance": dist,
                "normalized_distance": nd,
                "average_rating": agent_lookup[doc].average_rating,
                "normalized_rating": nr,
                "combined_score": score,
                "response_time": agent_lookup[doc].response_time,
                "input_cost": agent_lookup[doc].input_cost,
                "output_cost": agent_lookup[doc].output_cost,
                "popularity": agent_lookup[doc].popularity,
            }
            for doc, dist, nd, nr, score in zip(
                documents, distances, normalized_distances, normalized_ratings, scores
            )
        ],
        key=lambda x: x["combined_score"],
        reverse=True,
    )


def write_results(
    query: str, best_agent: object, selection_details: List[Dict], output_dir: str
) -> None:
    """Write results to files with consistent formatting"""
    # Write to text file
    with open(os.path.join(output_dir, "results_1.txt"), "a") as txt_file:
        txt_file.write(f"Query: {query}\nSelected Agent: {best_agent.name}\n\n")

    # Write to markdown file
    with open(os.path.join(output_dir, "results_1.md"), "a") as md_file:
        md_file.write(f"## Query: {query}\n\n")
        md_file.write(f"**Selected Agent**: {best_agent.name}\n\n")
        md_file.write("### Top 3 Agent Matches:\n\n")

        for detail in selection_details[:3]:  # Only show top 3 matches
            md_file.write(
                f"**Agent**: {detail['agent_name']}\n"
                f"- **Combined Score**: {detail['combined_score']:.4f}\n"
                f"- **Distance**: {detail['distance']:.4f}\n"
                f"- **Average Rating**: {detail['average_rating']:.2f}\n\n"
            )
        md_file.write("\n---\n\n")


def write_benchmark_results(predictions: dict, output_dir: str) -> None:
    """Write benchmark results to output files"""
    metrics = get_benchmark_metrics(predictions)
    detailed_results = get_detailed_results(predictions)

    # Write to markdown file
    with open(os.path.join(output_dir, "results_1.md"), "r+", encoding="utf-8") as f:
        content = f.read()
        f.seek(0)
        f.write("# Agent Selection Results - Algorithm 1\n\n")
        f.write("## Benchmark Summary\n\n")
        f.write(f"**Accuracy**: {metrics['accuracy']:.2%}\n")
        f.write(
            f"**Correct Predictions**: {metrics['correct_predictions']}/{metrics['total_queries']}\n\n"
        )
        f.write("---\n\n")

        # Split content into individual query sections
        sections = content.split("## Query:")

        # Write each section with benchmark information added
        for i, section in enumerate(sections):
            if i == 0:  # Skip the header section
                continue

            # Get the query from the section
            query = section.split("\n")[0].strip()

            # Find the corresponding benchmark result
            result = next((r for r in detailed_results if r["query"] == query), None)

            if result:
                # Write the query with benchmark status
                f.write("## Query:" + query + "\n")
                status = (
                    "[CORRECT]"
                    if result["is_correct"]
                    else f"[INCORRECT - Expected: {result['correct_agent']}]"
                )
                f.write(f"**Benchmark**: {status}\n\n")

                # Write the rest of the section
                remaining_content = "\n".join(section.split("\n")[1:])
                f.write(remaining_content)
            else:
                f.write("## Query:" + section)

    # Update text file similarly
    with open(os.path.join(output_dir, "results_1.txt"), "r+", encoding="utf-8") as f:
        content = f.read()
        f.seek(0)
        f.write("Agent Selection Results - Algorithm 1\n\n")
        f.write("Benchmark Summary\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.2%}\n")
        f.write(
            f"Correct Predictions: {metrics['correct_predictions']}/{metrics['total_queries']}\n\n"
        )
        f.write("---\n\n")

        # Split content into individual query sections
        sections = content.split("Query:")

        # Write each section with benchmark information added
        for i, section in enumerate(sections):
            if i == 0:  # Skip the header section
                continue

            # Get the query from the section
            query = section.split("\n")[0].strip()

            # Find the corresponding benchmark result
            result = next((r for r in detailed_results if r["query"] == query), None)

            if result:
                # Write the query with benchmark status
                f.write("Query:" + query + "\n")
                status = (
                    "[CORRECT]"
                    if result["is_correct"]
                    else f"[INCORRECT - Expected: {result['correct_agent']}]"
                )
                f.write(f"Benchmark: {status}\n\n")

                # Write the rest of the section
                remaining_content = "\n".join(section.split("\n")[1:])
                f.write(remaining_content)
            else:
                f.write("Query:" + section)


def main():
    """Optimized main execution flow with improved output"""
    # Add total timing
    total_start_time = time.time()

    # Initialize process and start memory
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024

    console = Console()

    # Display GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[bold green]Using GPU: {gpu_name}[/bold green]")
    else:
        console.print("[bold yellow]GPU not available, using CPU[/bold yellow]")

    # Initialize timing for each step
    step_times = {}

    # Load agents
    step_start = time.time()
    agents = load_agents()
    max_rating = max(agent.average_rating for agent in agents)
    step_times["load_agents"] = time.time() - step_start

    # Initialize ChromaDB
    step_start = time.time()
    chroma = ChromaDB(
        embedding_function=ChromaDBDefaultEF(), collection_name="agents_collection"
    )
    step_times["init_chroma"] = time.time() - step_start

    # Add agent descriptions
    step_start = time.time()
    agent_descriptions = [agent.description for agent in agents]
    chroma.add_data(
        documents=agent_descriptions,
        ids=[agent.name for agent in agents],
    )
    step_times["add_descriptions"] = time.time() - step_start

    # Print step times
    console.print("\n[bold cyan]Step-by-Step Timing:[/bold cyan]")
    console.print(
        f"Loading agents: [yellow]{step_times['load_agents']:.2f}[/yellow] seconds"
    )
    console.print(
        f"Initializing ChromaDB: [yellow]{step_times['init_chroma']:.2f}[/yellow] seconds"
    )
    console.print(
        f"Adding descriptions: [yellow]{step_times['add_descriptions']:.2f}[/yellow] seconds"
    )

    # Preload CUDA kernels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Pre-warm the model
    _ = chroma.query_data(["warmup query"])

    # Setup output directory
    output_dir = "output/algorithm_1"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize output files
    for filename, header in [
        ("results_1.md", "# Agent Selection Results - Algorithm 1\n\n"),
        ("results_1.txt", "Agent Selection Results - Algorithm 1\n\n"),
    ]:
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(header)

    # Add predictions dictionary
    predictions = {}

    # Process queries with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[bold blue]Processing queries...", total=len(QUERY_AGENT_MAPPING)
        )

        for query in QUERY_AGENT_MAPPING:
            best_agent, selection_details = select_best_agent(
                agents, query, chroma, max_rating
            )

            # Track prediction
            predictions[query] = best_agent.name

            write_results(query, best_agent, selection_details, output_dir)
            progress.advance(task)

    # Write benchmark results before performance metrics
    write_benchmark_results(predictions, output_dir)

    # Calculate and display performance metrics
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024
    execution_time = end_time - total_start_time
    memory_used = end_memory - start_memory

    # Print performance summary
    console.print("\n[bold green]Performance Summary:[/bold green]")
    console.print(f"📂 Results saved to: [cyan]{output_dir}[/cyan]")
    console.print(f"⏱️  Execution time: [yellow]{execution_time:.2f}[/yellow] seconds")
    console.print(f"💾 Memory used: [yellow]{format_memory(memory_used)}[/yellow]")
    console.print(f"📊 Peak memory: [yellow]{format_memory(end_memory)}[/yellow]")

    # Write metrics to files
    metrics = (
        f"\nPerformance Metrics:\n"
        f"Total execution time: {execution_time:.2f} seconds\n"
        f"Memory used: {format_memory(memory_used)}\n"
        f"Peak memory: {format_memory(end_memory)}\n"
    )

    with open(os.path.join(output_dir, "results_1.txt"), "a") as f:
        f.write(metrics)

    with open(os.path.join(output_dir, "results_1.md"), "a") as f:
        f.write(
            f"\n## Performance Metrics\n\n"
            f"- **Total execution time**: {execution_time:.2f} seconds\n"
            f"- **Memory used**: {format_memory(memory_used)}\n"
            f"- **Peak memory**: {format_memory(end_memory)}\n"
        )


if __name__ == "__main__":
    main()
