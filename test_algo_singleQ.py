import os
import json
import torch
from universa.memory.chromadb.persistent_chromadb_single import ChromaDB
from data.single_query import QUERY
from typing import List, Dict, Tuple
from rich.console import Console
import functools
import numpy as np
from universa.utils.agent_compute import agent_cache


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


def select_best_agent(
    agents: List, query: str, chroma: ChromaDB, max_rating: float = None
) -> Tuple[object, List[Dict]]:
    """Select best agent with optimized processing."""
    result = chroma.query_data(query_text=[query])

    documents = result["documents"][0]
    distances = np.array(result["distances"][0])

    # Use cached values
    cache_values = agent_cache.values
    agent_lookup = cache_values["agent_lookup"]
    agent_values = cache_values["agent_values"]

    # Get pre-calculated values for matched agents
    agents_list = [agent_lookup[doc] for doc in documents]
    agent_data = [agent_values[doc] for doc in documents]

    # Get pre-calculated weights
    response_weights = np.array([data["response_weight"] for data in agent_data])
    distance_weights = np.array([data["distance_weight"] for data in agent_data])
    normalized_ratings = np.array([data["normalized_rating"] for data in agent_data])

    # Normalize distances in single operation
    normalized_distances = distances / distances.max()

    # Compute final scores
    combined_scores = (
        1 - normalized_distances**2
    ) * distance_weights + normalized_ratings * response_weights

    best_idx = np.argmax(combined_scores)
    best_agent = agents_list[best_idx]

    # Create selection details using pre-calculated values
    selection_details = [
        {
            "agent_name": data["name"],
            "distance": float(dist),
            "normalized_distance": float(norm_dist),
            "average_rating": float(data["average_rating"]),
            "normalized_rating": float(data["normalized_rating"]),
            "rating_weight": float(data["response_weight"]),
            "combined_score": float(score),
            "rated_responses": data["rated_responses"],
        }
        for data, dist, norm_dist, score in zip(
            agent_data, distances, normalized_distances, combined_scores
        )
    ]

    return best_agent, sorted(
        selection_details, key=lambda x: x["combined_score"], reverse=True
    )


def main():
    """Main execution function"""
    console = Console()

    # Initial console output for GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[green]Using GPU: {gpu_name}[/green]")
    else:
        console.print("[yellow]GPU not available, using CPU[/yellow]")

    # Load agents and initialize cache
    agents = load_agents()
    agent_cache.initialize(agents)

    # Initialize ChromaDB
    chroma = ChromaDB(collection_name="agent_descriptions")

    # Check if collection exists and has data
    if chroma.get_count() > 0:
        console.print("[green]Using existing ChromaDB collection[/green]")
    else:
        console.print("[green]Initializing new ChromaDB collection[/green]")
        agent_descriptions = [agent.description for agent in agents]
        chroma.add_data(
            documents=agent_descriptions,
            ids=[agent.name for agent in agents],
        )

    # Pre-warm model if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _ = chroma.query_data(query_text=["warmup query"])

    import time

    query_start = time.time()
    query = QUERY
    best_agent, selection_details = select_best_agent(agents, query, chroma)
    query_time = time.time() - query_start

    # Print results
    console.print(f"\n[yellow]Query:[/yellow] {query}")
    console.print(f"[green]Selected Agent:[/green] {best_agent.name}")
    console.print(f"Query processing Time: {query_time:.4f} seconds")


if __name__ == "__main__":
    main()
