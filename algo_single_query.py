import os
import json
import torch
from universa.memory.chromadb.persistent_chromadb_single import ChromaDB
from data.single_query import QUERY
from typing import List, Dict, Tuple
from rich.console import Console
import functools
import numpy as np


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
    distances = result["distances"][0]
    max_distance = max(distances)
    max_responses = max(agent.rated_responses for agent in agents)
    agent_lookup = {agent.description: agent for agent in agents}

    normalized_distances = np.array(distances) / max_distance
    agents_list = [agent_lookup[doc] for doc in documents]

    response_weights = np.array(
        [0.2 + (0.1 * (agent.rated_responses / max_responses)) for agent in agents_list]
    )

    normalized_ratings = np.array(
        [
            agent.average_rating / max_rating if max_rating > 0 else 0
            for agent in agents_list
        ]
    )

    distance_weights = 1 - response_weights
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


def main():
    """Main execution function"""
    console = Console()

    # Initial console output for GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[green]Using GPU: {gpu_name}[/green]")
    else:
        console.print("[yellow]GPU not available, using CPU[/yellow]")

    # Load agents
    agents = load_agents()
    max_rating = max(agent.average_rating for agent in agents)

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
    best_agent, selection_details = select_best_agent(agents, query, chroma, max_rating)
    query_time = time.time() - query_start

    # Print results
    console.print(f"\n[yellow]Query:[/yellow] {query}")
    console.print(f"[green]Selected Agent:[/green] {best_agent.name}")
    console.print(f"Query processing Time: {query_time:.4f} seconds")


if __name__ == "__main__":
    main()
