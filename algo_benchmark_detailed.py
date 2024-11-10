import os
import sys
import time
import numpy as np
from typing import List, Dict, Tuple, Any
import time
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rapidfuzz import fuzz
import hashlib

# Add the project root to Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from benchmark.selection import SelectionAlgorithm
from benchmark.benchmark import Benchmark
from universa.memory.chromadb.persistent_chromadb import ChromaDB
from universa.utils.agent_compute_dict import agent_dict_cache

console = Console()


def write_results(
    query: str,
    agent_name: str,
    expected_agent: str,
    details: List[Dict],
    output_dir: str,
    is_correct: bool,
):
    """Write detailed results for each query"""
    # Store results to be written later in correct order
    result_text = f"\n## Query:{query}\n"
    result_text += f"**Benchmark**: [{'CORRECT' if is_correct else f'INCORRECT - Expected: {expected_agent}'}]**\n\n"
    result_text += f"**Selected Agent**: {agent_name}\n"
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
    return result_text, is_correct


def compute_lexical_score(query: str, description: str) -> float:
    """Compute normalized lexical similarity score"""
    # Use token sort ratio to handle word order differences
    return fuzz.token_sort_ratio(query.lower(), description.lower()) / 100.0


class StellaDetailedAlgorithm(SelectionAlgorithm):
    def __init__(self, agents: List[Dict[str, Any]], ids: List[str]) -> None:
        self.total_time = 0
        self.query_count = 0
        super().__init__(agents, ids)

    def initialize(self, agents: List[Dict[str, Any]], ids: List[str]) -> None:
        """Initialize the algorithm with provided agents and IDs"""
        start_time = time.time()

        # Initialize agent cache with dictionary-based agents
        agent_dict_cache.initialize(agents)

        # Create a unique collection name using a stable hash of the agent descriptions
        descriptions = [
            agent["description"] + "\n\n" + agent["system_prompt"] for agent in agents
        ]
        descriptions_str = "||".join(
            sorted(descriptions)
        )  # Join sorted descriptions with delimiter
        collection_hash = hashlib.sha256(descriptions_str.encode()).hexdigest()[
            :8
        ]  # Use first 8 chars
        self.chroma = ChromaDB(
            collection_name=f"agent_descriptions_benchmark_detailed_{collection_hash}"
        )

        # Initialize ChromaDB if empty
        if self.chroma.get_count() == 0:
            self.chroma.add_data(
                documents=descriptions,
                ids=ids,
            )

        self.init_time = time.time() - start_time

    def select(self, query: str) -> Tuple[str, str, List[Dict]]:
        """Select best agent for the given query using optimized processing"""
        start_time = time.time()

        result = self.chroma.query_data(query_text=[query])
        documents = result["documents"][0]
        distances = np.array(result["distances"][0])

        # Use cached values from dict cache
        cache_values = agent_dict_cache.values
        agent_lookup = cache_values["agent_lookup"]
        agent_values = cache_values["agent_values"]

        # Get pre-calculated values for matched agents
        agent_data = [agent_values[doc] for doc in documents]

        # Get pre-calculated weights and normalized values
        response_weights = np.array([data["response_weight"] for data in agent_data])
        semantic_weights = np.array([data["semantic_weight"] for data in agent_data])
        lexical_weights = np.array([data["lexical_weight"] for data in agent_data])
        normalized_ratings = np.array(
            [data["normalized_rating"] for data in agent_data]
        )

        # Calculate lexical scores
        lexical_scores = np.array(
            [compute_lexical_score(query, doc) for doc in documents]
        )

        # Normalize distances
        normalized_distances = distances / distances.max()

        # Compute final scores with all three components
        combined_scores = (
            (1 - normalized_distances**2) * semantic_weights
            + normalized_ratings * response_weights
            + lexical_scores * lexical_weights
        )

        # Create selection details
        selection_details = [
            {
                "agent_name": data["name"],
                "distance": float(dist),
                "normalized_distance": float(norm_dist),
                "average_rating": float(data["average_rating"]),
                "normalized_rating": float(data["normalized_rating"]),
                "rating_weight": float(data["response_weight"]),
                "semantic_weight": float(data["semantic_weight"]),
                "lexical_score": float(lex_score),
                "combined_score": float(score),
                "rated_responses": data["rated_responses"],
                "lexical_weight": float(data["lexical_weight"]),
            }
            for data, dist, norm_dist, lex_score, score in zip(
                agent_data,
                distances,
                normalized_distances,
                lexical_scores,
                combined_scores,
            )
        ]

        best_idx = np.argmax(combined_scores)
        best_doc = documents[best_idx]
        best_agent_data = agent_values[best_doc]
        best_id = best_agent_data["object_id"]

        query_time = time.time() - start_time
        self.total_time += query_time
        self.query_count += 1

        return best_id, best_agent_data["name"], selection_details

    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics"""
        return {
            "initialization_time": self.init_time,
            "total_query_time": self.total_time,
            "average_query_time": (
                self.total_time / self.query_count if self.query_count > 0 else 0
            ),
            "query_count": self.query_count,
        }


def main():
    """Detailed benchmark testing"""
    output_dir = os.path.join("output", "benchmark")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize files with UTF-8 encoding
    with open(
        os.path.join(output_dir, "benchmark_result.md"), "w", encoding="utf-8"
    ) as f:
        f.write("# Agent Selection Results - Detailed\n\n")

    benchmark = Benchmark()
    total_start_time = time.time()
    correct_predictions = 0
    total_queries = len(benchmark.queries)

    incorrect_results = []
    correct_results = []

    algorithm = StellaDetailedAlgorithm(benchmark.agents, benchmark.agent_ids)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Processing queries...", total=total_queries)

        for query in benchmark.queries:
            result_id, result_agent, details = algorithm.select(query["query"])
            result_text, is_correct = write_results(
                query["query"],
                result_agent,
                query["agent"],
                details,
                output_dir,
                query["object_id"] == result_id,
            )

            if is_correct:
                correct_results.append(result_text)
                correct_predictions += 1
            else:
                incorrect_results.append(result_text)

            progress.advance(task)

    # Calculate metrics
    accuracy = correct_predictions / total_queries
    total_time = time.time() - total_start_time
    stats = algorithm.get_stats()

    # Write results in correct order
    with open(
        os.path.join(output_dir, "benchmark_result.md"), "a", encoding="utf-8"
    ) as f:
        # Write summary
        f.write("\n## Benchmark Summary\n\n")
        f.write(f"**Accuracy**: {accuracy:.2%}\n")
        f.write(f"**Correct Predictions**: {correct_predictions}/{total_queries}\n\n")

        # Write incorrect predictions first
        if incorrect_results:
            f.write("\n## ❌ Incorrect Predictions\n")
            for result in incorrect_results:
                f.write(result)

        # Write correct predictions second
        if correct_results:
            f.write("\n## ✅ Correct Predictions\n")
            for result in correct_results:
                f.write(result)

    # Add completion message
    console.print(f"\nResults saved to: {(output_dir)}")


if __name__ == "__main__":
    main()
