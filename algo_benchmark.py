import os
import sys
import time
import numpy as np
from typing import List, Dict, Tuple, Any
import time
from rapidfuzz import fuzz

# Add the project root to Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from benchmark.selection import SelectionAlgorithm
from benchmark.benchmark import Benchmark
from universa.memory.chromadb.persistent_chromadb import ChromaDB
from universa.utils.agent_compute_dict import agent_dict_cache


def compute_lexical_score(query: str, description: str) -> float:
    """Compute normalized lexical similarity score"""
    return fuzz.token_sort_ratio(query.lower(), description.lower()) / 100.0


class StellaAlgorithm(SelectionAlgorithm):
    def __init__(self, agents: List[Dict[str, Any]], ids: List[str]) -> None:
        self.total_time = 0
        self.query_count = 0
        super().__init__(agents, ids)

    def initialize(self, agents: List[Dict[str, Any]], ids: List[str]) -> None:
        """Initialize the algorithm with provided agents and IDs"""
        start_time = time.time()

        # Initialize agent cache with dictionary-based agents
        agent_dict_cache.initialize(agents)

        self.chroma = ChromaDB(collection_name="agent_descriptions_benchmark")

        # Initialize ChromaDB if empty
        if self.chroma.get_count() == 0:
            agent_descriptions = [
                agent["description"] + "\n\n" + agent["system_prompt"]
                for agent in agents
            ]
            self.chroma.add_data(
                documents=agent_descriptions,
                ids=ids,
            )

        self.init_time = time.time() - start_time

    def select(self, query: str) -> Tuple[str, str]:
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

        best_idx = np.argmax(combined_scores)
        best_doc = documents[best_idx]
        best_agent_data = agent_values[best_doc]
        best_id = best_agent_data["object_id"]

        query_time = time.time() - start_time
        self.total_time += query_time
        self.query_count += 1

        return best_id, best_agent_data["name"]

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
    """Example usage of the benchmark"""
    benchmark = Benchmark()
    total_start_time = time.time()
    algorithm = StellaAlgorithm(benchmark.agents, benchmark.agent_ids)

    print("\nProcessing queries...")
    accuracy = benchmark.validate(algorithm, verbose=True)

    total_queries = len(benchmark.queries)
    correct_predictions = int(accuracy * total_queries)
    total_time = time.time() - total_start_time
    stats = algorithm.get_stats()

    # Print clean summary
    print("\nBenchmark Results:")
    print(f"Accuracy: {correct_predictions}/{total_queries} ({accuracy:.1%})")
    print(f"Total Runtime: {total_time:.4f} seconds")
    print(f"├─ Initialization Time: {stats['initialization_time']:.4f} seconds")
    print(f"└─ Query Processing Time: {stats['total_query_time']:.4f} seconds")
    print(f"   ├─ Number of Queries: {stats['query_count']}")
    print(f"   └─ Average Query Time: {stats['average_query_time']:.4f} seconds")


if __name__ == "__main__":
    main()
