# Test with the benchmark from the official github repo
# Accuracy: 18/24 (75.0%)

import os
import sys
import time
import numpy as np
from typing import List, Dict, Tuple, Any
<<<<<<< Updated upstream
import time

# Add the project root to Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
=======
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rapidfuzz import fuzz # Calculate lexical similarity
import hashlib # Create a short, unique ID
import torch # Use PyTorch for GPU acceleration
>>>>>>> Stashed changes

from benchmark.selection import SelectionAlgorithm 
from benchmark.benchmark import Benchmark
from universa.memory.chromadb.persistent_chromadb import ChromaDB
<<<<<<< Updated upstream
from universa.utils.agent_compute_dict import agent_dict_cache
=======
from config.weights import ( 
    BASE_RATING_WEIGHT,
    RATING_RATIO_WEIGHT,
    BASE_SEMANTIC_WEIGHT,
    FIXED_LEXICAL_WEIGHT,
)

console = Console()


def compute_lexical_score(query: str, description: str) -> float:
    """Compute normalized lexical similarity score"""
    return fuzz.token_sort_ratio(query.lower(), description.lower()) / 100.0


class AgentCache:
    """Internal agent cache for the algorithm"""

    def __init__(self):
        self.max_rating = 0
        self.max_responses = 0
        self.agent_lookup = {}
        self.agent_values = {}

    def _calculate_weights(self, response_ratio: float) -> Dict[str, float]:
        """Calculate weights based on response ratio"""
        # Increase the rating weight proportionally with the response ratio
        rating_weight = BASE_RATING_WEIGHT + (RATING_RATIO_WEIGHT * response_ratio)
        # Decrease the semantic weight as the response ratio increases
        semantic_weight = BASE_SEMANTIC_WEIGHT - (RATING_RATIO_WEIGHT * response_ratio)
        # The lexical weight is fixed since it does not depend on the response ratio
        return {
            "rating_weight": rating_weight,
            "semantic_weight": semantic_weight,
            "lexical_weight": FIXED_LEXICAL_WEIGHT,
        }
        # BASE_RATING_WEIGHT, RATING_RATIO_WEIGHT, BASE_SEMANTIC_WEIGHT, FIXED_LEXICAL_WEIGHT can be edit in the config/weights.py

    def _process_agent(
        self, agent: Dict, max_rating: float, max_responses: float
    ) -> Dict:
        """Process individual agent data"""
        # Calculate the response ratio, which indicates how active or popular the agent is
        response_ratio = agent["rated_responses"] / max_responses
        # Get the rating, semantic, and lexical weights based on the response ratio
        weights = self._calculate_weights(response_ratio)

        # Normalize the agent's average rating by dividing it by the maximum rating
        return {
            "normalized_rating": (
                agent["average_rating"] / max_rating if max_rating > 0 else 0
            ),
            "response_weight": weights["rating_weight"],
            "semantic_weight": weights["semantic_weight"],
            "lexical_weight": weights["lexical_weight"],
            "rated_responses": agent["rated_responses"],
            "average_rating": agent["average_rating"],
            "name": agent["name"],
            "object_id": agent["object_id"],
        }

    def initialize(self, agents: List[Dict]) -> None:
        """Initialize cache with agent data"""
        # Determine the maximum rating and responses among all agents
        self.max_rating = max(agent["average_rating"] for agent in agents)
        self.max_responses = max(agent["rated_responses"] for agent in agents)

        # Create lookup dictionary for agents using their description and system prompt
        self.agent_lookup = {
            agent["description"] + "\n\n" + agent["system_prompt"]: agent
            for agent in agents
        }

        # Process agent values and store them in a dictionary
        self.agent_values = {
            agent["description"]
            + "\n\n"
            + agent["system_prompt"]: self._process_agent(
                agent,
                self.max_rating,
                self.max_responses,
            )
            for agent in agents
        }
>>>>>>> Stashed changes


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

<<<<<<< Updated upstream
        self.chroma = ChromaDB(collection_name="agent_descriptions_benchmark")
=======
        # Create a unique collection name using a stable hash of the agent descriptions
        descriptions = [
            agent["description"] + "\n\n" + agent["system_prompt"] for agent in agents
        ]
        # Combine each agent's description and system prompt into a single string
        descriptions_str = "||".join(sorted(descriptions))
        # Create a short, unique ID by hashing the combined string and taking the first 8 characters
        collection_hash = hashlib.sha256(descriptions_str.encode()).hexdigest()[:8]

        # Set up ChromaDB with a unique name using the hash
        self.chroma = ChromaDB(
            collection_name=f"agent_descriptions_benchmark_{collection_hash}"
        )
>>>>>>> Stashed changes

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

        # Get pre-calculated weights
        response_weights = np.array([data["response_weight"] for data in agent_data])
        distance_weights = np.array([data["distance_weight"] for data in agent_data])
        normalized_ratings = np.array(
            [data["normalized_rating"] for data in agent_data]
        )

        # Normalize distances
        normalized_distances = distances / distances.max()

        # Compute final scores
        combined_scores = (
            1 - normalized_distances**2
        ) * distance_weights + normalized_ratings * response_weights

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
