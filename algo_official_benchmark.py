"""
Test with the benchmark from the official github repo
Accuracy: 18/24 (75.0%)
"""

import os
import time
import numpy as np
from typing import List, Dict, Tuple, Any
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
import torch

from benchmark.selection import SelectionAlgorithm
from benchmark.benchmark import Benchmark
from universa.memory.chromadb.persistent_chromadb import ChromaDB
from config.weights import (
    BASE_RATING_WEIGHT,
    RATING_RATIO_WEIGHT,
    BASE_SEMANTIC_WEIGHT,
    FIXED_LEXICAL_WEIGHT,
)
from benchmark.scoring import AgentScoring, WeightConfig
from universa.tools.result_writer import ResultWriter

console = Console()


def compute_lexical_score(query: str, description: str) -> float:
    """
    Computes normalized lexical similarity between query and description using RapidFuzz.
    
    quick description:
    - Uses token_sort_ratio to handle word order variations
    - Normalizes output to [0,1] range by dividing by 100
    - Case-insensitive comparison via .lower()
    """
    # Calculate lexical similarity score
    return fuzz.token_sort_ratio(query.lower(), description.lower()) / 100.0


class AgentCache:
    """
    Thread-safe caching mechanism for agent-related computations.
    
    quick description:
    - O(1) lookup time for agent data via hash tables
    - Pre-computed normalization of ratings and weights
    - Dynamic weight adjustment based on response history
    - Memory-efficient storage of frequently accessed metrics
    
    performance considerations:
    - Initialization is O(n) where n is number of agents
    - All subsequent lookups are O(1)
    - Memory usage is O(n) for n agents
    """

    def __init__(self):
        # Track maximum values for normalization
        self.max_rating = 0
        self.max_responses = 0
        # Lookup dictionaries for O(1) access to agent data
        self.agent_lookup = {}
        self.agent_values = {}

    def _calculate_weights(self, response_ratio: float) -> Dict[str, float]:
        """
        Calculates dynamic weights for agent scoring components.
        
        Algorithm:
        1. Increases rating weight as response_ratio increases
        2. Decreases semantic weight proportionally
        3. Maintains fixed lexical weight
        """
        rating_weight = BASE_RATING_WEIGHT + (RATING_RATIO_WEIGHT * response_ratio)
        semantic_weight = BASE_SEMANTIC_WEIGHT - (RATING_RATIO_WEIGHT * response_ratio)
        return {
            "rating_weight": rating_weight,
            "semantic_weight": semantic_weight,
            "lexical_weight": FIXED_LEXICAL_WEIGHT,
        }
        # BASE_RATING_WEIGHT, RATING_RATIO_WEIGHT, BASE_SEMANTIC_WEIGHT, FIXED_LEXICAL_WEIGHT can be edited in the config/weights.py

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
        self.max_rating = max(agent["average_rating"] for agent in agents)
        self.max_responses = max(agent["rated_responses"] for agent in agents)

        # Create lookup dictionary
        self.agent_lookup = {
            agent["description"] + "\n\n" + agent["system_prompt"]: agent
            for agent in agents
        }

        # Process agent values
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


class StellaAlgorithm(SelectionAlgorithm):
    """
    Selection algorithm using semantic search with weighted scoring.
    Combines ChromaDB vector search with multi-factor scoring including:
    - Semantic similarity
    - Historical performance (ratings)
    - Lexical matching
    """

    def __init__(self, agents: List[Dict[str, Any]], ids: List[str]) -> None:
        self.total_time = 0
        self.query_count = 0
        self.agent_cache = AgentCache()
        self.scoring = AgentScoring(WeightConfig()) 
        super().__init__(agents, ids)

    def initialize(self, agents: List[Dict[str, Any]], ids: List[str]) -> None:
        """
        Initializes the algorithm's search infrastructure.
        
        steps:
        1. Populates agent cache for quick access to normalized metrics
        2. Creates deterministic collection name using agent descriptions
        3. Initializes ChromaDB vector store if empty
        
        Args:
            agents: List of agent configurations and historical data
            ids: Unique identifiers for each agent
        """
        start_time = time.time()

        # Initialize agent cache
        self.agent_cache.initialize(agents)

        # Combine each agent's description and system prompt into a single string
        descriptions = [
            agent["description"] + "\n\n" + agent["system_prompt"] for agent in agents
        ]
        # Create a short, unique ID by hashing the combined string and taking the first 8 characters
        descriptions_str = "||".join(sorted(descriptions))
        collection_hash = hashlib.sha256(descriptions_str.encode()).hexdigest()[:8]

        # Set up ChromaDB with a unique name using the hash
        self.chroma = ChromaDB(
            collection_name=f"agent_descriptions_benchmark_{collection_hash}"
        )

        # Initialize ChromaDB if empty
        if self.chroma.get_count() == 0:
            self.chroma.add_data(
                documents=descriptions,
                ids=ids,
            )

        self.init_time = time.time() - start_time

    def select(self, query: str) -> Tuple[str, str, List[Dict]]:
        """
        Performs optimized agent selection for given query.
        
        Process flow:
        1. Vector similarity search via ChromaDB
        2. Retrieves cached agent metrics
        3. Computes weighted scores combining:
           - Semantic similarity (from vector search)
           - Historical performance
           - Lexical similarity
        4. Returns best matching agent with detailed scoring breakdown
        
        Returns:
            Tuple of (agent_id, agent_name, detailed_scoring_results)
        """
        start_time = time.time()

        # Get matches from ChromaDB
        result = self.chroma.query_data(query_text=[query])
        documents = result["documents"][0]
        distances = np.array(result["distances"][0])

        # Get agent data
        agent_data = [self.agent_cache.agent_values[doc] for doc in documents]

        # Compute scores using new scoring system
        combined_scores, detailed_results = self.scoring.compute_scores(
            query, documents, distances, agent_data
        )

        # Select best match
        best_idx = np.argmax(combined_scores)
        best_doc = documents[best_idx]
        best_agent_data = self.agent_cache.agent_values[best_doc]

        # Convert detailed_results to the expected format for backwards compatibility
        selection_details = [
            {
                "agent_name": result.agent_name,
                "distance": result.raw_distance,
                "normalized_distance": result.normalized_distance,
                "average_rating": result.average_rating,
                "normalized_rating": result.score_components.rating_score,
                "rating_weight": result.score_components.weights.rating_weight,
                "semantic_weight": result.score_components.weights.semantic_weight,
                "lexical_score": result.score_components.lexical_score,
                "combined_score": result.score_components.combined_score,
                "rated_responses": result.rated_responses,
                "lexical_weight": result.score_components.weights.lexical_weight,
            }
            for result in detailed_results
        ]

        # Update timing stats
        self.total_time += time.time() - start_time
        self.query_count += 1

        return best_agent_data["object_id"], best_agent_data["name"], selection_details

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
    """
    Benchmark execution pipeline for agent selection algorithm.
    
    Pipeline stages:
    1. Environment setup (GPU/CPU detection, output directory creation)
    2. Algorithm initialization with agent corpus
    3. Iterative query processing with progress tracking
    4. Performance metrics calculation
    5. Results persistence and reporting
    
    Output artifacts:
    - Detailed selection results for each query
    - Performance metrics (timing, accuracy)
    - Summary report
    """
    output_dir = os.path.join("output", "benchmark")
    os.makedirs(output_dir, exist_ok=True)

    benchmark = Benchmark()
    total_start_time = time.time()
    algorithm = StellaAlgorithm(benchmark.agents, benchmark.agent_ids)
    result_writer = ResultWriter()

    if torch.cuda.is_available():
        console.print(f"[green]Using GPU: {torch.cuda.get_device_name(0)}[/green]")
        torch.cuda.empty_cache()
    else:
        console.print("[yellow]GPU not available, using CPU[/yellow]")

    # Process queries with progress bar
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            "[cyan]Processing queries...", total=len(benchmark.queries)
        )

        for query in benchmark.queries:
            result_id, result_agent, details = algorithm.select(query["query"])
            is_correct = query["object_id"] == result_id

            results.append(
                {
                    "query": query["query"],
                    "predicted_agent": result_agent,
                    "expected_agent": query["agent"],
                    "is_correct": is_correct,
                    "details": result_writer.write_result(
                        query["query"],
                        result_agent,
                        query["agent"],
                        details,
                        is_correct,
                    ),
                }
            )

            progress.advance(task)

    # Calculate final metrics
    total_time = time.time() - total_start_time
    stats = algorithm.get_stats()
    correct_predictions = sum(1 for r in results if r["is_correct"])
    total_queries = len(results)

    # Write results and print summary
    result_writer.write_benchmark_results(results, output_dir, stats)
    result_writer.print_summary(output_dir, total_time, stats, correct_predictions, total_queries)


if __name__ == "__main__":
    main()
