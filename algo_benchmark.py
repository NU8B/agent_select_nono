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
from rapidfuzz import fuzz
import hashlib
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
        rating_weight = BASE_RATING_WEIGHT + (RATING_RATIO_WEIGHT * response_ratio)
        semantic_weight = BASE_SEMANTIC_WEIGHT - (RATING_RATIO_WEIGHT * response_ratio)
        return {
            "rating_weight": rating_weight,
            "semantic_weight": semantic_weight,
            "lexical_weight": FIXED_LEXICAL_WEIGHT,
        }

    def _process_agent(
        self, agent: Dict, max_rating: float, max_responses: float
    ) -> Dict:
        """Process individual agent data"""
        response_ratio = agent["rated_responses"] / max_responses
        weights = self._calculate_weights(response_ratio)

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
    def __init__(self, agents: List[Dict[str, Any]], ids: List[str]) -> None:
        self.total_time = 0
        self.query_count = 0
        self.agent_cache = AgentCache()
        self.scoring = AgentScoring(WeightConfig())  # Use default config or pass custom
        super().__init__(agents, ids)

    def initialize(self, agents: List[Dict[str, Any]], ids: List[str]) -> None:
        """Initialize the algorithm with provided agents and IDs"""
        start_time = time.time()

        # Initialize agent cache
        self.agent_cache.initialize(agents)

        # Create a unique collection name using a stable hash of the agent descriptions
        descriptions = [
            agent["description"] + "\n\n" + agent["system_prompt"] for agent in agents
        ]
        descriptions_str = "||".join(sorted(descriptions))
        collection_hash = hashlib.sha256(descriptions_str.encode()).hexdigest()[:8]

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
        """Select best agent with optimized processing"""
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
    """Main benchmark execution"""
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
    accuracy = correct_predictions / total_queries

    # Write detailed results to file
    result_writer.write_benchmark_results(results, output_dir, stats)

    # Print summary to console
    console.print(f"\n📂 Results saved to: [cyan]{output_dir}[/cyan]")
    console.print(
        f"⏱️  [bold white]Total execution time: {total_time:.2f} seconds[/bold white]"
    )
    console.print(
        f"⚙️  Initialization time: [bold yellow]{stats['initialization_time']:.2f}[/bold yellow] seconds"
    )
    console.print(
        f"🔄 Query processing time: [bold yellow]{stats['total_query_time']:.2f}[/bold yellow] seconds"
    )
    console.print(f"   ├─ Number of queries: [dim]{stats['query_count']}[/dim]")
    console.print(
        f"   ├─ Average query time: [dim]{stats['average_query_time']:.4f}[/dim] seconds"
    )
    console.print(
        f"   └─ Accuracy: [dim]{correct_predictions}/{total_queries}[/dim] ({accuracy:.1%})"
    )


if __name__ == "__main__":
    main()
