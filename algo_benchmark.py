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

    def _compute_scores(
        self,
        query: str,
        documents: List[str],
        distances: np.ndarray,
        agent_data: List[Dict],
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Compute scores and return combined scores with details"""
        # Get pre-calculated weights and normalized values
        response_weights = np.array([data["response_weight"] for data in agent_data])
        semantic_weights = np.array([data["semantic_weight"] for data in agent_data])
        lexical_weights = np.array([data["lexical_weight"] for data in agent_data])
        normalized_ratings = np.array(
            [data["normalized_rating"] for data in agent_data]
        )

        # Calculate components
        lexical_scores = np.array(
            [compute_lexical_score(query, doc) for doc in documents]
        )
        normalized_distances = distances / distances.max()

        # Compute final scores
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

        return combined_scores, selection_details

    def select(self, query: str) -> Tuple[str, str, List[Dict]]:
        """Select best agent with optimized processing"""
        start_time = time.time()

        # Get matches from ChromaDB
        result = self.chroma.query_data(query_text=[query])
        documents = result["documents"][0]
        distances = np.array(result["distances"][0])

        # Get agent data
        cache_values = self.agent_cache.agent_values
        agent_lookup = self.agent_cache.agent_lookup
        agent_data = [cache_values[doc] for doc in documents]

        # Compute scores and get details
        combined_scores, selection_details = self._compute_scores(
            query, documents, distances, agent_data
        )

        # Select best match
        best_idx = np.argmax(combined_scores)
        best_doc = documents[best_idx]
        best_agent_data = cache_values[best_doc]
        best_id = best_agent_data["object_id"]

        # Update timing stats
        self.total_time += time.time() - start_time
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


def write_results(
    query: str,
    agent_name: str,
    expected_agent: str,
    details: List[Dict],
    is_correct: bool,
) -> str:
    """Format detailed results for a query"""
    result_text = f"\n## Query: {query}\n"
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
    return result_text


def write_benchmark_results(results: List[Dict], output_dir: str, stats: Dict) -> None:
    """Write detailed benchmark results to markdown file"""
    with open(
        os.path.join(output_dir, "benchmark_result.md"), "w", encoding="utf-8"
    ) as f:
        # Write header and summary
        f.write("# Agent Selection Results\n\n")
        f.write("## Benchmark Summary\n\n")

        total_queries = len(results)
        correct_predictions = sum(1 for r in results if r["is_correct"])
        accuracy = correct_predictions / total_queries

        f.write(f"**Accuracy**: {accuracy:.2%}\n")
        f.write(f"**Correct Predictions**: {correct_predictions}/{total_queries}\n\n")

        # Sort results into correct and incorrect
        incorrect_results = [r for r in results if not r["is_correct"]]
        correct_results = [r for r in results if r["is_correct"]]

        # Write incorrect predictions first
        if incorrect_results:
            f.write("## ❌ Incorrect Predictions\n")
            for result in incorrect_results:
                f.write(result["details"])

        # Write correct predictions second
        if correct_results:
            f.write("## ✅ Correct Predictions\n")
            for result in correct_results:
                f.write(result["details"])


def main():
    """Main benchmark execution"""
    output_dir = os.path.join("output", "benchmark")
    os.makedirs(output_dir, exist_ok=True)

    benchmark = Benchmark()
    total_start_time = time.time()
    algorithm = StellaAlgorithm(benchmark.agents, benchmark.agent_ids)

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
                    "details": write_results(
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
    write_benchmark_results(results, output_dir, stats)

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
