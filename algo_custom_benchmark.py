"""
This also use the same algorithm as the official benchmark, but with our custom benchmark.
Accuracy: 66/69 (95.7%)
"""

import os
import json
import torch
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
import numpy as np
from rapidfuzz import fuzz # Calculate lexical similarity
import hashlib # Create a short, unique ID
import functools # Cache agent data for efficiency

from universa.memory.chromadb.persistent_chromadb import ChromaDB
from data.test_data import (
    QUERY_AGENT_MAPPING,
    get_benchmark_metrics,
    get_detailed_results,
)
from config.weights import (
    BASE_RATING_WEIGHT,
    RATING_RATIO_WEIGHT,
    BASE_SEMANTIC_WEIGHT,
    FIXED_LEXICAL_WEIGHT,
)

console = Console()


@functools.lru_cache(maxsize=1)
def load_agents() -> List:
    """Load agent definitions with memory-efficient processing.
    Uses LRU cache to prevent repeated file reads and dynamic type creation
    for optimal memory usage when dealing with multiple agent definitions."""
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


def compute_lexical_score(query: str, description: str) -> float:
    """Compute normalized lexical similarity score using fuzzy string matching.
    Returns a value between 0 and 1, where 1 indicates perfect match.
    Uses token_sort_ratio to handle word order differences."""
    return fuzz.token_sort_ratio(query.lower(), description.lower()) / 100.0


class AgentCache:
    """Internal agent cache for the algorithm that stores pre-computed values and weights
    to avoid redundant calculations during agent selection"""

    def __init__(self):
        self.max_rating = 0
        self.max_responses = 0
        self.agent_lookup = {}
        self.agent_values = {}

    def _calculate_weights(self, response_ratio: float) -> Dict[str, float]:
        """Calculate dynamic weights based on response ratio (number of rated responses)
        - Higher response ratio increases rating weight and decreases semantic weight
        - Lexical weight remains fixed"""
        rating_weight = BASE_RATING_WEIGHT + (RATING_RATIO_WEIGHT * response_ratio)
        semantic_weight = BASE_SEMANTIC_WEIGHT - (RATING_RATIO_WEIGHT * response_ratio)
        return {
            "rating_weight": rating_weight,
            "semantic_weight": semantic_weight,
            "lexical_weight": FIXED_LEXICAL_WEIGHT,
        }

    def _process_agent(self, agent, max_rating: float, max_responses: float) -> Dict:
        """Process individual agent data"""
        response_ratio = agent.rated_responses / max_responses
        weights = self._calculate_weights(response_ratio)

        return {
            "normalized_rating": (
                agent.average_rating / max_rating if max_rating > 0 else 0
            ),
            "response_weight": weights["rating_weight"],
            "semantic_weight": weights["semantic_weight"],
            "lexical_weight": weights["lexical_weight"],
            "rated_responses": agent.rated_responses,
            "average_rating": agent.average_rating,
            "name": agent.name,
        }

    def initialize(self, agents: List) -> None:
        """Initialize cache with agent data"""
        self.max_rating = max(agent.average_rating for agent in agents)
        self.max_responses = max(agent.rated_responses for agent in agents)

        # Create lookup dictionary
        self.agent_lookup = {agent.description: agent for agent in agents}

        # Process agent values
        self.agent_values = {
            agent.description: self._process_agent(
                agent,
                self.max_rating,
                self.max_responses,
            )
            for agent in agents
        }

    @property
    def values(self) -> Dict:
        """Get cached values"""
        return {
            "max_rating": self.max_rating,
            "max_responses": self.max_responses,
            "agent_lookup": self.agent_lookup,
            "agent_values": self.agent_values,
        }


class StellaTestAlgorithm:
    def __init__(self):
        self.total_time = 0
        self.query_count = 0
        self.agent_cache = AgentCache()
        self.initialize()

    def initialize(self):
        """Initialize the algorithm with agents and ChromaDB.
        - Loads and caches agent data
        - Creates a unique collection name based on agent descriptions
        - Initializes ChromaDB for vector similarity search
        - Measures initialization time for performance tracking"""
        start_time = time.time()

        # Load agents and initialize cache
        self.agents = load_agents()
        self.agent_cache.initialize(self.agents)

        # Create unique collection name
        descriptions = [agent.description for agent in self.agents]
        descriptions_str = "||".join(sorted(descriptions))
        collection_hash = hashlib.sha256(descriptions_str.encode()).hexdigest()[:8]

        # Initialize ChromaDB
        self.chroma = ChromaDB(
            collection_name=f"agent_descriptions_test_{collection_hash}"
        )

        # Initialize ChromaDB if empty
        if self.chroma.get_count() == 0:
            self.chroma.add_data(
                documents=descriptions,
                ids=[agent.name for agent in self.agents],
            )

        self.init_time = time.time() - start_time

    def _compute_scores(
        self,
        query: str,
        documents: List[str],
        distances: np.ndarray,
        agent_data: List[Dict],
        agents_list: List,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Compute final scores using a weighted combination of three factors:
        1. Semantic similarity (from ChromaDB embeddings) - weighted by semantic_weight
        2. Agent rating performance - weighted by response_weight
        3. Lexical similarity (fuzzy string matching) - weighted by lexical_weight
        
        The weights are dynamically adjusted based on how many rated responses an agent has"""
        # Get pre-calculated weights and normalized values
        response_weights = np.array([data["response_weight"] for data in agent_data])
        semantic_weights = np.array([data["semantic_weight"] for data in agent_data])
        lexical_weights = np.array([data["lexical_weight"] for data in agent_data])
        normalized_ratings = np.array(
            [data["normalized_rating"] for data in agent_data]
        )

        # Calculate components
        lexical_scores = np.array(
            [compute_lexical_score(query, agent.description) for agent in agents_list]
        )
        normalized_distances = distances / distances.max()

        # Final score combines three components:
        # 1. Semantic score: (1 - normalized_distances²) gives higher score for smaller distances
        # 2. Rating score: normalized_ratings weighted by response confidence
        # 3. Lexical score: direct string similarity
        combined_scores = (
            (1 - normalized_distances**2) * semantic_weights
            + normalized_ratings * response_weights
            + lexical_scores * lexical_weights
        )

        # Create selection details
        selection_details = [
            {
                "agent_name": agent.name,
                "distance": float(dist),
                "normalized_distance": float(norm_dist),
                "average_rating": float(agent.average_rating),
                "normalized_rating": float(norm_rating),
                "rating_weight": float(r_weight),
                "semantic_weight": float(s_weight),
                "lexical_weight": float(l_weight),
                "lexical_score": float(lex_score),
                "combined_score": float(score),
                "rated_responses": agent.rated_responses,
            }
            for agent, dist, norm_dist, norm_rating, r_weight, s_weight, l_weight, lex_score, score in zip(
                agents_list,
                distances,
                normalized_distances,
                normalized_ratings,
                response_weights,
                semantic_weights,
                lexical_weights,
                lexical_scores,
                combined_scores,
            )
        ]

        return combined_scores, selection_details

    def select(self, query: str) -> Tuple[object, List[Dict]]:
        """Select best agent with optimized processing.
        
        Process:
        1. Query ChromaDB for semantic similarity matches
        2. Retrieve pre-calculated agent data from cache
        3. Compute combined scores using semantic, rating, and lexical factors
        4. Return best matching agent and detailed selection metrics
        
        Returns:
            Tuple containing (best_agent, selection_details)"""
        start_time = time.time()

        # Get matches from ChromaDB
        result = self.chroma.query_data(query_text=[query])
        documents = result["documents"][0]
        distances = np.array(result["distances"][0])

        # Get agent data directly from cache properties
        agent_lookup = self.agent_cache.agent_lookup
        agent_values = self.agent_cache.agent_values

        # Get pre-calculated values for matched agents
        agent_data = [agent_values[doc] for doc in documents]
        agents_list = [agent_lookup[doc] for doc in documents]

        # Compute scores and get details
        combined_scores, selection_details = self._compute_scores(
            query, documents, distances, agent_data, agents_list
        )

        # Select best match
        best_idx = np.argmax(combined_scores)
        best_agent = agents_list[best_idx]

        # Update timing stats
        self.total_time += time.time() - start_time
        self.query_count += 1

        return best_agent, selection_details

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


def write_results(query: str, selected_agent: object, details: List[Dict]) -> str:
    """Write detailed results for each query in markdown format.
    
    Includes:
    - Selected agent name
    - Top 3 agent matches with detailed scoring breakdown
    - Component scores (combined, distance, lexical)
    - Agent statistics (ratings, responses)
    - Weight distributions used in selection"""
    result_text = f"**Selected Agent**: {selected_agent.name}\n"
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


def write_benchmark_results(predictions: dict, output_dir: str) -> None:
    """Write benchmark results to output files in markdown format.
    Organizes results into two sections:
    1. Incorrect predictions - Shows where the algorithm made mistakes
    2. Correct predictions - Shows successful matches
    
    Each prediction includes detailed scoring breakdown and weights used"""
    metrics = get_benchmark_metrics(
        {k: v["predicted_agent"] for k, v in predictions.items()}
    )
    detailed_results = get_detailed_results(
        {k: v["predicted_agent"] for k, v in predictions.items()}
    )

    with open(os.path.join(output_dir, "test_result.md"), "w", encoding="utf-8") as f:
        # Write header and summary
        f.write("# Agent Selection Results - Test\n\n")
        f.write("## Benchmark Summary\n\n")
        f.write(f"**Accuracy**: {metrics['accuracy']:.2%}\n")
        f.write(
            f"**Correct Predictions**: {metrics['correct_predictions']}/{metrics['total_queries']}\n\n"
        )

        # Sort results
        incorrect_results = []
        correct_results = []

        for result in detailed_results:
            query = result["query"]
            result["details"] = predictions[query]["details"]
            if not result["is_correct"]:
                incorrect_results.append(result)
            else:
                correct_results.append(result)

        # Write results sections
        if incorrect_results:
            f.write("## ❌ Incorrect Predictions\n\n")
            for result in incorrect_results:
                f.write(f"## Query:{result['query']}\n")
                f.write(
                    f"**Benchmark**: [INCORRECT - Expected: {result['correct_agent']}]**\n\n"
                )
                f.write(result["details"])
                f.write("\n---\n\n")

        if correct_results:
            f.write("## ✅ Correct Predictions\n\n")
            for result in correct_results:
                f.write(f"## Query:{result['query']}\n")
                f.write("**Benchmark**: [CORRECT]**\n\n")
                f.write(result["details"])
                f.write("\n---\n\n")


def main():
    """Main execution function that:
    1. Initializes the algorithm and GPU if available
    2. Processes all test queries with progress tracking
    3. Writes detailed results to output directory
    4. Displays performance metrics including timing and accuracy"""
    # Setup
    output_dir = "output/test"
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()
    algorithm = StellaTestAlgorithm()

    # Pre-warm model if using GPU
    if torch.cuda.is_available():
        console.print(f"[green]Using GPU: {torch.cuda.get_device_name(0)}[/green]")
        torch.cuda.empty_cache()
        _ = algorithm.chroma.query_data(query_text=["warmup query"])
    else:
        console.print("[yellow]GPU not available, using CPU[/yellow]")

    # Process queries
    predictions = {}
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

        for query in QUERY_AGENT_MAPPING:
            best_agent, selection_details = algorithm.select(query)
            predictions[query] = {
                "predicted_agent": best_agent.name,
                "details": write_results(query, best_agent, selection_details),
            }
            progress.advance(task)

    # Write results and print summary
    write_benchmark_results(predictions, output_dir)

    # Calculate metrics
    total_time = time.time() - total_start
    stats = algorithm.get_stats()
    metrics = get_benchmark_metrics(
        {k: v["predicted_agent"] for k, v in predictions.items()}
    )

    # Print summary
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
        f"   └─ Accuracy: [dim]{metrics['correct_predictions']}/{metrics['total_queries']}[/dim] ({metrics['accuracy']:.1%})"
    )


if __name__ == "__main__":
    main()
