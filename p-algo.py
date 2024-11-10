import os
import sys
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
from rapidfuzz import fuzz  # Import RapidFuzz

# Add the project root to Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from benchmark.selection import SelectionAlgorithm
from benchmark.benchmark import Benchmark
from universa.memory.chromadb.persistent_chromadb import ChromaDB
from universa.utils.agent_compute_dict import agent_dict_cache

console = Console()

def compute_lexical_score(query: str, description: str) -> float:
    """Compute lexical similarity using RapidFuzz"""
    return fuzz.ratio(query, description) / 100.0  # Normalize to 0-1 range

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

    def select(self, query: str) -> Tuple[str, str, List[Dict]]:
        """Select best agent for the given query using optimized processing"""
        start_time = time.time()

        result = self.chroma.query_data(query_text=[query])
        documents = result["documents"][0]
        distances = np.array(result["distances"][0])

        # Use cached values from dict cache
        cache_values = agent_dict_cache.values
        agent_values = cache_values["agent_values"]

        # Get pre-calculated values for matched agents
        agent_data = [agent_values[doc] for doc in documents]

        # Calculate scores
        response_weights, semantic_weights, normalized_ratings = self._get_weights(agent_data)
        lexical_scores = self._calculate_lexical_scores(query, documents)
        combined_scores = self._compute_combined_scores(
            distances, 
            semantic_weights, 
            normalized_ratings, 
            lexical_scores, 
            response_weights,
            query,
            documents
        )

        # Create selection details
        selection_details = self._create_selection_details(agent_data, distances, combined_scores, lexical_scores)

        best_idx = np.argmax(combined_scores)
        best_doc = documents[best_idx]
        best_agent_data = agent_values[best_doc]
        best_id = best_agent_data["object_id"]

        query_time = time.time() - start_time
        self.total_time += query_time
        self.query_count += 1

        return best_id, best_agent_data["name"], selection_details

    def _get_weights(self, agent_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get pre-calculated weights and normalized values"""
        response_weights = np.array([data["response_weight"] for data in agent_data])
        semantic_weights = np.array([data["semantic_weight"] for data in agent_data])
        normalized_ratings = np.array([data["normalized_rating"] for data in agent_data])
        return response_weights, semantic_weights, normalized_ratings

    def _calculate_lexical_scores(self, query: str, documents: List[str]) -> np.ndarray:
        """Calculate lexical scores for each document"""
        return np.array([compute_lexical_score(query, doc) for doc in documents])

    def _compute_combined_scores(self, distances: np.ndarray, semantic_weights: np.ndarray,
        normalized_ratings: np.ndarray, lexical_scores: np.ndarray, response_weights: np.ndarray,
        query: str, documents: List[str]) -> np.ndarray:
        """
        Compute final scores combining all components with context-aware weighting.
        """
        # Calculate normalized distances
        normalized_distances = distances / distances.max() if distances.max() > 0 else distances
        
        # Get domain scores
        domain_scores = self._calculate_domain_specificity(query, documents)
        
        # Analyze query context
        context = self._analyze_query_context(query)
        
        # Adjust weights based on context
        if context['research'] > 0:
            semantic_weights *= 1.2
            response_weights *= 0.8
        elif context['implementation'] > 0:
            semantic_weights *= 0.9
            response_weights *= 1.1
        
        # Compute final combined scores
        return (
            (1 - normalized_distances**2) * semantic_weights * 0.5
            + normalized_ratings * response_weights * 0.25
            + lexical_scores * 0.15
            + domain_scores * 0.10
        )

    def _create_selection_details(self, agent_data: List[Dict], distances: np.ndarray, combined_scores: np.ndarray, lexical_scores: np.ndarray) -> List[Dict]:
        """Create detailed selection information for each agent"""
        return [
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
            }
            for data, dist, norm_dist, lex_score, score in zip(
                agent_data,
                distances,
                distances / distances.max(),
                lexical_scores,
                combined_scores,
            )
        ]

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

    def _calculate_domain_specificity(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Calculate domain specificity scores for each document using comprehensive domain knowledge.
        
        Args:
            query (str): The user's query
            documents (List[str]): List of agent descriptions/documents
            
        Returns:
            np.ndarray: Normalized domain specificity scores for each document
        """
        domain_scores = np.zeros(len(documents))
        
        # Comprehensive domain keyword mappings with weighted terms
        domain_keywords = {
            'web_development': {
                'high': ['frontend', 'backend', 'fullstack', 'web', 'css', 'html', 'javascript', 'react', 'angular', 'vue'],
                'medium': ['api', 'rest', 'http', 'route', 'endpoint', 'server', 'client', 'database'],
                'low': ['design', 'user', 'interface', 'responsive']
            },
            'machine_learning': {
                'high': ['neural network', 'deep learning', 'transformer', 'model', 'training', 'inference'],
                'medium': ['dataset', 'feature', 'prediction', 'accuracy', 'classification', 'regression'],
                'low': ['data', 'analysis', 'algorithm', 'optimization']
            },
            'system_architecture': {
                'high': ['distributed', 'scalable', 'microservice', 'kubernetes', 'docker', 'cloud'],
                'medium': ['deployment', 'infrastructure', 'service', 'container', 'orchestration'],
                'low': ['performance', 'monitoring', 'logging', 'security']
            },
            'game_development': {
                'high': ['unity', 'unreal', 'game engine', 'gameplay', 'physics', 'rendering'],
                'medium': ['animation', 'collision', 'input', 'multiplayer', 'networking'],
                'low': ['design', 'mechanics', 'level', 'optimization']
            },
            'quantum_computing': {
                'high': ['qubit', 'quantum gate', 'superposition', 'entanglement', 'quantum circuit'],
                'medium': ['error correction', 'quantum algorithm', 'measurement', 'coherence'],
                'low': ['simulation', 'computation', 'optimization']
            },
            'finance': {
                'high': ['portfolio', 'trading', 'investment', 'risk', 'market analysis'],
                'medium': ['stock', 'asset', 'financial', 'valuation', 'metrics'],
                'low': ['analysis', 'strategy', 'management']
            },
            'travel': {
                'high': ['itinerary', 'destination', 'travel', 'tour', 'accommodation'],
                'medium': ['culture', 'local', 'guide', 'planning', 'transportation'],
                'low': ['experience', 'recommendation', 'tips']
            },
                'ai_research': {
                'high': ['transformer', 'architecture', 'theoretical', 'research', 'paper implementation'],
                'medium': ['neural network', 'deep learning', 'machine learning'],
                'low': ['model', 'training', 'inference']
            },
            'deep_learning': {
                'high': ['pytorch', 'tensorflow', 'training pipeline', 'optimization'],
                'medium': ['neural network', 'deep learning', 'machine learning'],
                'low': ['model', 'training', 'inference']
            },
            'system_design': {
                'high': ['distributed system', 'scalable architecture', 'system design'],
                'medium': ['microservices', 'cloud', 'infrastructure'],
                'low': ['performance', 'scaling', 'deployment']
            },
            'game_systems': {
                'high': ['game engine', 'ECS', 'inventory system', 'game architecture'],
                'medium': ['unity', 'unreal', 'gameplay systems'],
                        'low': ['design', 'implementation', 'optimization']
            }
        }
        
        # Term importance weights
        weights = {
            'high': 1.0,
            'medium': 0.6,
            'low': 0.3
        }

        role_weights = {
            'Machine Learning Researcher': {
                'research_bias': 1.2,
                'implementation_bias': 0.8
    },
            'Deep Learning Engineer': {
                'research_bias': 0.8,
                'implementation_bias': 1.2
            }
        }
        
        # Preprocess query
        query_lower = query.lower()
        
        # Calculate scores for each document
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            domain_score = 0.0
            
            # Calculate score for each domain
            for domain, term_categories in domain_keywords.items():
                domain_relevance = 0.0
                
                # Check each category of terms
                for importance, terms in term_categories.items():
                    # Query relevance
                    query_matches = sum(term in query_lower for term in terms)
                    # Document matches
                    doc_matches = sum(term in doc_lower for term in terms)
                    
                    # Combine matches with weight
                    relevance = weights[importance] * (query_matches + doc_matches)
                    domain_relevance += relevance
                
                # Add domain score
                domain_score += domain_relevance
            
            domain_scores[i] = domain_score
        
        # Normalize scores
        max_score = np.max(domain_scores)
        if max_score > 0:
            domain_scores = domain_scores / max_score
        
        # Apply smoothing to prevent extreme values
        domain_scores = 0.1 + (0.9 * domain_scores)
        
        # Add role-specific adjustments
        for i, doc in enumerate(documents):
            role = self._get_role_from_doc(doc)
            if role in role_weights:
                if any(term in query.lower() for term in ['research', 'theoretical', 'paper']):
                    domain_scores[i] *= role_weights[role]['research_bias']
                if any(term in query.lower() for term in ['implement', 'build', 'develop']):
                    domain_scores[i] *= role_weights[role]['implementation_bias']
        
        return domain_scores

    def _analyze_query_context(self, query: str) -> Dict[str, float]:
        contexts = {
            'research': sum(term in query.lower() for term in ['research', 'theoretical', 'paper']),
            'implementation': sum(term in query.lower() for term in ['implement', 'build', 'develop']),
            'architecture': sum(term in query.lower() for term in ['design', 'structure', 'system']),
            'domain_specific': sum(term in query.lower() for term in ['game', 'web', 'ai', 'ml'])
        }
        return contexts

    def _get_role_from_doc(self, doc: str) -> str:
        """
        Extract the role/agent type from the document description.
        
        Args:
            doc (str): The document text (agent description + system prompt)
            
        Returns:
            str: The identified role or 'Unknown'
        """
        # Common role identifiers in descriptions
        role_patterns = {
            'Machine Learning Researcher': ['machine learning researcher', 'ml researcher', 'research scientist'],
            'Deep Learning Engineer': ['deep learning engineer', 'dl engineer', 'neural network engineer'],
            'AI Architect': ['ai architect', 'artificial intelligence architect'],
            'Python Systems Architect': ['systems architect', 'system architect', 'infrastructure architect'],
            'Full Stack Developer': ['full stack', 'fullstack', 'full-stack'],
            'Python Backend Developer': ['backend developer', 'back-end developer', 'back end'],
            'Game Tech Director': ['game tech', 'game director', 'game architect']
        }
        
        doc_lower = doc.lower()
        
        # Try to match role patterns
        for role, patterns in role_patterns.items():
            if any(pattern in doc_lower for pattern in patterns):
                return role
                
        # If no match found, try to extract from the first line
        # Assuming the role might be in the first line of the description
        first_line = doc.split('\n')[0].strip()
        if first_line:
            return first_line
            
        return 'Unknown'


def write_results(
    query: str,
    selected_agent: str,
    expected_agent: str,
    selection_details: List[Dict],
    is_correct: bool,
) -> Tuple[str, bool]:
    """Write results to a markdown file and return the result text and correctness."""
    result_text = f"## Query: {query}\n\n"
    result_text += f"**Selected Agent**: {selected_agent}\n"
    result_text += f"**Expected Agent**: {expected_agent}\n"
    result_text += f"**Correct**: {'Yes' if is_correct else 'No'}\n\n"
    result_text += "### Top 3 Agent Matches:\n\n"

    sorted_details = sorted(selection_details, key=lambda x: x["combined_score"], reverse=True)[:3]
    for detail in sorted_details:
        result_text += f"**Agent**: {detail['agent_name']}\n"
        result_text += f"- **Combined Score**: {detail['combined_score']:.4f}\n"
        result_text += f"- **Distance**: {detail['distance']:.4f}\n"
        result_text += f"- **Lexical Score**: {detail['lexical_score']:.4f}\n"
        result_text += f"- **Average Rating**: {detail['average_rating']:.2f}\n"
        result_text += f"- **Rated Responses**: {detail['rated_responses']}\n"
        result_text += f"- **Distance Weight**: {1 - detail['rating_weight'] - 0.15:.2f}\n"
        result_text += f"- **Rating Weight**: {detail['rating_weight']:.2f}\n"
        result_text += f"- **Lexical Weight**: 0.15\n\n"

    result_text += "\n---\n"
    return result_text, is_correct


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
    console.print(f"\nResults saved to: Your ASS")


if __name__ == "__main__":
    main()
