from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from rapidfuzz import fuzz

# Import the weight configurations
from universa.utils.weights import WeightConfig, DynamicWeights

@dataclass
class ScoreComponents:
    """Holds all components that went into the final score calculation"""
    semantic_score: float
    rating_score: float
    lexical_score: float
    combined_score: float
    weights: DynamicWeights

@dataclass
class ScoringResult:
    """Complete scoring result for an agent"""
    agent_name: str
    score_components: ScoreComponents
    raw_distance: float
    normalized_distance: float
    rated_responses: int
    average_rating: float

class AgentScoring:
    def __init__(self, weight_config: WeightConfig):
        self.weight_config = weight_config

    def compute_lexical_score(self, query: str, description: str) -> float:
        """Compute normalized lexical similarity score"""
        return fuzz.token_sort_ratio(query.lower(), description.lower()) / 100.0

    def compute_semantic_score(self, distances: np.ndarray) -> np.ndarray:
        """Convert distances to similarity scores"""
        normalized_distances = distances / distances.max() if distances.max() > 0 else distances
        return 1 - normalized_distances ** 2

    def compute_rating_score(self, ratings: np.ndarray, max_rating: float) -> np.ndarray:
        """Normalize and compute rating scores"""
        return ratings / max_rating if max_rating > 0 else np.zeros_like(ratings)

    def compute_scores(
        self,
        query: str,
        documents: List[str],
        distances: np.ndarray,
        agent_data: List[Dict]
    ) -> Tuple[np.ndarray, List[ScoringResult]]:
        """Compute all scores and return combined scores with detailed results"""
        
        # Calculate weights for each agent based on their response ratio
        weights_list = [
            self.weight_config.calculate_dynamic_weights(
                agent["rated_responses"] / max(a["rated_responses"] for a in agent_data)
            )
            for agent in agent_data
        ]

        # Compute individual components
        semantic_scores = self.compute_semantic_score(distances)
        rating_scores = self.compute_rating_score(
            np.array([agent["average_rating"] for agent in agent_data]),
            max(agent["average_rating"] for agent in agent_data)
        )
        lexical_scores = np.array([
            self.compute_lexical_score(query, doc) for doc in documents
        ])

        # Compute combined scores
        combined_scores = np.zeros(len(documents))
        detailed_results = []

        for i, (agent, weights) in enumerate(zip(agent_data, weights_list)):
            # Calculate combined score
            combined_scores[i] = (
                semantic_scores[i] * weights.semantic_weight +
                rating_scores[i] * weights.rating_weight +
                lexical_scores[i] * weights.lexical_weight
            )

            # Store detailed results
            detailed_results.append(ScoringResult(
                agent_name=agent["name"],
                score_components=ScoreComponents(
                    semantic_score=semantic_scores[i],
                    rating_score=rating_scores[i],
                    lexical_score=lexical_scores[i],
                    combined_score=combined_scores[i],
                    weights=weights
                ),
                raw_distance=distances[i],
                normalized_distance=distances[i] / distances.max() if distances.max() > 0 else 0,
                rated_responses=agent["rated_responses"],
                average_rating=agent["average_rating"]
            ))

        return combined_scores, detailed_results