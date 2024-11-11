from dataclasses import dataclass
from typing import Optional
from config.weights import (
    BASE_RATING_WEIGHT,
    BASE_SEMANTIC_WEIGHT,
    FIXED_LEXICAL_WEIGHT,
    RATING_RATIO_WEIGHT,
)

@dataclass(frozen=True)
class WeightConfig:
    # Base weights
    base_rating_weight: float = BASE_RATING_WEIGHT  # Changed to 0.2
    base_semantic_weight: float = BASE_SEMANTIC_WEIGHT  # Changed to 0.7
    fixed_lexical_weight: float = FIXED_LEXICAL_WEIGHT  # Changed to 0.1
    
    # Adjustment factors
    rating_ratio_weight: float = RATING_RATIO_WEIGHT  # Changed to 0.1
    
    # Optional min/max bounds for safety
    min_semantic_weight: float = 0.6  # Adjusted to match new semantic range
    max_rating_weight: float = 0.3  # Adjusted to match new rating range

    def calculate_dynamic_weights(self, response_ratio: float) -> "DynamicWeights":
        """Calculate weights based on response ratio"""
        # Adjust weights based on response ratio while respecting bounds
        rating_weight = min(
            self.max_rating_weight,
            self.base_rating_weight + (self.rating_ratio_weight * response_ratio)
        )
        semantic_weight = max(
            self.min_semantic_weight,
            self.base_semantic_weight - (self.rating_ratio_weight * response_ratio)
        )
        
        return DynamicWeights(
            rating_weight=rating_weight,
            semantic_weight=semantic_weight,
            lexical_weight=self.fixed_lexical_weight
        )

@dataclass(frozen=True)
class DynamicWeights:
    """Holds the calculated weights for a specific agent"""
    rating_weight: float
    semantic_weight: float
    lexical_weight: float