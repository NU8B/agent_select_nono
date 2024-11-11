"""
Configuration for agent selection weights.
"""

# Base weight for rating component (20-30%)
BASE_RATING_WEIGHT = 0.2

# Additional weight based on response ratio (0-10%)
RATING_RATIO_WEIGHT = 0.1

# Base weight for semantic similarity (60-70%)
BASE_SEMANTIC_WEIGHT = 0.7

# Fixed weight for lexical similarity (10%)
FIXED_LEXICAL_WEIGHT = 0.1

# Weight calculation:
# - Rating weight varies from 20% to 30% based on number of responses
# - Semantic weight varies from 70% to 60% inversely with rating weight
# - Lexical weight remains fixed at 10%
# Total always equals 100%
