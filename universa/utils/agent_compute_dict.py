from typing import List, Dict


class AgentComputeDictCache:
    def __init__(self):
        self.initialized = False

    def _calculate_weights(self, response_ratio: float) -> Dict[str, float]:
        """Calculate weights based on response ratio"""
        rating_weight = 0.2 + (0.1 * response_ratio)
        semantic_weight = 0.7 - (0.1 * response_ratio)
        return {
            "rating_weight": rating_weight,
            "semantic_weight": semantic_weight,
            "lexical_weight": 0.1,  # Fixed weight
        }

    def _process_agent(
        self, agent: Dict, doc_key: str, max_rating: float, max_responses: float
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

    def initialize(self, agents: List[Dict]):
        if not self.initialized:
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
                    agent["description"] + "\n\n" + agent["system_prompt"],
                    self.max_rating,
                    self.max_responses,
                )
                for agent in agents
            }

            self.initialized = True

    @property
    def values(self) -> Dict:
        if not self.initialized:
            raise RuntimeError("AgentComputeDictCache not initialized")
        return {
            "max_rating": self.max_rating,
            "max_responses": self.max_responses,
            "agent_lookup": self.agent_lookup,
            "agent_values": self.agent_values,
        }


# Global instance
agent_dict_cache = AgentComputeDictCache()
