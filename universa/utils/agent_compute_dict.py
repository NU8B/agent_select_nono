from typing import List, Dict


class AgentComputeDictCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentComputeDictCache, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self, agents: List[Dict]):
        if not self.initialized:
            # Basic agent lookups
            self.max_rating = max(agent["average_rating"] for agent in agents)
            self.max_responses = max(agent["rated_responses"] for agent in agents)
            self.agent_lookup = {
                agent["description"] + "\n\n" + agent["system_prompt"]: agent
                for agent in agents
            }

            # Pre-calculate normalized values for each agent
            self.agent_values = {}
            for agent in agents:
                doc_key = agent["description"] + "\n\n" + agent["system_prompt"]

                # Calculate rating weight (20-30% based on responses)
                rating_weight = 0.2 + (
                    0.1 * (agent["rated_responses"] / self.max_responses)
                )

                # Calculate semantic weight (50-60% based on inverse of rating weight)
                semantic_weight = 0.60 - (
                    rating_weight - 0.25
                )  # Adjusts to maintain total with 20% lexical

                self.agent_values[doc_key] = {
                    "normalized_rating": (
                        agent["average_rating"] / self.max_rating
                        if self.max_rating > 0
                        else 0
                    ),
                    "response_weight": rating_weight,
                    "semantic_weight": semantic_weight,  # New field
                    "rated_responses": agent["rated_responses"],
                    "average_rating": agent["average_rating"],
                    "name": agent["name"],
                    "object_id": agent["object_id"],
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
