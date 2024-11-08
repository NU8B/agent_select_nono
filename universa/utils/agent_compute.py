from typing import List, Dict


class AgentComputeCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentComputeCache, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self, agents: List):
        if not self.initialized:
            # Basic agent lookups
            self.max_rating = max(agent.average_rating for agent in agents)
            self.max_responses = max(agent.rated_responses for agent in agents)
            self.agent_lookup = {agent.description: agent for agent in agents}

            # Pre-calculate normalized values for each agent
            self.agent_values = {}
            for agent in agents:
                self.agent_values[agent.description] = {
                    "normalized_rating": (
                        agent.average_rating / self.max_rating
                        if self.max_rating > 0
                        else 0
                    ),
                    "response_weight": 0.2
                    + (0.1 * (agent.rated_responses / self.max_responses)),
                    "distance_weight": 1
                    - (0.2 + (0.1 * (agent.rated_responses / self.max_responses))),
                    "rated_responses": agent.rated_responses,
                    "average_rating": agent.average_rating,
                    "name": agent.name,
                }

            self.initialized = True

    @property
    def values(self) -> Dict:
        if not self.initialized:
            raise RuntimeError("AgentComputeCache not initialized")
        return {
            "max_rating": self.max_rating,
            "max_responses": self.max_responses,
            "agent_lookup": self.agent_lookup,
            "agent_values": self.agent_values,
        }


# Global instance
agent_cache = AgentComputeCache()
