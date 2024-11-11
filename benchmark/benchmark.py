import json
import os.path as op

# Get the root directory
ROOT_DIR = op.dirname(op.dirname(op.abspath(__file__)))

from benchmark.selection import SelectionAlgorithm, ExampleAlgorithm


class Benchmark:
    def __init__(self) -> None:
        self.agents = open(op.join(ROOT_DIR, "benchmark", "benchmark.json"), "r").read()
        self.agents = json.loads(self.agents)
        self.agent_ids = [agent["object_id"] for agent in self.agents]

        self.queries = open(op.join(ROOT_DIR, "benchmark", "queries.json"), "r").read()
        self.queries = json.loads(self.queries)

        self.results = []

    def validate(self, algorithm: SelectionAlgorithm, verbose: bool = True) -> float:
        total = len(self.queries)
        correct = 0

        for query in self.queries:
            result_id, result_agent = algorithm.select(query["query"])
            if query["object_id"] == result_id:
                correct += 1
                self.results.append(True)
            else:
                self.results.append(False)

            if verbose:
                print(f"\nQuery: {query['query']}")
                print(f"Selected Agent: {result_agent}")
                print(f"Expected Agent: {query['agent']}")
                print("Result: " + ("✓" if query["object_id"] == result_id else "✗"))
                print("-" * 50)

        return correct / total


if __name__ == "__main__":

    # Example usage
    benchmark = Benchmark()
    algorithm = ExampleAlgorithm(benchmark.agents, benchmark.agent_ids)
    result = benchmark.validate(algorithm, verbose=True)
    print(result)
