import os
from typing import List, Dict

class ResultWriter:
    """Handles writing benchmark results to files"""

    @staticmethod
    def write_result(
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

        sorted_details = sorted(details, key=lambda x: x["combined_score"], reverse=True)[:3]
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

    @staticmethod
    def write_benchmark_results(results: List[Dict], output_dir: str, stats: Dict) -> None:
        """Write detailed benchmark results to markdown file"""
        os.makedirs(output_dir, exist_ok=True)
        
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