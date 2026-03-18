"""Result analysis utility for WindowsAgentArena evaluations.

Reads ``result.txt`` from each task directory and computes per-domain and
overall success rates.

Usage::

    python show_results.py --result_dir ./results --model gpt-4o \
        --observation_type screenshot
"""

import argparse
import os
from collections import defaultdict


def show_results(result_dir: str, model: str, observation_type: str) -> None:
    target_dir = os.path.join(result_dir, "pyautogui", observation_type, model)
    if not os.path.exists(target_dir):
        print(f"Result directory not found: {target_dir}")
        return

    domain_results: dict = defaultdict(list)

    for domain in sorted(os.listdir(target_dir)):
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        for example_id in sorted(os.listdir(domain_path)):
            example_path = os.path.join(domain_path, example_id)
            if not os.path.isdir(example_path):
                continue
            result_file = os.path.join(example_path, "result.txt")
            if os.path.isfile(result_file):
                try:
                    score = float(open(result_file).read().strip())
                except (ValueError, OSError):
                    score = 0.0
                domain_results[domain].append(score)

    if not domain_results:
        print("No results found.")
        return

    # Print per-domain results
    print(f"\n{'Domain':<30} {'Tasks':>6} {'Success Rate':>14}")
    print("-" * 54)

    all_scores = []
    for domain in sorted(domain_results):
        scores = domain_results[domain]
        all_scores.extend(scores)
        rate = sum(scores) / len(scores) * 100 if scores else 0.0
        print(f"{domain:<30} {len(scores):>6} {rate:>13.2f}%")

    print("-" * 54)
    overall = sum(all_scores) / len(all_scores) * 100 if all_scores else 0.0
    print(f"{'Overall':<30} {len(all_scores):>6} {overall:>13.2f}%")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show WAA evaluation results")
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--observation_type", type=str, default="screenshot")
    args = parser.parse_args()
    show_results(args.result_dir, args.model, args.observation_type)
