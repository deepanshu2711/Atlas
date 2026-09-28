"""Compare two full-eval reports: accuracy gained against cost added.

Usage:
    uv run python evals/compare.py <baseline.json> <candidate.json>

For each question type, prints the accuracy delta (in points) next to how
many times more LLM calls, tokens and latency the candidate spent, and
"points per 1x extra tokens" so a big cost for a small gain stands out.
Both reports must come from a run_eval.py that records cost per question.
"""
import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    report = json.loads(Path(path).read_text())
    if report["overall"].get("mean_tokens") is None:
        sys.exit(f"{path} has no cost data; re-run it with the current run_eval.py")
    return report


def ratio(candidate, baseline):
    return candidate / baseline if candidate is not None and baseline else None


def fmt(value, spec, suffix=""):
    return "-" if value is None else format(value, spec) + suffix


def row(name: str, base: dict, cand: dict) -> str:
    points = (cand["accuracy"] - base["accuracy"]) * 100
    token_x = ratio(cand["mean_tokens"], base["mean_tokens"])
    # accuracy points bought per extra multiple of token spend
    per_x = points / (token_x - 1) if token_x and token_x > 1.001 else None
    return (f"{name:30s}{cand['total']:3d} {base['accuracy']:6.1%} {cand['accuracy']:6.1%} "
            f"{points:+6.1f} "
            f"{fmt(ratio(cand['mean_llm_calls'], base['mean_llm_calls']), '6.2f', 'x')} "
            f"{fmt(token_x, '6.2f', 'x')} "
            f"{fmt(ratio(cand['mean_latency_s'], base['mean_latency_s']), '6.2f', 'x')} "
            f"{fmt(per_x, '+7.1f')}")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    base, cand = load(sys.argv[1]), load(sys.argv[2])
    print(f"baseline:  {sys.argv[1]}  {base.get('config', {})}")
    print(f"candidate: {sys.argv[2]}  {cand.get('config', {})}\n")
    print(f"{'':30s}{'n':>3s} {'base':>6s} {'cand':>6s} {'Δpts':>6s} "
          f"{'calls':>7s} {'tokens':>7s} {'lat':>7s} {'pts/1x':>7s}")
    print(row("overall", base["overall"], cand["overall"]))
    for name in sorted(set(base["by_type"]) & set(cand["by_type"])):
        print(row(name, base["by_type"][name], cand["by_type"][name]))

    # questions whose outcome flipped, to see what the extra spend bought
    base_by_id = {r["id"]: r for r in base["results"]}
    gained = [r["id"] for r in cand["results"] if r["passed"] and not base_by_id.get(r["id"], {}).get("passed")]
    lost = [r["id"] for r in cand["results"] if not r["passed"] and base_by_id.get(r["id"], {}).get("passed")]
    print(f"\nfixed by candidate ({len(gained)}): {', '.join(gained) or '-'}")
    print(f"broken by candidate ({len(lost)}): {', '.join(lost) or '-'}")


if __name__ == "__main__":
    main()
