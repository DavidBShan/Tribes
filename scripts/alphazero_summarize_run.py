#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from pathlib import Path


EVAL_RE = re.compile(
    r"eval AZ vs (?P<opponent>\S+): W=(?P<wins>\d+) L=(?P<losses>\d+) I=(?P<incomplete>\d+) "
    r"N=(?P<n>\d+) winRate=(?P<rate>[0-9.]+)"
)
VALUE_RE = re.compile(r"trained value on (?P<examples>\d+) examples; mse=(?P<loss>[0-9.]+)")
POLICY_RE = re.compile(r"trained policy on (?P<examples>\d+) examples; xent=(?P<loss>[0-9.]+)")
TRAJ_RE = re.compile(r"wrote (?P<examples>\d+) SFT trajectory examples to (?P<path>\S+) \((?P<skipped>\d+) skipped\)")


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: alphazero_summarize_run.py TRAIN_LOG SUMMARY_JSON [STATUS]", file=sys.stderr)
        return 2

    log_path = Path(sys.argv[1])
    summary_path = Path(sys.argv[2])
    status = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    repo = Path(__file__).resolve().parents[1]

    evals: dict[str, list[dict[str, float | int]]] = {}
    value_training: list[dict[str, float | int]] = []
    policy_training: list[dict[str, float | int]] = []
    trajectories: list[dict[str, float | int | str]] = []

    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = EVAL_RE.search(line)
            if match:
                item = {
                    "wins": int(match.group("wins")),
                    "losses": int(match.group("losses")),
                    "incomplete": int(match.group("incomplete")),
                    "n": int(match.group("n")),
                    "win_rate": float(match.group("rate")),
                }
                evals.setdefault(match.group("opponent"), []).append(item)
                continue

            match = VALUE_RE.search(line)
            if match:
                value_training.append({
                    "examples": int(match.group("examples")),
                    "mse": float(match.group("loss")),
                })
                continue

            match = POLICY_RE.search(line)
            if match:
                policy_training.append({
                    "examples": int(match.group("examples")),
                    "xent": float(match.group("loss")),
                })
                continue

            match = TRAJ_RE.search(line)
            if match:
                trajectories.append({
                    "examples": int(match.group("examples")),
                    "path": match.group("path"),
                    "skipped": int(match.group("skipped")),
                })

    latest = {opponent: rows[-1] for opponent, rows in evals.items() if rows}
    best = {
        opponent: max((float(row["win_rate"]) for row in rows), default=0.0)
        for opponent, rows in evals.items()
    }
    score = min(best.get("SIMPLE", 0.0), best.get("OSLA", 0.0))

    summary = {
        "status": status,
        "git_commit": git_commit(repo),
        "log": str(log_path),
        "score": score,
        "best_win_rates": best,
        "latest_eval": latest,
        "num_eval_rounds": {opponent: len(rows) for opponent, rows in evals.items()},
        "latest_value_training": value_training[-1] if value_training else None,
        "latest_policy_training": policy_training[-1] if policy_training else None,
        "latest_trajectory_write": trajectories[-1] if trajectories else None,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
