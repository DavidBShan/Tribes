#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def append_sweep(sweep_id: str, sweep_dir: Path, leaderboard: Path) -> list[dict]:
    existing: list[dict] = []
    seen = set()
    if leaderboard.exists():
        for line in leaderboard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing.append(row)
            seen.add((row.get("sweep_id"), row.get("variant"), row.get("run_dir")))

    new_rows = []
    for summary_path in sorted(sweep_dir.glob("*/summary.json")):
        variant = summary_path.parent.name
        summary = load_json(summary_path)
        key = (sweep_id, variant, str(summary_path.parent))
        if key in seen:
            continue
        latest = summary.get("latest_eval") or {}
        best = summary.get("best_win_rates") or {}
        row = {
            "sweep_id": sweep_id,
            "variant": variant,
            "run_dir": str(summary_path.parent),
            "git_commit": summary.get("git_commit"),
            "status": summary.get("status"),
            "score": float(summary.get("score") or 0.0),
            "best_simple": float(best.get("SIMPLE") or 0.0),
            "best_osla": float(best.get("OSLA") or 0.0),
            "latest_simple": float((latest.get("SIMPLE") or {}).get("win_rate") or 0.0),
            "latest_osla": float((latest.get("OSLA") or {}).get("win_rate") or 0.0),
            "trajectory_examples": int(((summary.get("latest_trajectory_write") or {}).get("examples") or 0)),
        }
        existing.append(row)
        new_rows.append(row)

    leaderboard.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in existing), encoding="utf-8")
    return existing


def write_tsv(rows: list[dict], out: Path) -> None:
    headers = [
        "sweep_id",
        "variant",
        "score",
        "best_simple",
        "best_osla",
        "latest_simple",
        "latest_osla",
        "trajectory_examples",
        "git_commit",
        "run_dir",
    ]
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append("\t".join(str(row.get(header, "")) for header in headers))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(rows: list[dict], out: Path) -> None:
    rows = sorted(rows, key=lambda row: (row.get("sweep_id", ""), row.get("variant", "")))[-40:]
    width = max(900, 80 + 90 * len(rows))
    height = 420
    chart_top = 45
    chart_bottom = 320
    scale = chart_bottom - chart_top

    def y(value: float) -> float:
        return chart_bottom - max(0.0, min(1.0, value)) * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="24" y="28" font-family="monospace" font-size="18" fill="#111827">AlphaZero autoresearch leaderboard</text>',
        '<line x1="52" y1="320" x2="{}" y2="320" stroke="#111827" stroke-width="1"/>'.format(width - 24),
        '<line x1="52" y1="45" x2="52" y2="320" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = y(tick)
        parts.append(f'<line x1="48" y1="{yy:.1f}" x2="{width - 24}" y2="{yy:.1f}" stroke="#d1d5db" stroke-width="1"/>')
        parts.append(f'<text x="10" y="{yy + 4:.1f}" font-family="monospace" font-size="11" fill="#374151">{tick:.2f}</text>')

    colors = {"score": "#111827", "best_simple": "#2563eb", "best_osla": "#059669"}
    for i, row in enumerate(rows):
        x = 70 + i * 90
        values = [
            ("score", float(row.get("score") or 0.0)),
            ("best_simple", float(row.get("best_simple") or 0.0)),
            ("best_osla", float(row.get("best_osla") or 0.0)),
        ]
        for j, (name, value) in enumerate(values):
            bar_x = x + j * 18
            yy = y(value)
            parts.append(f'<rect x="{bar_x}" y="{yy:.1f}" width="14" height="{chart_bottom - yy:.1f}" fill="{colors[name]}"/>')
        label = str(row.get("variant", ""))[:12]
        parts.append(f'<text x="{x - 4}" y="344" transform="rotate(35 {x - 4},344)" font-family="monospace" font-size="10" fill="#111827">{escape(label)}</text>')

    legend_x = width - 300
    for idx, (name, color) in enumerate(colors.items()):
        y0 = 24 + idx * 18
        parts.append(f'<rect x="{legend_x}" y="{y0 - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{y0}" font-family="monospace" font-size="12" fill="#111827">{name}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: alphazero_plot_research.py SWEEP_ID SWEEP_DIR LEADERBOARD_JSONL GRAPH_SVG", file=sys.stderr)
        return 2
    sweep_id = sys.argv[1]
    sweep_dir = Path(sys.argv[2])
    leaderboard = Path(sys.argv[3])
    graph = Path(sys.argv[4])
    rows = append_sweep(sweep_id, sweep_dir, leaderboard)
    write_tsv(rows, leaderboard.with_suffix(".tsv"))
    write_svg(rows, graph)
    print(f"updated {leaderboard} and {graph} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
