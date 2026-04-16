#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SWEEP_ID="${SWEEP_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE_RUN_DIR="${BASE_RUN_DIR:-research/alphazero/runs/${SWEEP_ID}}"
CONCURRENCY="${CONCURRENCY:-4}"
BUDGET_SECONDS="${BUDGET_SECONDS:-1200}"
VARIANTS_CSV="${VARIANTS:-baseline,mixed-fast,selfplay-only,deep-search}"
IFS=',' read -r -a VARIANTS_ARRAY <<< "$VARIANTS_CSV"
mkdir -p "$BASE_RUN_DIR"

run_variant() {
  local variant="$1"
  local dir="$BASE_RUN_DIR/$variant"
  mkdir -p "$dir"
  case "$variant" in
    baseline)
      RUN_ID="${SWEEP_ID}-${variant}" RUN_DIR="$dir" BUDGET_SECONDS="$BUDGET_SECONDS" \
        ./scripts/alphazero_autoresearch.sh "$variant"
      ;;
    mixed-fast)
      RUN_ID="${SWEEP_ID}-${variant}" RUN_DIR="$dir" BUDGET_SECONDS="$BUDGET_SECONDS" \
        SEARCH_CALLS=250 OPPONENT_CALLS=350 DEPTH=10 GAMES=16 EVAL_GAMES=4 \
        ./scripts/alphazero_autoresearch.sh "$variant"
      ;;
    selfplay-only)
      RUN_ID="${SWEEP_ID}-${variant}" RUN_DIR="$dir" BUDGET_SECONDS="$BUDGET_SECONDS" \
        SELF_PLAY_ONLY=true SEARCH_CALLS=400 OPPONENT_CALLS=500 DEPTH=14 GAMES=12 EVAL_GAMES=4 \
        ./scripts/alphazero_autoresearch.sh "$variant"
      ;;
    deep-search)
      RUN_ID="${SWEEP_ID}-${variant}" RUN_DIR="$dir" BUDGET_SECONDS="$BUDGET_SECONDS" \
        SEARCH_CALLS=700 OPPONENT_CALLS=700 DEPTH=18 GAMES=6 EVAL_GAMES=4 \
        ./scripts/alphazero_autoresearch.sh "$variant"
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      return 2
      ;;
  esac
}

active=0
pids=()
for variant in "${VARIANTS_ARRAY[@]}"; do
  run_variant "$variant" &
  pids+=("$!")
  active=$((active + 1))
  if [[ "$active" -ge "$CONCURRENCY" ]]; then
    wait -n || true
    active=$((active - 1))
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

python3 scripts/alphazero_plot_research.py \
  "$SWEEP_ID" \
  "$BASE_RUN_DIR" \
  research/alphazero/leaderboard.jsonl \
  research/alphazero/leaderboard.svg
