#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${1:-baseline-current}"
BUDGET_SECONDS="${BUDGET_SECONDS:-1200}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${RUN_NAME}}"
RUN_DIR="${RUN_DIR:-research/alphazero/runs/${RUN_ID}}"
mkdir -p "$RUN_DIR"

LOG="$RUN_DIR/train.log"
SUMMARY="$RUN_DIR/summary.json"
STATUS_FILE="$RUN_DIR/status.txt"

SEARCH_CALLS="${SEARCH_CALLS:-400}"
OPPONENT_CALLS="${OPPONENT_CALLS:-500}"
DEPTH="${DEPTH:-14}"
GAMES="${GAMES:-10}"
EVAL_GAMES="${EVAL_GAMES:-4}"
MAX_TURNS="${MAX_TURNS:-50}"
EPOCHS="${EPOCHS:-3}"
POLICY_EPOCHS="${POLICY_EPOCHS:-8}"
MAX_PROMPT_ACTIONS="${MAX_PROMPT_ACTIONS:-96}"
MAX_TRAJECTORIES_PER_GAME="${MAX_TRAJECTORIES_PER_GAME:-240}"
TRAJECTORY_SAMPLE="${TRAJECTORY_SAMPLE:-1.0}"
ITERATIONS="${ITERATIONS:-500}"
SELF_PLAY_ONLY="${SELF_PLAY_ONLY:-false}"
SELF_PLAY_AFTER_TARGET="${SELF_PLAY_AFTER_TARGET:-true}"
CONTINUE_AFTER_TARGET="${CONTINUE_AFTER_TARGET:-true}"
ISOLATED="${ISOLATED:-true}"

if [[ "$ISOLATED" == "true" ]]; then
  mkdir -p "$RUN_DIR/models" "$RUN_DIR/training"
  cp models/alphazero-value.tsv "$RUN_DIR/models/alphazero-value.tsv"
  cp models/alphazero-policy.tsv "$RUN_DIR/models/alphazero-policy.tsv"
  cp training/alphazero-value-data.tsv "$RUN_DIR/training/alphazero-value-data.tsv"
  cp training/alphazero-policy-data.tsv "$RUN_DIR/training/alphazero-policy-data.tsv"
  MODEL_PATH="$RUN_DIR/models/alphazero-value.tsv"
  POLICY_PATH="$RUN_DIR/models/alphazero-policy.tsv"
  DATA_PATH="$RUN_DIR/training/alphazero-value-data.tsv"
  POLICY_DATA_PATH="$RUN_DIR/training/alphazero-policy-data.tsv"
  TRAJECTORY_PATH="$RUN_DIR/training/alphazero-sft-trajectories.jsonl"
else
  MODEL_PATH="${MODEL_PATH:-models/alphazero-value.tsv}"
  POLICY_PATH="${POLICY_PATH:-models/alphazero-policy.tsv}"
  DATA_PATH="${DATA_PATH:-training/alphazero-value-data.tsv}"
  POLICY_DATA_PATH="${POLICY_DATA_PATH:-training/alphazero-policy-data.tsv}"
  TRAJECTORY_PATH="${TRAJECTORY_PATH:-training/alphazero-sft-trajectories.jsonl}"
fi

{
  echo "run_id=${RUN_ID}"
  echo "run_name=${RUN_NAME}"
  echo "budget_seconds=${BUDGET_SECONDS}"
  echo "git_commit=$(git rev-parse --short HEAD)"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "search_calls=${SEARCH_CALLS}"
  echo "opponent_calls=${OPPONENT_CALLS}"
  echo "depth=${DEPTH}"
  echo "games=${GAMES}"
  echo "eval_games=${EVAL_GAMES}"
  echo "max_turns=${MAX_TURNS}"
  echo "self_play_only=${SELF_PLAY_ONLY}"
  echo "isolated=${ISOLATED}"
} > "$RUN_DIR/config.env"

javac -cp lib/json.jar -d build $(find src -name '*.java')

set +e
timeout "${BUDGET_SECONDS}s" java -cp build:lib/json.jar players.alphazero.AlphaZeroTrainer \
  --iterations "$ITERATIONS" \
  --games "$GAMES" \
  --eval-games "$EVAL_GAMES" \
  --search-calls "$SEARCH_CALLS" \
  --opponent-calls "$OPPONENT_CALLS" \
  --depth "$DEPTH" \
  --max-turns "$MAX_TURNS" \
  --epochs "$EPOCHS" \
  --policy-epochs "$POLICY_EPOCHS" \
  --continue-after-target "$CONTINUE_AFTER_TARGET" \
  --self-play-after-target "$SELF_PLAY_AFTER_TARGET" \
  --self-play-only "$SELF_PLAY_ONLY" \
  --self-play-games "$GAMES" \
  --model "$MODEL_PATH" \
  --policy "$POLICY_PATH" \
  --data "$DATA_PATH" \
  --policy-data "$POLICY_DATA_PATH" \
  --trajectory-data "$TRAJECTORY_PATH" \
  --max-prompt-actions "$MAX_PROMPT_ACTIONS" \
  --max-trajectories-per-game "$MAX_TRAJECTORIES_PER_GAME" \
  --trajectory-sample "$TRAJECTORY_SAMPLE" \
  2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

echo "$status" > "$STATUS_FILE"
python3 scripts/alphazero_summarize_run.py "$LOG" "$SUMMARY" "$status"

if [[ "$status" -eq 124 ]]; then
  echo "Run reached fixed budget: ${BUDGET_SECONDS}s"
  exit 0
fi
exit "$status"
