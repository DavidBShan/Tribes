#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

while true; do
  SWEEP_ID="$(date -u +%Y%m%dT%H%M%SZ)" ./scripts/alphazero_research_sweep.sh
done
