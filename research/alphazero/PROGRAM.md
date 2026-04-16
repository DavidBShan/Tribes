# AlphaZero Autoresearch Loop

Goal: improve the Polytopia AlphaZero-style training setup against SIMPLE and OSLA while keeping every strategy change reviewable in git.

This uses the fixed-budget experiment pattern from autoresearch: run experiments for the same wall-clock budget, compare one objective metric, keep logs, then make one strategy change at a time.

Primary score:

```text
score = min(best_simple_win_rate, best_osla_win_rate)
```

Tie breakers:

1. Latest SIMPLE win rate.
2. Latest OSLA win rate.
3. More SFT trajectory examples written.
4. Fewer incomplete games.

Concurrency rule:

- Concurrent experiments must use isolated model/data/trajectory files under their run directory.
- Do not run concurrent experiments that write to `models/` or `training/` directly.
- Promote a winning run to the shared model files only after review.

Run a concurrent 20-minute sweep:

```bash
BUDGET_SECONDS=1200 CONCURRENCY=4 ./scripts/alphazero_research_sweep.sh
```

Run continuously until interrupted:

```bash
BUDGET_SECONDS=1200 CONCURRENCY=4 ./scripts/alphazero_research_loop.sh
```

Candidate strategy changes:

- Add root Dirichlet exploration noise to PUCT priors during training/self-play.
- Add visit-count policy targets instead of only selected action type labels.
- Use temperature sampling from root visit counts early in the game, then greedy selection later.
- Bias the curriculum toward whichever baseline, SIMPLE or OSLA, is currently failing.
- Add richer exact-action features beyond action type.

Commit rule:

- Commit harness-only changes separately from agent/training-strategy changes.
- Commit every strategy code change before running the next sweep.
- Keep generated run logs out of git; use `research/alphazero/leaderboard.jsonl` and `research/alphazero/leaderboard.svg` when intentionally documenting results.
