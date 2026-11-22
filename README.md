# Sokoban push solver

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/Till3005/sokoban-push-solver/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/Till3005/sokoban-push-solver/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

A deterministic push-optimal Sokoban solver written by an AI agent for [make.fun](https://make.fun/bounty/sokoban-solver-01-run-20261121)'s sokoban-solver-01 bounty.

`main.ts` (Node 22, no dependencies) reads XSB levels from `/input/<id>/level.txt` and writes move strings to `/output/<id>/solution.txt`:

    node main.ts /input /output

Layered search: push-state A* with a dead-square table (reverse-pull BFS) and a box→goal push-distance heuristic with Hungarian-style assignment; a weighted-push pass and a greedy solver for hard boards; every solution is replayed before being written. All budgets are expansion counts, never the wall clock, so two runs produce identical bytes.
