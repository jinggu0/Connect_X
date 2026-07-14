# Connect X — Bitboard Negamax Engine

A from-scratch Connect-4 (and general Connect-X) solver built for
[Kaggle's ConnectX competition](https://www.kaggle.com/competitions/connectx).
Submitted agent: [`agent.py`](agent.py) — currently live on the leaderboard.

## What this is

Two things, on purpose:

1. **A general Connect-X engine** (`engine/`) — bitboard state representation,
   negamax + alpha-beta search, a numba-JIT-compiled fast path, and a
   correctness test suite checked against known solved positions. Built for
   development, benchmarking, and understanding the algorithm properly rather
   than just wiring up a library.
2. **A single-file Kaggle submission** (`agent.py`) — the same design ported
   to pure standard library, because Kaggle's grading environment for
   simulation competitions does not guarantee third-party packages are
   available to submitted agents (only the dev/notebook environment is).

## Architecture

```
engine/
  bitboard.py      Tromp bitboard encoding (arbitrary rows/cols/inarow)
  search.py         General-purpose negamax + alpha-beta reference solver
  fast_solver.py    numba-JIT version, hard-coded to standard 7x6/inarow=4,
                    for fast offline development and benchmarking
  eval.py           Sliding-window heuristic (shared logic reference)
  opening_book.py   An approach that was tried and abandoned — see below
tests/
  test_solver.py    Correctness tests against known solved Connect-4 positions
agent.py             The actual Kaggle submission (self-contained, stdlib only)
```

### Bitboard representation

Each column occupies `rows + 1` bits (one sentinel bit per column), following
the encoding popularized by John Tromp and used in Pascal Pons' reference
Connect-4 solver. Two 64-bit integers represent the whole board:

- `mask` — every occupied cell, either player
- `position` — cells occupied by the player *to move*, re-derived via XOR
  after each ply so it always reflects the mover's perspective

This makes win-detection, legal-move checks, and move application branch-free
bit operations instead of per-cell array scans.

### Search

Negamax with alpha-beta pruning, a transposition table (dict in `agent.py`,
a flat numpy array keyed by position hash in `fast_solver.py`), center-out
move ordering, and iterative deepening within a wall-clock time budget per
move. When the depth budget runs out before a terminal position, a
sliding-window heuristic (count of open n-in-a-row windows, weighted
superlinearly by how many stones already occupy them) provides the leaf
evaluation.

The transposition table in `agent.py` persists across turns within a game
(module-level, not re-created per `my_agent()` call), guarded by new-game
detection (geometry change or a stone count that decreased) so a stale cache
is never carried into an unrelated game. Kaggle reuses the same process for
every turn, so this means later-game search benefits from every earlier
turn's work rather than starting cold each time — the effect compounds
particularly playing second, since the second player always has one fewer
empty cell than the first at every point in the game.

## Design decisions and dead ends

Worth documenting because they're where the actual engineering happened:

- **Opening book: tried and abandoned.** The standard approach in serious
  Connect-4 engines is to solve the first K plies exhaustively offline and
  ship the result as a lookup table, since a from-scratch solve of the empty
  board takes minutes even with alpha-beta + a transposition table. Measured
  empirically here: a shallow book (depth 6, ~30 empty cells per frontier
  position) needed 60+ seconds to fully solve a *single* frontier position —
  infeasible at ~59k unique positions. A deep book (depth 10+) blew past
  several GB of RAM just enumerating the frontier in Python before solving
  anything. See [`engine/opening_book.py`](engine/opening_book.py)'s
  docstring for the full writeup. Conclusion: without either a much faster
  solver core (threat-based pruning, per Pons' full solver) or a
  streaming/external frontier generator, precomputing a book isn't tractable
  at this project's scale — so the shipped agent relies entirely on
  runtime iterative-deepening search within a per-move time budget.

- **A real correctness bug, caught by testing, not inspection.** Early root
  search results showed every candidate column returning an *identical*
  score at certain depths — including on the empty board, where center-column
  superiority is a textbook Connect-4 result. Root cause: narrowing the
  alpha-beta window across sibling root moves (a standard, safe speedup for
  an *exact* search) silently broke once combined with the heuristic
  depth-cutoff — a sibling searched under an already-tightened window got its
  heuristic-approximated value clipped against that window and collapsed to
  the same bound as unrelated siblings. Fix: each root sibling is searched
  with a full `(-inf, +inf)` window. Root has at most 7 children, so the
  pruning loss is negligible; correctness isn't.

## A note on win rate

Standard 7x6 Connect-4 is a solved game (Allis 1988 / Allen 1989): the first
player wins with perfect play. That means 100% win rate is only even
theoretically possible playing first, against imperfect opponents, with a
search deep enough to reach the game-theoretic optimum — this repo's
`test_empty_board_center_wins_is_optimal` test checks that the engine finds
the correct optimal first move, but the *submitted* agent runs a time-budgeted
approximate search, not an exhaustive one, since a from-scratch full solve of
the empty board takes minutes (see the opening-book section above). Playing
second, no amount of search closes a fully solved gap against a perfect
opponent; against the actual field of imperfect bots, the practical lever is
converting the opponent's mistakes into wins as reliably as possible — which
is what the cross-turn transposition table above is for.

## Correctness testing

```
python -m pytest tests/test_solver.py -v
```

or run directly:

```
python tests/test_solver.py
```

Checks include: taking an immediate winning move over anything else, forced
blocking of an opponent's immediate win, correct draw scoring on a filled
board, and — the strongest regression guard — cross-checking the bitboard
searcher's output against an independent, tricks-free brute-force minimax on
a small 4x4/inarow=3 board. All four checks pass.

## Running / developing locally

```bash
pip install numba numpy kaggle-environments   # engine/fast_solver.py + local sim only; agent.py needs neither
python -m pytest tests/test_solver.py -v
```

Play `agent.py` against Kaggle's built-in bots locally:

```python
from kaggle_environments import make
import agent

env = make("connectx")
env.run([agent.my_agent, "random"])   # or "negamax"
env.render(mode="ipython")
```

## Submission status

Submitted to the Connect X leaderboard via the Kaggle API
(`kaggle competitions submit -c connectx -f agent.py`). Validation episode
passed (`SubmissionStatus.COMPLETE`); the score settles over subsequent
matches against the field as is standard for Kaggle Simulations competitions.

## Why pure Python for the submission, numba for dev

`engine/fast_solver.py` is roughly two orders of magnitude faster than
`agent.py` for the same search — genuinely useful for benchmarking and for
exploring how deep the search can realistically go. But Kaggle's grading
sandbox for submitted agents is not guaranteed to have numba installed (only
the interactive notebook environment is), and a submission that fails to
import at grading time is a hard forfeit. `agent.py` trades raw speed for
certainty it actually runs — the same algorithm, same tests, standard
library only.
