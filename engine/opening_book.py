"""ABANDONED — kept only as a record of a measured dead end; not used by
agent.py. See tests/test_solver.py and CLAUDE.md-style postmortems for why:
this module's approach does not scale on the hardware/time available for this
project, in either direction:

  - Shallow book depth (e.g. depth=6, 30 empty cells remaining per frontier
    position): measured >60s to fully solve a SINGLE frontier position with
    the numba negamax+TT solver. With ~59k unique positions after mirror
    dedup at depth 6, the full build would take on the order of weeks.
  - Deep book depth (e.g. depth=10+): enumerate_frontier's pure-Python BFS
    materializes the entire frontier as a list of Python tuples before
    dedup — at depth 10 this is up to 7^10 ≈ 282M raw entries, which
    exhausted multiple GB of RAM before finishing enumeration, let alone
    starting to solve any of them.

Conclusion (see engine/fast_solver.py's iterative-deepening time-budgeted
solve_root instead): for this project's constraints, a from-scratch
opening-book precomputation is not tractable without either (a) a much faster
solver core (e.g. threat-based pruning per Pascal Pons' full solver — see the
"desired but not implemented" note in fast_solver.py) or (b) an
external/streaming frontier generator that never materializes the full
position set in memory. Runtime alpha-beta within a per-move time budget,
without a precomputed book, is the approach actually shipped.

--- Original design notes below, preserved for reference ---

Motivation: a from-scratch negamax solve of the empty board takes minutes
even with alpha-beta + a transposition table (verified empirically — this is
consistent with published results; Pascal Pons' reference solver needs
additional threat-based pruning techniques to solve it in under a second).
Kaggle's per-move time budget is only a few seconds. The standard solution
used by every serious Connect-4 engine: solve the first K plies exhaustively
*once*, offline, and ship the results as a lookup table. At runtime, the
agent checks the book first (O(1) lookup) and only falls back to a live
alpha-beta search once the game leaves book territory (by which point far
fewer empty cells remain, so the live search is fast).

Build process (BFS by depth, depth-first negamax at the frontier):
  1. Enumerate all *reachable* positions at ply depths 0..K-1 via BFS over
     legal moves (skipping any position where the mover already has an
     immediate win available in the parent — such positions are terminal and
     never reached mid-game by two players avoiding pointless losses, and
     more importantly are irrelevant to the book: our own agent would have
     already taken that immediate win rather than needing a book lookup).
  2. For each depth-K frontier position not already resolved, run the full
     negamax solve and store {position_key: score}.
  3. Store only one representative per left-right mirror pair (see
     fast_solver.mirror) — at lookup time the runtime checks both the
     position's key and its mirror's key.

The resulting book is saved as a compact binary file: a sorted array of
int64 keys and a parallel int8 array of scores (scores are always in
[-21, 21] for this geometry), loadable via numpy.fromfile for fast startup.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import fast_solver as fs  # noqa: E402


def canonical_key(position: int, mask: int) -> int:
    """Position key that is invariant under left-right mirroring: the smaller
    of (key, mirrored key). Two mirror-symmetric positions map to the same
    canonical key, so the book only needs one entry per pair."""
    key = position + mask
    mkey = fs.mirror(position) + fs.mirror(mask)
    return key if key <= mkey else mkey


def enumerate_frontier(depth: int):
    """BFS-generate all (position, mask, moves) reached after exactly `depth`
    plies from the empty board, skipping any branch where a player had an
    immediate win available and (for book-building purposes) simply not
    taking it — i.e. we prune winning moves from the *parent's* expansion
    since a real game (or our own agent) would always take an immediate win
    rather than needing this book entry. This keeps the frontier to
    "genuinely undecided" positions, which is both smaller and the only case
    the book actually needs to answer."""
    frontier = [(0, 0, 0)]  # (position, mask, moves)
    for _ply in range(depth):
        next_frontier = []
        for position, mask, moves in frontier:
            for col in range(fs.COLS):
                if not fs.can_play(mask, col):
                    continue
                if fs.is_winning_move(position, mask, col):
                    continue  # terminal; not a useful book branch (see docstring)
                cp, cm = fs.play(position, mask, col)
                next_frontier.append((cp, cm, moves + 1))
        frontier = next_frontier
    return frontier


def build_book(depth: int, time_budget_per_position_s: float = 30.0, progress_every: int = 200):
    """Solve every unique (post-mirror-dedup) position reached after `depth`
    plies and return {canonical_key: score}. Reuses ONE shared transposition
    table across all positions at this depth — later positions in the sweep
    benefit from TT entries populated while solving earlier, related
    positions (they often share deep subtrees), which speeds up the sweep
    noticeably versus a fresh TT per position."""
    frontier = enumerate_frontier(depth)
    print(f"depth {depth}: {len(frontier)} raw positions before mirror-dedup")

    seen_keys = set()
    unique_positions = []
    for position, mask, moves in frontier:
        ck = canonical_key(position, mask)
        if ck in seen_keys:
            continue
        seen_keys.add(ck)
        unique_positions.append((position, mask, moves, ck))
    print(f"depth {depth}: {len(unique_positions)} unique positions after mirror-dedup")

    tt_keys, tt_vals = fs.new_tt()
    book: dict[int, int] = {}
    t0 = time.time()
    for i, (position, mask, moves, ck) in enumerate(unique_positions):
        if ck in book:
            continue
        # Immediate-win short circuit (mirrors the root check in fast_solver's
        # solve_root; frontier positions never have one by construction, but
        # kept here for defense-in-depth / reuse if enumerate_frontier's
        # pruning rule ever changes).
        best = -1_000_000
        for col in range(fs.COLS):
            if not fs.can_play(mask, col):
                continue
            if fs.is_winning_move(position, mask, col):
                best = (fs.MAX_CELLS - moves + 1) // 2
                break
            cp, cm = fs.play(position, mask, col)
            s = -fs.negamax(cp, cm, moves + 1, -1_000_000, 1_000_000, fs.MAX_CELLS, tt_keys, tt_vals)
            if s > best:
                best = s
        book[ck] = best
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(unique_positions) - i - 1) / rate if rate > 0 else float("inf")
            print(f"  [{i+1}/{len(unique_positions)}] {elapsed:.0f}s elapsed, "
                  f"{rate:.2f} pos/s, ETA {eta:.0f}s")
    print(f"depth {depth} book complete: {len(book)} entries in {time.time()-t0:.0f}s")
    return book


def save_book(book: dict[int, int], path: Path):
    keys = np.array(sorted(book.keys()), dtype=np.int64)
    scores = np.array([book[k] for k in keys], dtype=np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.array([len(keys)], dtype=np.int64).tofile(f)
        keys.tofile(f)
        scores.tofile(f)
    print(f"saved {len(keys)} entries to {path} ({path.stat().st_size / 1024:.0f} KB)")


def load_book(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        n = int(np.fromfile(f, dtype=np.int64, count=1)[0])
        keys = np.fromfile(f, dtype=np.int64, count=n)
        scores = np.fromfile(f, dtype=np.int8, count=n)
    return keys, scores


def book_lookup(keys: np.ndarray, scores: np.ndarray, position: int, mask: int):
    """Binary-search the sorted keys array for this position's canonical key.
    Returns the score if found, else None (caller should fall back to live
    search)."""
    ck = canonical_key(position, mask)
    idx = np.searchsorted(keys, ck)
    if idx < len(keys) and keys[idx] == ck:
        return int(scores[idx])
    return None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, default=8, help="opening book ply depth")
    ap.add_argument("--out", type=str, default="engine/book_data/book_d{depth}.bin")
    args = ap.parse_args()

    book = build_book(args.depth)
    out_path = Path(args.out.format(depth=args.depth))
    save_book(book, out_path)
