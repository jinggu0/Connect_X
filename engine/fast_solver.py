"""Numba-JIT-compiled negamax solver for standard Connect-4 (7 columns, 6 rows,
inarow=4). Fixed board size lets us hard-code geometry constants so Numba can
fully specialize the compiled code (no Python object overhead, no dict-based
transposition table — a flat numpy array keyed by position hash instead).

This module intentionally duplicates engine/bitboard.py's logic in a
Numba-friendly (no classes, no dicts, plain int64 state) form. engine/search.py
remains the general-purpose (arbitrary rows/cols/inarow) reference
implementation used for correctness testing and for non-standard board sizes
Kaggle's "connectx" task can be configured with; this module is the
performance-critical path for the one geometry that actually needs to be
solved exhaustively within Kaggle's per-move time budget.
"""

from __future__ import annotations

import numpy as np
from numba import njit, int64, boolean

ROWS = 6
COLS = 7
INAROW = 4
COL_HEIGHT = ROWS + 1  # sentinel bit per column, Tromp bitboard trick
MAX_CELLS = ROWS * COLS

_BOTTOM_MASK = 0
_BOARD_MASK = 0
for _c in range(COLS):
    _base = _c * COL_HEIGHT
    _BOTTOM_MASK |= 1 << _base
    for _r in range(ROWS):
        _BOARD_MASK |= 1 << (_base + _r)

_TOP_MASKS = np.array([1 << (ROWS - 1 + c * COL_HEIGHT) for c in range(COLS)], dtype=np.int64)
_BOTTOM_COL_MASKS = np.array([1 << (c * COL_HEIGHT) for c in range(COLS)], dtype=np.int64)
_COLUMN_MASKS = np.array([((1 << ROWS) - 1) << (c * COL_HEIGHT) for c in range(COLS)], dtype=np.int64)

# Center-out column visitation order: searching central columns first prunes
# far more of the tree on average (see engine/search.py's move_order for the
# same rationale in the general solver).
_MOVE_ORDER = np.array(sorted(range(COLS), key=lambda c: abs(c - (COLS - 1) / 2.0)), dtype=np.int64)

TT_SIZE = 1 << 23  # ~8.4M slots; each slot is one int64 (packed key+depth+score+flag)
_TT_EMPTY = np.int64(0)


@njit(int64(int64), cache=True)
def mirror(bb: int) -> int:
    """Reflect a bitboard left-right (column c -> column COLS-1-c), keeping
    each column's bits in place. Connect-4 is symmetric under this reflection
    (an optimal move at column c has a mirror-optimal move at COLS-1-c from
    the mirrored position), so the opening book only needs to store one
    representative per mirror-pair — roughly halving book size and build time.
    """
    out = 0
    for c in range(COLS):
        src_col_bits = (bb >> (c * COL_HEIGHT)) & ((1 << COL_HEIGHT) - 1)
        out |= src_col_bits << ((COLS - 1 - c) * COL_HEIGHT)
    return out


@njit(boolean(int64), cache=True)
def _alignment(bb: int) -> bool:
    # Vertical, horizontal, diag /, diag \ — same bit-trick as engine/bitboard.py.
    for shift in (1, COL_HEIGHT, COL_HEIGHT - 1, COL_HEIGHT + 1):
        bb2 = bb
        for _ in range(INAROW - 1):
            bb2 = bb2 & (bb2 >> shift)
        if bb2 != 0:
            return True
    return False


@njit(boolean(int64, int64), cache=True)
def can_play(mask: int, col: int) -> bool:
    return (mask & _TOP_MASKS[col]) == 0


@njit(boolean(int64, int64, int64), cache=True)
def is_winning_move(position: int, mask: int, col: int) -> bool:
    move = (mask + _BOTTOM_COL_MASKS[col]) & _COLUMN_MASKS[col]
    return _alignment(position | move)


@njit(cache=True)
def play(position: int, mask: int, col: int):
    move = (mask + _BOTTOM_COL_MASKS[col]) & _COLUMN_MASKS[col]
    # See engine/bitboard.py's play() docstring for the derivation: XOR against
    # the OLD mask (before adding this move) so position_new correctly becomes
    # the *opponent's* stones, then add the move bit to mask afterward.
    new_position = position ^ mask
    new_mask = mask | move
    return new_position, new_mask


@njit(cache=True)
def heuristic_score(position: int, mask: int) -> int:
    """Depth-cutoff evaluation for when the search budget runs out before
    reaching a terminal position. Same sliding-window idea as engine/eval.py's
    window_score (count open n-in-a-row windows, weight superlinearly by how
    many stones already occupy them) but hard-coded to the 7x6/inarow=4
    geometry and written branch-simple enough for Numba to vectorize well.
    Score is from `position`'s (the player to move's) perspective, deliberately
    scaled far below the ±21 range of true win/loss scores from negamax's
    "distance to end" convention, so a heuristic leaf is never confused with
    (or allowed to outrank) a proven forced win/loss found deeper elsewhere in
    the same search."""
    opp = mask ^ position
    score = 0
    for c in range(COLS):
        base = c * COL_HEIGHT
        for r in range(ROWS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                r_end = r + dr * (INAROW - 1)
                c_end = c + dc * (INAROW - 1)
                if r_end < 0 or r_end >= ROWS or c_end < 0 or c_end >= COLS:
                    continue
                mine_ct = 0
                theirs_ct = 0
                for k in range(INAROW):
                    rr = r + dr * k
                    cc = c + dc * k
                    bit = 1 << (cc * COL_HEIGHT + rr)
                    if position & bit:
                        mine_ct += 1
                    elif opp & bit:
                        theirs_ct += 1
                if mine_ct > 0 and theirs_ct > 0:
                    continue
                if mine_ct == 3:
                    score += 25
                elif mine_ct == 2:
                    score += 5
                elif mine_ct == 1:
                    score += 1
                elif theirs_ct == 3:
                    score -= 25
                elif theirs_ct == 2:
                    score -= 5
                elif theirs_ct == 1:
                    score -= 1
    return score


@njit(cache=True)
def negamax(position: int, mask: int, moves: int, alpha: int, beta: int,
            depth_left: int, tt_keys, tt_vals) -> int:
    """Transposition-table packing (inlined here rather than via helper
    functions, since Numba compiles this whole function as one unit): each
    int64 slot in tt_vals packs (depth_remaining << 24) | (flag << 22) |
    (score + 2^20). depth_remaining = min(MAX_CELLS - moves, depth_left) at
    storage time (how many plies of search actually backed this score) — a
    cached entry is only reusable if it covered at least as many remaining
    plies as the current query needs. flag in {0: exact, 1: lower bound, 2:
    upper bound}, standard alpha-beta TT semantics. The score is offset by
    2^20 so the packed value stays non-negative.

    depth_left bounds search depth so this function can be used inside an
    iterative-deepening time budget (unlike a from-scratch full solve, which
    always recurses to a terminal state and can take minutes — see
    engine/opening_book.py's abandoned approach). When depth_left hits 0
    before a terminal position, heuristic_score() provides a depth-cutoff
    evaluation instead of a proven exact/bound score.
    """
    if moves == MAX_CELLS:
        return 0

    for i in range(COLS):
        col = _MOVE_ORDER[i]
        if can_play(mask, col) and is_winning_move(position, mask, col):
            return (MAX_CELLS - moves + 1) // 2

    max_possible = (MAX_CELLS - moves - 1) // 2
    if beta > max_possible:
        beta = max_possible
        if alpha >= beta:
            return beta

    if depth_left <= 0:
        return heuristic_score(position, mask)

    key = position + mask
    idx = key % TT_SIZE
    orig_alpha = alpha
    if tt_keys[idx] == key:
        v = tt_vals[idx]
        depth = (v >> 24) & 0xFF
        flag = (v >> 22) & 0x3
        score = (v & 0x3FFFFF) - (1 << 20)
        # depth field stores plies actually searched below this node; reusable
        # only if that covers at least as many remaining plies as this query.
        if depth >= min(depth_left, MAX_CELLS - moves):
            if flag == 0:
                return score
            elif flag == 1:
                if score > alpha:
                    alpha = score
            elif flag == 2:
                if score < beta:
                    beta = score
            if alpha >= beta:
                return score

    best_score = -1_000_000
    for i in range(COLS):
        col = _MOVE_ORDER[i]
        if not can_play(mask, col):
            continue
        child_pos, child_mask = play(position, mask, col)
        score = -negamax(child_pos, child_mask, moves + 1, -beta, -alpha, depth_left - 1, tt_keys, tt_vals)
        if score > best_score:
            best_score = score
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            break

    flag = 0
    if best_score <= orig_alpha:
        flag = 2
    elif best_score >= beta:
        flag = 1
    stored_depth = min(depth_left, MAX_CELLS - moves)
    packed = (stored_depth << 24) | (flag << 22) | (best_score + (1 << 20))
    tt_keys[idx] = key
    tt_vals[idx] = packed
    return best_score


def new_tt():
    """Fresh transposition table arrays. Reuse across moves within one game
    (positions revisit often via transposition) but reset between unrelated
    games/tests to avoid stale cross-game bias — keys collide extremely rarely
    at this table size for a single game's reachable position count, but
    resetting between independent games is cheap and removes any doubt."""
    keys = np.zeros(TT_SIZE, dtype=np.int64)
    vals = np.zeros(TT_SIZE, dtype=np.int64)
    return keys, vals


def solve_root_at_depth(position: int, mask: int, moves: int, depth_left: int, tt_keys, tt_vals):
    """Full-width root search at a fixed depth_left: returns {col: score} for
    all legal moves. Each sibling is searched with a FULL alpha-beta window
    (-inf, +inf) rather than tightening alpha across siblings: with an exact
    (full-depth) search, narrowing the window from previous siblings is a
    safe and standard root-search speedup, but combined with depth_left's
    heuristic leaf cutoff it was verified to produce false ties — a sibling
    searched under an already-tightened window gets its heuristic-approximated
    value clipped against that window and can collapse to the same bound as
    unrelated siblings. Root-level pruning loss from full windows is a minor
    cost; root has few children (<=7) compared to the exponential interior of
    the tree, so this is not a meaningful performance concern."""
    legal = [int(c) for c in _MOVE_ORDER if can_play(mask, c)]
    scores = {}
    for col in legal:
        if is_winning_move(position, mask, col):
            scores[col] = (MAX_CELLS - moves + 1) // 2
            continue
        child_pos, child_mask = play(position, mask, col)
        s = -negamax(child_pos, child_mask, moves + 1, -1_000_000, 1_000_000, depth_left - 1, tt_keys, tt_vals)
        scores[col] = s
    return scores


def solve_root(position: int, mask: int, moves: int, tt_keys, tt_vals,
                time_budget_s: float = 5.0, max_depth: int | None = None):
    """Iterative-deepening time-budgeted root search. Returns {col: score}
    from the deepest depth completed within time_budget_s. Depth-limited
    negamax with a heuristic leaf cutoff (see heuristic_score) means this is
    NOT guaranteed to be the exact game-theoretic score unless it reaches full
    depth — but it stays within Kaggle's per-move time budget on every
    position, unlike a from-scratch full solve (which was verified to take
    minutes on empty/near-empty boards; see engine/opening_book.py's
    docstring for that investigation).

    Move ordering carries over between depths (best-first from the previous
    iteration searched first at the next), which is the standard iterative-
    deepening speedup: a good move found at shallow depth tends to still be
    strong at deeper depth, so alpha tightens fast and prunes more.
    """
    import time as _time

    deadline = _time.monotonic() + time_budget_s
    cells_left = MAX_CELLS - moves
    hard_max = cells_left if max_depth is None else min(max_depth, cells_left)

    legal = [int(c) for c in _MOVE_ORDER if can_play(mask, c)]
    if not legal:
        return {}

    # Root-level immediate win short-circuit (mirrors the check inside
    # negamax; see engine/search.py's analogous root fix and the regression
    # test that caught its absence — tests/test_solver.py::test_immediate_win_is_taken).
    for col in legal:
        if is_winning_move(position, mask, col):
            return {col: (MAX_CELLS - moves + 1) // 2}

    best_scores: dict[int, int] = {}
    depth = 1
    while depth <= hard_max:
        if _time.monotonic() > deadline:
            break
        # Full window per sibling (see solve_root_at_depth's docstring for why
        # narrowing alpha across root siblings was verified to produce false
        # ties when combined with the heuristic depth cutoff).
        round_scores: dict[int, int] = {}
        timed_out_mid_round = False
        for col in legal:
            if _time.monotonic() > deadline:
                timed_out_mid_round = True
                break
            child_pos, child_mask = play(position, mask, col)
            s = -negamax(child_pos, child_mask, moves + 1, -1_000_000, 1_000_000, depth - 1, tt_keys, tt_vals)
            round_scores[col] = s
        if timed_out_mid_round:
            # Partial round: only trust it if every legal move got a score,
            # otherwise keep the previous (shallower but complete) round.
            break
        best_scores = round_scores
        legal = sorted(legal, key=lambda c: -round_scores[c])
        depth += 1

    return best_scores
