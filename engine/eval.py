"""Heuristic evaluation for depth-limited search on boards too large to solve
exactly within the time budget (Kaggle's "connectx" task can be configured
with non-standard rows/columns/inarow, unlike the fixed 7x6/inarow=4 default).

The standard Connect-4 sliding-window heuristic: for every contiguous window
of `inarow` cells (in all four directions), count how many stones each player
has in it. A window that already contains both players' stones can never
become a win for either, so it contributes nothing. Windows that are "open"
(all remaining cells empty) are weighted by how many stones the player already
has placed in them — more stones in an open window means fewer additional
moves needed to complete it.
"""

from __future__ import annotations

from .bitboard import Board, Geometry


def _windows(geo: Geometry):
    """Yield all (row, col, dr, dc) window start points and directions for the
    board geometry. Cached per-geometry since it's independent of board state."""
    n = geo.inarow
    for r in range(geo.rows):
        for c in range(geo.cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                r_end = r + dr * (n - 1)
                c_end = c + dc * (n - 1)
                if 0 <= r_end < geo.rows and 0 <= c_end < geo.cols:
                    yield [(r + dr * k, c + dc * k) for k in range(n)]


_WINDOW_CACHE: dict[tuple[int, int, int], list] = {}


def _get_windows(geo: Geometry):
    key = (geo.rows, geo.cols, geo.inarow)
    cached = _WINDOW_CACHE.get(key)
    if cached is None:
        cached = list(_windows(geo))
        _WINDOW_CACHE[key] = cached
    return cached


def _bit(geo: Geometry, row: int, col: int) -> int:
    return 1 << (col * geo.col_height + row)


def window_score(board: Board) -> int:
    """Score from the current mover's (board.position) perspective. Positive
    favors the mover; the opponent's stones are in (board.mask ^ board.position)."""
    geo = board.geo
    mine = board.position
    theirs = board.mask ^ board.position
    n = geo.inarow

    score = 0
    for window in _get_windows(geo):
        mine_ct = 0
        theirs_ct = 0
        for (r, c) in window:
            b = _bit(geo, r, c)
            if mine & b:
                mine_ct += 1
            elif theirs & b:
                theirs_ct += 1
        if mine_ct > 0 and theirs_ct > 0:
            continue  # contested window, can't be won by either side
        if mine_ct > 0:
            score += _WINDOW_WEIGHT.get(mine_ct, mine_ct * mine_ct)
        elif theirs_ct > 0:
            score -= _WINDOW_WEIGHT.get(theirs_ct, theirs_ct * theirs_ct)
    return score


# Weight growth is intentionally superlinear: a window one stone away from
# completing a win (n-1 stones placed) is far more valuable than the sum of
# its parts, since the opponent must spend a move blocking it or lose.
_WINDOW_WEIGHT = {1: 1, 2: 5, 3: 25, 4: 500}
