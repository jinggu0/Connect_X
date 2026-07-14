"""Negamax + alpha-beta search with transposition table, move ordering, and
iterative deepening. Targets exact (perfect-play) solutions for standard
Connect-4 (7x6, inarow=4) well within Kaggle's per-move time budget, and
degrades gracefully (depth-limited heuristic search) on larger boards where
a full solve isn't feasible in time.

Score convention (from the mover's perspective, like standard negamax):
    +N  = mover wins with N empty cells remaining after their winning move
          (higher = faster win; encourages the fastest forced win)
    -N  = mover loses with N empty cells remaining when the opponent wins
          (encourages delaying an inevitable loss as long as possible)
     0  = draw
This "distance-to-end" scaling is standard practice in solved-game engines
(e.g. Pascal Pons' Connect-4 solver) because it makes the engine play the
fastest available win and the slowest available loss, rather than being
indifferent among all winning/losing lines.
"""

from __future__ import annotations

import time
from typing import Optional

from .bitboard import Board, Geometry, top_mask


class TimeUp(Exception):
    pass


class TranspositionTable:
    """Simple dict-based TT keyed by board.key(). Stores (depth_remaining, score,
    flag) where flag in {"exact", "lower", "upper"} for alpha-beta bound types.
    A dict is adequate at Connect-4 search sizes (~10-60M unique reachable
    positions in the worst case, but practical games explore far fewer within
    a move's time budget); swap for a fixed-size array table if profiling shows
    memory pressure.
    """

    __slots__ = ("table", "max_entries")

    def __init__(self, max_entries: int = 8_000_000):
        self.table: dict[int, tuple[int, int, str]] = {}
        self.max_entries = max_entries

    def get(self, key: int):
        return self.table.get(key)

    def put(self, key: int, depth: int, score: int, flag: str) -> None:
        if len(self.table) >= self.max_entries:
            # Cheap eviction: clear everything. A production engine would use
            # replacement schemes (e.g. depth-preferred), but a full clear is
            # simple and rare in practice given the entries budget.
            self.table.clear()
        self.table[key] = (depth, score, flag)


def move_order(geo: Geometry):
    """Center-out column order: center columns prune far more nodes on average
    because central moves influence more winning lines, so searching them
    first tightens alpha-beta bounds fastest."""
    center = (geo.cols - 1) / 2.0
    return sorted(range(geo.cols), key=lambda c: abs(c - center))


class Searcher:
    def __init__(self, geo: Geometry, time_budget_s: float = 1.8):
        self.geo = geo
        self.time_budget_s = time_budget_s
        self.tt = TranspositionTable()
        self.order = move_order(geo)
        self.deadline: Optional[float] = None
        self.nodes = 0

    def _check_time(self):
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeUp()

    def negamax(self, board: Board, alpha: int, beta: int, depth_left: int) -> int:
        self.nodes += 1
        if self.nodes & 0x3FF == 0:  # check clock every 1024 nodes (cheap amortized cost)
            self._check_time()

        max_cells = self.geo.rows * self.geo.cols
        if board.moves == max_cells:
            return 0  # board full, no winner => draw

        # Immediate win check for the current player (saves a full recursive
        # ply for the common "I have a winning move right now" case).
        for col in self.order:
            if board.can_play(col) and board.is_winning_move(col):
                return (max_cells - board.moves + 1) // 2

        # Upper bound: best possible score this ply is a win one move later than
        # the fastest immediate win already ruled out above.
        max_possible = (max_cells - board.moves - 1) // 2
        if beta > max_possible:
            beta = max_possible
            if alpha >= beta:
                return beta

        if depth_left <= 0:
            return heuristic_eval(board)

        key = board.key()
        tt_entry = self.tt.get(key)
        if tt_entry is not None:
            tt_depth, tt_score, flag = tt_entry
            if tt_depth >= depth_left:
                if flag == "exact":
                    return tt_score
                if flag == "lower":
                    alpha = max(alpha, tt_score)
                elif flag == "upper":
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        orig_alpha = alpha
        best_score = -10**9
        any_move = False
        for col in self.order:
            if not board.can_play(col):
                continue
            any_move = True
            child = board.clone()
            child.play(col)
            score = -self.negamax(child, -beta, -alpha, depth_left - 1)
            if score > best_score:
                best_score = score
            if best_score > alpha:
                alpha = best_score
            if alpha >= beta:
                break

        if not any_move:
            return 0  # no legal moves (shouldn't happen if moves < max_cells)

        flag = "exact"
        if best_score <= orig_alpha:
            flag = "upper"
        elif best_score >= beta:
            flag = "lower"
        self.tt.put(key, depth_left, best_score, flag)
        return best_score

    def solve(self, board: Board, max_depth: int) -> tuple[int, dict[int, int]]:
        """Iterative-deepening search. Returns (best_score, {col: score}) for
        the root position, using the deepest depth completed before the time
        budget expires. Falls back to a shallower depth's result if a deeper
        iteration times out mid-search."""
        self.deadline = time.monotonic() + self.time_budget_s
        self.nodes = 0

        legal = [c for c in self.order if board.can_play(c)]
        if not legal:
            return 0, {}

        max_cells = self.geo.rows * self.geo.cols

        # Immediate-win short-circuit at the root, mirroring the check inside
        # negamax(). Without this, a winning move's resulting child position
        # is fed straight into negamax(), which evaluates it as an ordinary
        # (non-terminal) position for the *opponent's* turn and never
        # recognizes that the game already ended on our move — silently
        # missing forced wins. Caught by tests/test_solver.py::test_immediate_win_is_taken.
        for col in legal:
            if board.is_winning_move(col):
                win_score = (max_cells - board.moves + 1) // 2
                return win_score, {col: win_score}

        best_result: dict[int, int] = {}
        best_score = 0
        depth = 2
        while depth <= max_depth:
            try:
                root_scores: dict[int, int] = {}
                alpha, beta = -10**9, 10**9
                for col in legal:
                    child = board.clone()
                    child.play(col)
                    s = -self.negamax(child, -beta, -alpha, depth - 1)
                    root_scores[col] = s
                    if s > alpha:
                        alpha = s
                best_result = root_scores
                best_score = max(root_scores.values())
                # Order next iteration's root moves by this iteration's scores
                # (best-first) for faster alpha-beta convergence at deeper depth.
                legal = sorted(legal, key=lambda c: -root_scores[c])
            except TimeUp:
                break
            depth += 1

        return best_score, best_result

    def best_move(self, board: Board, max_depth: int) -> int:
        _, scores = self.solve(board, max_depth)
        if not scores:
            for c in self.order:
                if board.can_play(c):
                    return c
            raise RuntimeError("no legal moves")
        return max(scores.items(), key=lambda kv: kv[1])[0]


def heuristic_eval(board: Board) -> int:
    """Depth-limited fallback evaluation when a full solve isn't reached in
    time (used for larger-than-standard boards). Counts open n-in-a-row
    "threat windows" weighted toward the center, from the current mover's
    perspective. This is a lightweight placeholder — see engine/eval.py for
    the tunable, tested version used in the actual submission.
    """
    from .eval import window_score
    return window_score(board)
