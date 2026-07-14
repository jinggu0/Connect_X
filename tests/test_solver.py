"""Correctness tests against known Connect-4 solved positions.

The reference scores come from the public Connect-4 solver test sets used by
Pascal Pons' benchmark suite (search "connect4 test set L3R1" / "Connect 4
Solver" for the canonical files: Test_L3_R1 etc). Each line is a sequence of
1-indexed column moves (no spaces) followed by the exact game-theoretic score
from the position reached, where positive = first player to move from that
position wins, 0 = draw, negative = loses. We hand-verify a handful of
well-known short lines directly against textbook Connect-4 theory (e.g. "the
first player wins with perfect play from the empty board by playing the
center column") rather than requiring network access to fetch the full
7-million-line official test set — that keeps this test suite runnable
offline while still catching regressions in the core search/eval logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import Board, Geometry, Searcher  # noqa: E402


def play_moves(board: Board, moves_1indexed: str) -> Board:
    for ch in moves_1indexed:
        board.play(int(ch) - 1)
    return board


def test_empty_board_center_wins_is_optimal():
    """Textbook Connect-4 result: from the empty 7x6 board, the first player
    has a forced win, and playing the center column (index 3) is the unique
    fastest winning first move under perfect play. This is the single most
    famous solved-Connect-4 fact (Victor Allis 1988 / James D. Allen 1989)."""
    geo = Geometry(rows=6, cols=7, inarow=4)
    board = Board(geo)
    searcher = Searcher(geo, time_budget_s=25.0)
    best = searcher.best_move(board, max_depth=42)
    assert best == 3, f"expected center column (3) as optimal first move, got {best}"


def test_immediate_win_is_taken():
    """If the mover has a one-move win available, the searcher must take it
    over any other move, regardless of search depth."""
    geo = Geometry(rows=6, cols=7, inarow=4)
    board = Board(geo)
    # Build: player A drops 3 in a row horizontally at the bottom, player B
    # plays elsewhere each time (column 6, out of the way).
    for col in (0, 6, 1, 6, 2, 6):
        board.play(col)
    # Position to move: A has stones at columns 0,1,2 row 0; playing column 3
    # completes four in a row horizontally.
    assert board.is_winning_move(3)
    searcher = Searcher(geo, time_budget_s=5.0)
    best = searcher.best_move(board, max_depth=10)
    assert best == 3


def test_must_block_opponent_win():
    """If the mover has no win but the opponent threatens to win next turn,
    the searcher must block."""
    geo = Geometry(rows=6, cols=7, inarow=4)
    board = Board(geo)
    # A plays a different out-of-the-way column each time (5, 6, 5) to avoid
    # accidentally stacking a vertical threat of its own; B builds three in a
    # row at columns 0,1,2 row 0. After this sequence it's A's turn and B
    # threatens to win at column 3.
    for col in (5, 0, 6, 1, 5, 2):
        board.play(col)
    # Sanity: A must not have an immediate win of its own from this stacking,
    # otherwise this wouldn't actually test the "must block" case.
    assert not any(board.can_play(c) and board.is_winning_move(c) for c in range(geo.cols))
    searcher = Searcher(geo, time_budget_s=5.0)
    best = searcher.best_move(board, max_depth=10)
    assert best == 3, f"expected forced block at column 3, got {best}"


def test_full_board_is_draw_score_zero():
    geo = Geometry(rows=1, cols=4, inarow=5)  # unreachable inarow => board fills, must draw
    board = Board(geo)
    for col in (0, 1, 2, 3):
        board.play(col)
    assert board.is_draw()


def test_small_board_full_solve_matches_bruteforce():
    """On a tiny 4x4 board with inarow=3, verify the searcher's root score
    matches an independent brute-force minimax (no bitboard tricks, no TT) —
    this is the primary regression guard for the bitboard win-detection and
    negamax logic, since bugs there would otherwise be invisible until they
    misplay real games."""
    geo = Geometry(rows=4, cols=4, inarow=3)

    def brute_force(board: Board, alpha: int, beta: int) -> int:
        max_cells = geo.rows * geo.cols
        for col in range(geo.cols):
            if board.can_play(col) and board.is_winning_move(col):
                return (max_cells - board.moves + 1) // 2
        if board.moves == max_cells:
            return 0
        best = -10**9
        for col in range(geo.cols):
            if not board.can_play(col):
                continue
            child = board.clone()
            child.play(col)
            score = -brute_force(child, -beta, -alpha)
            best = max(best, score)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    board = Board(geo)
    expected = brute_force(board, -10**9, 10**9)

    searcher = Searcher(geo, time_budget_s=15.0)
    got, _ = searcher.solve(board, max_depth=geo.rows * geo.cols)
    assert got == expected, f"searcher score {got} != brute-force score {expected}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        sys.exit(1)
