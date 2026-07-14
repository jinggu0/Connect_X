"""Kaggle ConnectX submission — single self-contained file (Kaggle's
submission format requires one .py file with a top-level my_agent(observation,
configuration) function; no local imports of engine/* work at grading time,
so all logic is duplicated here rather than imported).

Pure standard library only (no numba/numpy) — Kaggle's grading environment
for simulation competitions is not guaranteed to have third-party packages
available for submitted agents (only the notebook/dev environment is), so
this intentionally trades the ~100x raw speed of the numba version (see
engine/fast_solver.py, used for offline development/benchmarking) for
certainty that it actually runs at grading time.

Algorithm: bitboard-represented Connect-4 (Tromp encoding, see
engine/bitboard.py for the derivation and correctness tests this mirrors),
negamax + alpha-beta + a dict-based transposition table + center-out move
ordering, run under iterative deepening within a wall-clock time budget per
move. Falls back to a sliding-window heuristic evaluation at the depth
cutoff. This is the same design validated in engine/fast_solver.py and
tests/test_solver.py, ported to pure Python.
"""

from __future__ import annotations

import time

# ---- Cross-turn transposition table cache ----
# Kaggle reuses the same submitted process for every turn of a game (the
# submission is loaded once, not re-imported per move), so module-level state
# survives between my_agent() calls within one game. A fresh TT every turn
# was throwing away an entire turn's worth of search on every single call —
# wasteful in general, but especially costly when playing second: the second
# player always has one fewer empty cell than the first player at every point
# in the game, so with TT continuity the search converges toward the
# provably-optimal move faster than a from-scratch-every-turn search ever
# could, narrowing (though not eliminating — see README's Connect-4 first-
# player-win theorem note) the practical gap against an imperfect opponent.
#
# _TT_GAME_ID guards against carrying a stale TT into a NEW game (Kaggle can
# reuse a process across multiple games in local/dev harnesses, and even in
# a single competition run there is no guarantee every call belongs to the
# same game). A TT keyed by (position + mask) is only valid for one fixed
# (rows, cols, inarow) geometry and one specific move history, so any signal
# that we've started over — fewer stones on the board than last call, or a
# different board geometry — must invalidate the cache before use rather than
# silently returning a plausible-looking but wrong cached score.
_tt_cache: dict[int, tuple[int, int, int]] = {}
_tt_last_moves = -1
_tt_geometry: tuple[int, int, int] | None = None


def my_agent(observation, configuration):
    global _tt_last_moves, _tt_geometry
    rows = configuration.rows
    cols = configuration.columns
    inarow = configuration.inarow
    my_mark = observation.mark
    board = observation.board  # length rows*cols, row-major from the TOP, 0=empty, 1/2=player marks

    col_height = rows + 1  # sentinel bit per column (Tromp bitboard trick)
    max_cells = rows * cols

    top_masks = [1 << (rows - 1 + c * col_height) for c in range(cols)]
    bottom_col_masks = [1 << (c * col_height) for c in range(cols)]
    column_masks = [((1 << rows) - 1) << (c * col_height) for c in range(cols)]
    move_order = sorted(range(cols), key=lambda c: abs(c - (cols - 1) / 2.0))

    def alignment(bb: int) -> bool:
        for shift in (1, col_height, col_height - 1, col_height + 1):
            bb2 = bb
            for _ in range(inarow - 1):
                bb2 = bb2 & (bb2 >> shift)
            if bb2:
                return True
        return False

    def can_play(mask: int, col: int) -> bool:
        return (mask & top_masks[col]) == 0

    def is_winning_move(position: int, mask: int, col: int) -> bool:
        move = (mask + bottom_col_masks[col]) & column_masks[col]
        return alignment(position | move)

    def play(position: int, mask: int, col: int):
        move = (mask + bottom_col_masks[col]) & column_masks[col]
        new_position = position ^ mask  # XOR against OLD mask -> becomes opponent's stones
        new_mask = mask | move
        return new_position, new_mask

    # Zugzwang / column-parity theory (Victor Allis, "A Knowledge-Based
    # Approach to Connect-Four", 1988 — the "Claimeven" rule): if a full
    # 7x6 game were played out to completion with strict bottom-up column
    # filling and no player ever skips a forced reply, every cell that is an
    # EVEN distance from the bottom of its column (2nd, 4th, 6th stone in
    # that column) is eventually claimed by whichever player is NOT first to
    # move overall — i.e. by Player 2 when Player 1 opens the game — because
    # Player 2 can always "pair up" with whatever odd-row stone Player 1 just
    # played directly below an even cell, guaranteeing they place the even
    # one. This is exactly what happened in the incident that motivated this
    # fix: the opponent (P2) built a hanging diagonal threat whose only open
    # cell was the 2nd-from-bottom (even) square of an otherwise-untouched
    # column; because we (P1) had no other legal move by that point, we were
    # forced to play the bottom (odd) cell of that column ourselves, handing
    # P2 the even cell directly above it on their very next move — a textbook
    # Claimeven loss.
    #
    # We approximate "is this threat cell claimable by parity" as: (row's
    # distance from the bottom of its column, 1-indexed) is even. Which side
    # that favors is fixed by the theory (always P2, never P1 or the current
    # mover), so heuristic_score below checks which real player — P1 or P2,
    # computed from the move count, not from "whose turn is it at this
    # negamax node" which flips every ply — owns each threat before applying
    # the parity bonus/penalty. This is a heuristic bias, not a full zugzwang
    # search (proving a parity claim exactly requires reasoning about the
    # whole board's column-pairing strategy, out of scope for a per-move
    # evaluation), but it is enough to make the search avoid *creating or
    # leaving open* an even-row threat for the opponent when equally good
    # alternatives exist, and to steer away from being the one who is forced
    # to fill the odd cell beneath one late in the game.
    def bottom_distance_1indexed(row: int) -> int:
        # Our bitboard convention has row 0 = bottom (see engine/bitboard.py),
        # matching Kaggle's board only after the top-down conversion done
        # below when reconstructing state — heuristic_score always receives
        # rows in this bottom-up convention, so distance-from-bottom is just
        # row + 1.
        return row + 1

    def heuristic_score(position: int, mask: int, moves: int) -> int:
        # Whose actual turn (P1 or P2, in the real game's fixed seating — not
        # just "the mover at this negamax node", which flips every ply and
        # says nothing about which side is genuinely first-player) is it to
        # play next, i.e. who does `position` belong to at this node? Move
        # number (moves + 1), 1-indexed, is odd for P1's moves and even for
        # P2's (P1 always plays move 1). This must be recomputed per node
        # (not captured from the outer my_is_first_player, which only tells
        # us whether *we* are P1) because heuristic_score is evaluated at
        # arbitrary depths where the mover alternates every ply.
        mover_is_p1 = ((moves + 1) % 2 == 1)
        opp_is_p1 = not mover_is_p1
        opp = mask ^ position
        score = 0
        for c in range(cols):
            for r in range(rows):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    r_end = r + dr * (inarow - 1)
                    c_end = c + dc * (inarow - 1)
                    if r_end < 0 or r_end >= rows or c_end < 0 or c_end >= cols:
                        continue
                    mine_ct = 0
                    theirs_ct = 0
                    empty_cell = None
                    for k in range(inarow):
                        rr = r + dr * k
                        cc = c + dc * k
                        bit = 1 << (cc * col_height + rr)
                        if position & bit:
                            mine_ct += 1
                        elif opp & bit:
                            theirs_ct += 1
                        else:
                            empty_cell = (rr, cc)
                    if mine_ct > 0 and theirs_ct > 0:
                        continue
                    if mine_ct == inarow - 1:
                        score += 25
                        # Symmetric Claimeven bonus: if OUR hanging threat's
                        # empty cell is an even row and WE are P2, that cell
                        # is ours by column-parity theory (see the penalty
                        # branch below for the full explanation) — the flat
                        # +25 undersells how strong this actually is.
                        if empty_cell is not None and not mover_is_p1:
                            er, ec = empty_cell
                            if bottom_distance_1indexed(er) % 2 == 0:
                                score += 60
                    elif mine_ct == 2:
                        score += 5
                    elif mine_ct == 1:
                        score += 1
                    elif theirs_ct == inarow - 1:
                        score -= 25
                        # Claimeven parity check (Victor Allis 1988): if a
                        # 7x6 game is played to completion with strict
                        # bottom-up column filling, every cell an EVEN
                        # distance from the bottom of its column is
                        # eventually claimed by Player 2, because P2 can
                        # always "pair up" underneath whatever odd-row stone
                        # P1 just played. A hanging opponent (theirs) threat
                        # whose one empty cell is even-row and belongs to P2
                        # (opp_is_p1 is False, i.e. opp == P2) is therefore
                        # not just "an open threat" but destined to fall to
                        # them regardless of tactics — this is exactly the
                        # pattern that caused the real loss this fix targets
                        # (see README's parity-loss writeup). Steeply
                        # penalize so search actively avoids being forced
                        # into opening that column.
                        if empty_cell is not None and not opp_is_p1:
                            er, ec = empty_cell
                            if bottom_distance_1indexed(er) % 2 == 0:
                                score -= 60
                    elif theirs_ct == 2:
                        score -= 5
                    elif theirs_ct == 1:
                        score -= 1
        return score

    tt = _tt_cache  # key -> (depth_stored, score, flag)  flag: 0=exact,1=lower,2=upper — reset logic below, before this is used

    def negamax(position: int, mask: int, moves: int, alpha: int, beta: int, depth_left: int) -> int:
        if moves == max_cells:
            return 0

        for col in move_order:
            if can_play(mask, col) and is_winning_move(position, mask, col):
                return (max_cells - moves + 1) // 2

        max_possible = (max_cells - moves - 1) // 2
        if beta > max_possible:
            beta = max_possible
            if alpha >= beta:
                return beta

        if depth_left <= 0:
            return heuristic_score(position, mask, moves)

        key = position + mask
        entry = tt.get(key)
        orig_alpha = alpha
        if entry is not None:
            depth_stored, score, flag = entry
            if depth_stored >= min(depth_left, max_cells - moves):
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

        best_score = -10**9
        for col in move_order:
            if not can_play(mask, col):
                continue
            child_pos, child_mask = play(position, mask, col)
            score = -negamax(child_pos, child_mask, moves + 1, -beta, -alpha, depth_left - 1)
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
        # Cap the cross-turn cache size (see the module-level _tt_cache
        # comment): it now survives across an entire game rather than being
        # thrown away every turn, so an unbounded dict is a real memory risk
        # over a full 42-ply game. Once over the cap, stop adding NEW entries
        # for the rest of this search call rather than evicting existing ones
        # — a full clear-and-restart would defeat the entire point of cross-
        # turn reuse, and 5M entries is already far more than a single-move
        # search realistically populates before its time budget runs out.
        if len(tt) < 5_000_000:
            tt[key] = (min(depth_left, max_cells - moves), best_score, flag)
        return best_score

    def solve_root(position: int, mask: int, moves: int, time_budget_s: float):
        deadline = time.monotonic() + time_budget_s
        cells_left = max_cells - moves
        legal = [c for c in move_order if can_play(mask, c)]
        if not legal:
            return {}
        for col in legal:
            if is_winning_move(position, mask, col):
                return {col: (max_cells - moves + 1) // 2}

        best_scores: dict[int, int] = {}
        depth = 1
        while depth <= cells_left:
            if time.monotonic() > deadline:
                break
            round_scores: dict[int, int] = {}
            timed_out_mid_round = False
            for col in legal:
                if time.monotonic() > deadline:
                    timed_out_mid_round = True
                    break
                child_pos, child_mask = play(position, mask, col)
                # Full window per sibling — see engine/fast_solver.py's
                # solve_root_at_depth docstring: narrowing alpha across
                # siblings combined with the heuristic depth cutoff produced
                # verified-wrong tied scores across unrelated root moves.
                s = -negamax(child_pos, child_mask, moves + 1, -10**9, 10**9, depth - 1)
                round_scores[col] = s
            if timed_out_mid_round:
                break
            best_scores = round_scores
            legal = sorted(legal, key=lambda c: -round_scores[c])
            depth += 1
        return best_scores

    # ---- Reconstruct bitboard state from Kaggle's flat board array ----
    # Kaggle's board is row-major from the TOP (index 0 = top-left), while our
    # bitboard row 0 = bottom. Convert: kaggle row kr (0=top) -> our row
    # (rows-1-kr). This is purely a coordinate transform; it does not affect
    # win detection (alignment() is direction-symmetric) or move legality
    # (can_play checks the top sentinel bit either way), only how we map the
    # observation into our internal representation.
    my_position = 0
    opp_position = 0
    mask = 0
    moves = 0
    opp_mark = 2 if my_mark == 1 else 1
    for kr in range(rows):
        for c in range(cols):
            cell = board[kr * cols + c]
            if cell == 0:
                continue
            r = rows - 1 - kr
            bit = 1 << (c * col_height + r)
            mask |= bit
            moves += 1
            if cell == my_mark:
                my_position |= bit
            elif cell == opp_mark:
                opp_position |= bit

    # Bitboard "position" convention: position = stones of the player TO
    # MOVE. It is always our turn when my_agent is called, so position = my
    # stones directly (no XOR juggling needed at this entry point — the XOR
    # trick in play() is only for advancing the state after a move).
    position = my_position

    # New-game detection: invalidate the cross-turn TT cache if this call
    # can't be a continuation of the same game we were last tracking. Two
    # signals, either one is sufficient to force a reset:
    #   1. Geometry changed (different rows/cols/inarow) — a stale TT keyed
    #      by (position + mask) under one geometry is meaningless (and could
    #      even alias) under another.
    #   2. moves < _tt_last_moves — stone count can only increase within a
    #      single game (Connect-X has no captures/removal), so a decrease
    #      means we're looking at an earlier or unrelated game state.
    # A false "same game" positive here is the dangerous direction (a stale
    # TT entry could return a confidently wrong score with no way to detect
    # it downstream), so both checks err toward resetting when in doubt.
    geometry = (rows, cols, inarow)
    if geometry != _tt_geometry or moves < _tt_last_moves:
        tt.clear()
    _tt_geometry = geometry
    _tt_last_moves = moves

    # Leave headroom under Kaggle's ~ (configured) per-move limit + byoyomi;
    # 1.8s per iterative-deepening call is a conservative default that leaves
    # margin for board-reconstruction overhead and Python's own call overhead
    # (no numba here, so raw speed is lower — budget accordingly).
    time_budget_s = float(getattr(configuration, "actTimeout", 2) or 2) - 0.3
    if time_budget_s < 0.3:
        time_budget_s = 0.3

    scores = solve_root(position, mask, moves, time_budget_s)
    if not scores:
        # Should not happen (there is always at least one legal move on a
        # non-full board), but never return an invalid/no move.
        for c in move_order:
            if can_play(mask, c):
                return c
        return 0

    best_col = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_col
