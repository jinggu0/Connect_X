"""Bitboard representation for Connect-4-style games (arbitrary rows/columns/inarow).

Uses the classic Connect-4 bitboard encoding (as popularized by John Tromp and
Pascal Pons' solvers): each column occupies (rows + 1) bits, with the extra bit
per column acting as a sentinel to make win-detection and "column full" checks
branch-free bit operations instead of per-cell array scans.

Board layout for a board with R rows, C columns (bit index = col * (R + 1) + row,
row 0 = bottom):

    col0        col1        col2   ...
    bit R       bit 2R+1    bit 3R+2   <- sentinel row (always 0 for legal boards)
    bit R-1     bit 2R      bit 3R+1
    ...
    bit 0       bit R+1     bit 2R+2

Two bitboards are kept: `mask` (all occupied cells, either player) and
`position` (cells occupied by the player to move, XORed after each ply so it
always represents "current player's stones" from the mover's perspective —
this halves the win-check surface since we only ever need to check the player
who just moved).
"""

from __future__ import annotations


class Geometry:
    """Precomputed shift/mask tables for a given (rows, columns, inarow) config."""

    __slots__ = ("rows", "cols", "inarow", "col_height", "bottom_mask", "board_mask", "full_col_top")

    def __init__(self, rows: int, cols: int, inarow: int):
        self.rows = rows
        self.cols = cols
        self.inarow = inarow
        self.col_height = rows + 1  # +1 sentinel bit per column

        bottom = 0
        board = 0
        for c in range(cols):
            base = c * self.col_height
            bottom |= 1 << base
            for r in range(rows):
                board |= 1 << (base + r)
        self.bottom_mask = bottom
        self.board_mask = board
        # top playable bit index for each column (the sentinel bit)
        self.full_col_top = [c * self.col_height + rows for c in range(cols)]


def top_mask(geo: Geometry, col: int) -> int:
    return 1 << (geo.rows - 1 + col * geo.col_height)


def bottom_mask_col(geo: Geometry, col: int) -> int:
    return 1 << (col * geo.col_height)


def column_mask(geo: Geometry, col: int) -> int:
    return ((1 << geo.rows) - 1) << (col * geo.col_height)


class Board:
    """Mutable bitboard game state. `position` = current mover's stones,
    `mask` = all stones. XOR trick: after playing, position ^= mask reinterprets
    the board from the *next* mover's perspective without needing a color flag.
    """

    __slots__ = ("geo", "position", "mask", "moves", "heights")

    def __init__(self, geo: Geometry):
        self.geo = geo
        self.position = 0
        self.mask = 0
        self.moves = 0
        # heights[c] = number of stones dropped in column c (next free row)
        self.heights = [c * geo.col_height for c in range(geo.cols)]

    def clone(self) -> "Board":
        b = Board.__new__(Board)
        b.geo = self.geo
        b.position = self.position
        b.mask = self.mask
        b.moves = self.moves
        b.heights = self.heights[:]
        return b

    def can_play(self, col: int) -> bool:
        return (self.mask & top_mask(self.geo, col)) == 0

    def is_winning_move(self, col: int) -> bool:
        pos = self.position | ((self.mask + bottom_mask_col(self.geo, col)) & column_mask(self.geo, col))
        return alignment(pos, self.geo)

    def play(self, col: int) -> None:
        move = (self.mask + bottom_mask_col(self.geo, col)) & column_mask(self.geo, col)
        # Order matters: XOR against the OLD mask (mover's stones | opponent's
        # stones, NOT yet including this move) so position_new = mask_old ^
        # position_old = opponent's stones exactly. XOR-ing after adding the
        # move to mask would incorrectly fold the mover's brand-new stone into
        # the "next player's stones" bitboard (verified by hand-trace + the
        # small-board brute-force regression test).
        self.position ^= self.mask
        self.mask |= move
        self.moves += 1

    def key(self) -> int:
        """Unique position key for transposition tables (position + mask encodes
        both players' stones and whose turn, collision-free)."""
        return self.position + self.mask

    def possible_moves_mask(self) -> int:
        return (self.mask + self.geo.bottom_mask) & self.geo.board_mask

    def is_draw(self) -> bool:
        return self.moves == self.geo.rows * self.geo.cols


def alignment(bb: int, geo: Geometry) -> bool:
    """True if bitboard `bb` contains `inarow` consecutive set bits in any
    of the four directions (vertical, horizontal, two diagonals).

    Standard bit-trick: repeatedly AND the bitboard with itself shifted by
    `shift`, n-1 times. After k iterations, a bit set at position p means the
    original bb has (k+1) consecutive set bits starting at p in that
    direction. This runs in O(inarow) shifts regardless of board size.
    """
    n = geo.inarow
    h = geo.col_height
    for shift in (1, h, h - 1, h + 1):  # vertical, horizontal, diag /, diag \
        bb2 = bb
        for _ in range(n - 1):
            bb2 = bb2 & (bb2 >> shift)
        if bb2:
            return True
    return False
