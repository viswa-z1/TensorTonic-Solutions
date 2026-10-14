import numpy as np

def go_group_liberties(board: list, row: int, col: int) -> tuple:
    """
    Returns: a tuple of two sorted coordinate lists: group and liberties.
    """
    # Board dimensions
    if isinstance(board, np.ndarray):
        R, C = board.shape
        b = board
    else:
        R = len(board)
        C = len(board[0]) if R > 0 else 0
        b = board

    colour = b[row][col]
    # Occupied starting point required; treat 0 as no group
    if colour == 0:
        return ([], [])

    group = []
    liberties = set()
    visited = set()
    stack = [(row, col)]
    visited.add((row, col))

    while stack:
        r, c = stack.pop()
        group.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                v = b[nr][nc]
                if v == 0:
                    liberties.add((nr, nc))
                elif v == colour and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))

    group.sort()
    liberties_sorted = sorted(liberties)
    return (group, liberties_sorted)