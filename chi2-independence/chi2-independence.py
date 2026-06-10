import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.

    Parameters
    ----------
    C : 2D array-like
        Contingency table (observed frequencies)

    Returns
    -------
    (chi2, expected)
        chi2 : float
        expected : np.ndarray
    """
    C = np.asarray(C, dtype=float)

    row_totals = np.sum(C, axis=1)
    col_totals = np.sum(C, axis=0)
    total = np.sum(C)

    # Expected frequency table
    expected = np.outer(row_totals, col_totals) / total

    # Chi-square statistic
    chi2 = np.sum((C - expected) ** 2 / expected)

    return float(chi2), expected