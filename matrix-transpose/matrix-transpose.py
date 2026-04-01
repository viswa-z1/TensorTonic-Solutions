import numpy as np

def matrix_transpose(A):
    A = np.array(A)
    N, M = A.shape
    return A[:, :, None].swapaxes(0, 1).reshape(M, N)# Write your code here
pass