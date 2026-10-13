#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    float acc = 0.0f;

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; ++t) {
        int k0 = t * TILE_DIM;

        int aCol = k0 + threadIdx.x;
        int bRow = k0 + threadIdx.y;

        if (row < M && aCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

extern "C" __attribute__((visibility("default")))
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    tiled_matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}