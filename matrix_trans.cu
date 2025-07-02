#include <stdio.h>
#include <cuda_runtime.h>
#include <cassert>

# define ROWS 1024
# define COLS 512
# define SIZE ROWS * COLS  // Define the size of the matrices

__global__ void matrixMul(int *A, int *B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the column index

    if (row < ROWS && col < COLS) {
        B[col * ROWS + row] = A[row * COLS + col];
    }
}


// Function to initialize matrices A and B
void initializeMatrices(int *A, int *B) {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            A[i * COLS + j] = i + j;
            B[j * ROWS + i] = 0;
        }
    }
}

// Function to verify the result of matrix transposition
void verifyResult(int *A, int *B) {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int a_idx = i * COLS + j;
            int b_idx = j * ROWS + i;
            assert(A[a_idx] == B[b_idx]);  // Check the result
        }
    }
    printf("All results are correct!\n");
}

int main() {
    int *A, *B;
    int *d_A, *d_B;

    size_t size = ROWS * COLS * sizeof(int);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    A = (int *)malloc(size);
    B = (int *)malloc(size);
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);

    initializeMatrices(A, B);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);  

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time on GPU: %f milliseconds\n", milliseconds);

    verifyResult(A, B);

    cudaFree(d_A);
    cudaFree(d_B);

    free(A);
    free(B);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}