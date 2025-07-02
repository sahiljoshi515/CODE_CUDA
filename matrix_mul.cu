#include <stdio.h>
#include <cuda_runtime.h>
#include <cassert>

#define SIZE 1024 * 1024  // Define the size of the matrices
#define N 1024  // Define the size of the matrices

__global__ void matrixMul(int *A, int *B, int *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the column index

    if (row < N && col < N) {
        int sum = 0;
        for(int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col]; // accumulate the product
        }
        C[row * N + col] = sum; // Store the result in C
    }
}


// Function to initialize matrices A, B, and C
void initializeMatrices(int *A, int *B, int *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i + j;
            B[i * N + j] = i - j;
            C[i * N + j] = 0;  // Initialize C to zero
        }
    }
}

// Function to verify the result of matrix multiplication
// This function checks if the result C is correct by performing the multiplication on the CPU
void verifyResult(int *A, int *B, int *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int temp = 0;
            for (int k = 0; k < N; k++) {
                temp += A[i * N + k] * B[k * N + j];
            }

            assert(C[i * N + j] == temp);  // Check the result
        }
    }
    printf("All results are correct!\n");
}

int main() {
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;

    size_t size = N * N * sizeof(int);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    initializeMatrices(A, B, C);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);   

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time on GPU: %f milliseconds\n", milliseconds);

    verifyResult(A, B, C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}