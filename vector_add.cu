#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 1024*1024*32  // Define the size of the vectors

__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void verifyResult(int *C) {
    for (int i = 0; i < SIZE; i++) {
        if (C[i] != SIZE) {  // Each element should be SIZE - 1
            printf("Error at index %d: expected %d, got %d\n", i, SIZE - 1, C[i]);
            return;
        }
    }
    printf("All results are correct!\n");
}

int main() {
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    int size = SIZE * sizeof(int);

    // CUDA event creation, used for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    for(int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i] = SIZE - i;
    }

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Start recording
    cudaEventRecord(start);

    int numThreadsPerBlock = 256;
    int blocksPerGrid = (SIZE + numThreadsPerBlock - 1) / numThreadsPerBlock;
    vectorAdd<<<blocksPerGrid, numThreadsPerBlock>>>(d_A, d_B, d_C, SIZE);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Stop recording
    cudaEventRecord(stop);

    // Calculate and print the execution time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time on GPU: %f milliseconds\n", milliseconds);

    verifyResult(C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}