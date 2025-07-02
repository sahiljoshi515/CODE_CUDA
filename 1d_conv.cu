#include <stdio.h>
#include <cuda_runtime.h>
#include <cassert>

# define INPUT 4096
# define KERNEL 1024
# define OUTPUT (INPUT - KERNEL + 1)

__global__ void conv1d(int *input, int *kernel, int *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the index of the output element
    if (idx < OUTPUT) { // Ensure we don't go out of bounds
        int sum = 0;
        for (int j = 0; j < KERNEL; j++) {
            sum += input[idx + j] * kernel[j]; // Perform the convolution operation
        }
        output[idx] = sum; // Store the result in output
    }
}


// Function to initialize the colors in the image
void initializeConvolution(int *input, int *kernel, int *output) {
    for (int i = 0; i < INPUT; i++) {
        input[i] = 1.0; // Initialize input with some values
    }
    for (int i = 0; i < KERNEL; i++) {
        kernel[i] = 2.0; // Initialize kernel with some values 
    }
    for (int i = 0; i < INPUT - KERNEL + 1; i++) {
        output[i] = 0.0f; // Initialize output to zero
    }
}

void verifyResult(int *input, int *kernel, int *output) {
    for (int i = 0; i < OUTPUT; i++) {
        int expected = 0;
        for (int j = 0; j < KERNEL; j++) {
            expected += input[i + j] * kernel[j]; // Perform the convolution operation
        }
        assert(output[i] == expected); // Check the result
    }
    printf("All results are correct!\n");
}

int main() {
    int *input, *kernel, *output;
    int *d_input, *d_kernel, *d_output;

    size_t input_size = INPUT * sizeof(int);
    size_t kernel_size = KERNEL * sizeof(int);
    size_t output_size = OUTPUT * sizeof(int);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    input = (int *)malloc(input_size);
    kernel = (int *)malloc(kernel_size);
    output = (int *)malloc(output_size);
    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_kernel, kernel_size);
    cudaMalloc((void **)&d_output, output_size);

    initializeConvolution(input, kernel, output);

    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice); 

    int threadsPerBlock = 256; // Number of threads per block
    int blocksPerGrid = (OUTPUT + threadsPerBlock - 1) / threadsPerBlock; // Calculate the number of blocks needed

    cudaEventRecord(start);

    conv1d<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output);

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time on GPU: %f milliseconds\n", milliseconds);

    verifyResult(input, kernel, output);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(input);
    free(kernel);
    free(output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}