#include <stdio.h>
#include <cuda_runtime.h>
#include <cassert>

# define WIDTH 4096
# define HEIGHT 1024
# define N WIDTH * HEIGHT * 4

__global__ void colorInv(unsigned char* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the index of the pixel
    if (idx < WIDTH * HEIGHT) { // Ensure we don't go out of bounds
        int base = idx * 4; // Each pixel has 4 components (RGBA)

        image[base + 0] = 255 - image[base + 0]; // Invert Red channel
        image[base + 1] = 255 - image[base + 1]; // Invert Green channel
        image[base + 2] = 255 - image[base + 2]; // Invert Blue channel
        // Alpha channel remains unchanged
    }
}


// Function to initialize the colors in the image
void initializeColors(unsigned char* image) {
    for (int i = 0; i < N; i++) {
        image[i] = (unsigned char)(i % 256); // Initialize with some values
    }
}

void verifyResult(unsigned char* image) {
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        int base = i * 4;
        assert(image[base + 0] == (255 - (i * 4 + 0) % 256)); // R
        assert(image[base + 1] == (255 - (i * 4 + 1) % 256)); // G
        assert(image[base + 2] == (255 - (i * 4 + 2) % 256)); // B
        assert(image[base + 3] == (i * 4 + 3) % 256);         // A stays unchanged
    }
    printf("All results are correct!\n");
}

int main() {
    unsigned char* image;
    unsigned char* d_image;

    size_t size = N * sizeof(unsigned char);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    image = (unsigned char *)malloc(size);
    cudaMalloc((void **)&d_image, size);

    initializeColors(image);

    cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice); 

    int threadsPerBlock = 256; // Number of threads per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Calculate the number of blocks needed

    cudaEventRecord(start);

    colorInv<<<blocksPerGrid, threadsPerBlock>>>(d_image);

    cudaMemcpy(image, d_image, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time on GPU: %f milliseconds\n", milliseconds);

    verifyResult(image);

    cudaFree(d_image);

    free(image);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}