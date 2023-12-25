#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <math.h>

// Use "module load mpi devtoolset/9 Cuda11.4" on huff

// salloc --partition GPU --qos gpu --nodes 1 --ntasks-per-node 1 --cpus-per-task 1 --mem 64G --exclude huff44 ./test 3 > output.txt

float kernel_runtime;
float kernel_memcpy_runtime;

__device__ unsigned char inputChars[] = "abcdefghijklmnopqrstuvwxyz""ABCDEFGHIJKLMNOPQRSTUVWXYZ""0123456789";

__device__ int alphabetSize = 62;

__device__ unsigned long long totThr = 0;

// Input Length Combinations (62^length)
// This is without special characters " !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
// 1: 62
// 2: 3,844
// 3: 238,328
// 4: 14,776,336
// 5: 9,161,322,832
// 6: 56,800,235,584

// working length 3 1:1 mapping
// __global__ void kernel_MD2_brute(int passLen) {

//     // 2D
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     int totalComb = pow((double) alphabetSize, (double) passLen);

//     // 3D
//     // int tid = threadIdx.x + threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
//     atomicAdd(&totThr, 1);

//     // printf("TID: %d\n", tid);

//     if (tid < totalComb) {

//         unsigned char output[3];

//         if (tid < (totalComb/alphabetSize)) {

//             int firstIndex = tid/(totalComb/alphabetSize);
//             int subCombo = tid;
//             // int middleIndex = (tid/(tid/(totalComb/alphabetSize)))/alphabetSize;
//             int secondIndex = (subCombo/alphabetSize);
//             int thirdIndex = tid % alphabetSize;

//             output[0] = inputChars[firstIndex];
//             output[1] = inputChars[secondIndex];
//             output[2] = inputChars[thirdIndex];

//             // printf("Here: %s\n", output);

//         }
//         else {

//             int firstIndex = tid/(totalComb/alphabetSize);
//             int subCombo = tid - ((totalComb/alphabetSize) * (firstIndex));
//             // int middleIndex = (tid/(tid/(totalComb/alphabetSize)))/alphabetSize;
//             int secondIndex = (subCombo/alphabetSize);
//             int thirdIndex = tid % alphabetSize;

//             output[0] = inputChars[firstIndex];
//             output[1] = inputChars[secondIndex];
//             output[2] = inputChars[thirdIndex];

//             // printf("Tid: %d Middle Index: %d First Index: %d\n", tid, middleIndex, firstIndex);
//             // if (output[0] == '9'){
//             //     printf("Here: %s\n", output);
//             // }
//             // printf("Here: %s\n", output);

//         }
//         free(output);
//     }
// }

// currently working for a password length of 4
// __global__ void kernel_MD2_brute(int passLen) {

//     // 2D
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;

//     int totalComb = pow((double) alphabetSize, (double) passLen);

//     // 3D
//     // int tid = threadIdx.x + threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
//     atomicAdd(&totThr, 1);

//     // printf("TID: %d\n", tid);

//     if (tid < totalComb) {

//         // unsigned char* output = (unsigned char*) malloc(passLen * sizeof(unsigned char));

//         unsigned char output[4];

//         // cudaMalloc((void**)&output, passLen*sizeof(unsigned char));

//         if (tid < (totalComb/alphabetSize)) {

//             int firstIndex = tid/(totalComb/alphabetSize);
//             int subCombo = tid;
//             // int subCombo = tid - ((totalComb/alphabetSize) * (firstIndex));
//             // int middleIndex = (tid/(tid/(totalComb/alphabetSize)))/alphabetSize;
//             // int middleIndex = (subCombo/alphabetSize);
//             int secondIndex = (subCombo/(alphabetSize*alphabetSize));
//             int thirdIndex = (tid/alphabetSize) % alphabetSize;
//             int lastIndex = tid % alphabetSize;

//             output[0] = inputChars[firstIndex];
//             output[1] = inputChars[secondIndex];
//             output[2] = inputChars[thirdIndex];
//             output[3] = inputChars[lastIndex];

//             // printf("Here: %s\n", output);

//             // if (output[1] == '9'){
//             //     printf("Here: %s\n", output);
//             // }
//             // free(output);
//         }
//         else {

//             int firstIndex = tid/(totalComb/alphabetSize);
//             int subCombo = tid - ((totalComb/alphabetSize) * (firstIndex));
//             // int middleIndex = (tid/(tid/(totalComb/alphabetSize)))/alphabetSize;
//             // int middleIndex = (subCombo/alphabetSize);
//             int secondIndex = (subCombo/(alphabetSize*alphabetSize));
//             int thirdIndex = (subCombo/alphabetSize) % alphabetSize;
//             int lastIndex = tid % alphabetSize;

//             output[0] = inputChars[firstIndex];
//             output[1] = inputChars[secondIndex];
//             output[2] = inputChars[thirdIndex];
//             output[3] = inputChars[lastIndex];

//             // printf("Tid: %d Middle Index: %d First Index: %d\n", tid, middleIndex, firstIndex);
//             // if (output[0] == 'b' && output[1] == '9'){
//             //     printf("Here: %s\n", output);
//             // }
//             // free(output);
//         }
//         free(output);
//     }
// }

// Working on a password length of 5
// __global__ void kernel_MD2_brute(int passLen) {

//     // 2D
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;

//     int totalComb = pow((double) alphabetSize, (double) passLen);

//     // 3D
//     // int tid = threadIdx.x + threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
//     atomicAdd(&totThr, 1);

//     // printf("TID: %d\n", tid);

//     if (tid < totalComb) {

//         // unsigned char* output = (unsigned char*) malloc(passLen * sizeof(unsigned char));

//         unsigned char output[5];

//         if (tid < (totalComb/alphabetSize)) {

//             int firstIndex = tid/(totalComb/alphabetSize);
//             int secondIndex = (tid/(alphabetSize*alphabetSize*alphabetSize));
//             int thirdIndex = (tid/(alphabetSize*alphabetSize)) % alphabetSize;
//             int fourthIndex = (tid/(alphabetSize)) % (alphabetSize);
//             int lastIndex = tid % alphabetSize;

//             output[0] = inputChars[firstIndex];
//             output[1] = inputChars[secondIndex];
//             output[2] = inputChars[thirdIndex];
//             output[3] = inputChars[fourthIndex];
//             output[4] = inputChars[lastIndex];


//             // printf("Here: %s\n", output);

//             // if (output[0] == 'a' && output[1] == 'a' && output[2] == 'c'){
//             //     printf("Here: %s\n", output);
//             // }

//         }
//         else {

//             int firstIndex = tid/(totalComb/alphabetSize);
//             int subCombo = tid - ((totalComb/alphabetSize) * (firstIndex));
//             int secondIndex = (subCombo/(alphabetSize*alphabetSize*alphabetSize));
//             int thirdIndex = (subCombo/(alphabetSize*alphabetSize)) % alphabetSize;
//             int fourthIndex = (subCombo/(alphabetSize)) % (alphabetSize);
//             int lastIndex = tid % alphabetSize;

//             output[0] = inputChars[firstIndex];
//             output[1] = inputChars[secondIndex];
//             output[2] = inputChars[thirdIndex];
//             output[3] = inputChars[fourthIndex];
//             output[4] = inputChars[lastIndex];

//             // printf("Tid: %d Middle Index: %d First Index: %d\n", tid, middleIndex, firstIndex);
//             // if (output[0] == '9' && output[1] == '9'){
//             //     printf("Here: %s\n", output);
//             // }

//             if (output[0] == '9' && output[1] == '9' && output[2] == '9'){
//                 printf("Here: %s\n", output);
//             }
//         }
//         free(output);
//     }
// }

// this will be length 6
__global__ void kernel_MD2_brute(int passLen) {

    // 2D
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int totalComb = pow((double) alphabetSize, (double) passLen);

    // 3D
    // int tid = threadIdx.x + threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
    atomicAdd(&totThr, 1);

    // printf("TID: %d\n", tid);

    if (tid < totalComb) {

        // unsigned char* output = (unsigned char*) malloc(passLen * sizeof(unsigned char));

        unsigned char output[5];

        if (tid < (totalComb/alphabetSize)) {

            int firstIndex = tid/(totalComb/alphabetSize);
            int secondIndex = (tid/(alphabetSize*alphabetSize*alphabetSize*alphabetSize));
            int thirdIndex = (tid/(alphabetSize*alphabetSize*alphabetSize)) % alphabetSize;
            int fourthIndex = (tid/(alphabetSize*alphabetSize)) % (alphabetSize);
            int fifthIndex = (tid/(alphabetSize)) % (alphabetSize);
            int lastIndex = tid % alphabetSize;

            output[0] = inputChars[firstIndex];
            output[1] = inputChars[secondIndex];
            output[2] = inputChars[thirdIndex];
            output[3] = inputChars[fourthIndex];
            output[4] = inputChars[fifthIndex];
            output[5] = inputChars[lastIndex];


            // printf("Here: %s\n", output);

            // if (output[0] == 'a' && output[1] == 'a' && output[2] == 'c'){
            //     printf("Here: %s\n", output);
            // }

        }
        else {

            int firstIndex = tid/(totalComb/alphabetSize);
            int subCombo = tid - ((totalComb/alphabetSize) * (firstIndex));
            int secondIndex = (subCombo/(alphabetSize*alphabetSize*alphabetSize*alphabetSize));
            int thirdIndex = (subCombo/(alphabetSize*alphabetSize*alphabetSize)) % alphabetSize;
            int fourthIndex = (subCombo/(alphabetSize*alphabetSize)) % (alphabetSize);
            int fifthIndex = (tid/(alphabetSize)) % (alphabetSize);
            int lastIndex = tid % alphabetSize;

            output[0] = inputChars[firstIndex];
            output[1] = inputChars[secondIndex];
            output[2] = inputChars[thirdIndex];
            output[3] = inputChars[fourthIndex];
            output[4] = inputChars[fifthIndex];
            output[5] = inputChars[lastIndex];

            // printf("Tid: %d Middle Index: %d First Index: %d\n", tid, middleIndex, firstIndex);
            // if (output[0] == '9' && output[1] == '9'){
            //     printf("Here: %s\n", output);
            // }

            // if (output[0] == 'd' && output[1] == 'd' && output[2] == 'c'){
            //     printf("Here: %s\n", output);
            // }
        }
        free(output);
    }
}

void BFS(int passLen) {   

    // Declare timers
    cudaEvent_t startEvent, stopEvent;
    cudaEvent_t startEventKernel, stopEventKernel;
    cudaEventCreate(&startEvent);           cudaEventCreate(&stopEvent);
    cudaEventCreate(&startEventKernel);     cudaEventCreate(&stopEventKernel);

    // Start timer measuring kernel + memory copy times
    cudaEventRecord(startEvent, 0);
    
    // Start timer measuring kernel time only
    cudaEventRecord(startEventKernel, 0);

    int alphabetSize = 62;
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (alphabetSize * alphabetSize - 1) / alphabetSize;

    // this needs to be the total combination of passwords (total comb * 62)
    int totalComb = pow((double) alphabetSize, (double) passLen - 1);

    // dim3 gridDim;
    // dim3 blockDim;

    // printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    kernel_MD2_brute<<<totalComb,alphabetSize>>>(passLen);

    // Wait for kernel completion
    cudaDeviceSynchronize();

    int devNo = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);
    printf("Maximum grid size is: (");
    for (int i = 0; i < 3; i++)
        printf("%d\t", iProp.maxGridSize[i]);
    printf(")\n");
    printf("Maximum block dim is: (");
    for (int i = 0; i < 3; i++)
        printf("%d\t", iProp.maxThreadsDim[i]);
    printf(")\n");
    printf("Max threads per block: %d\n", iProp.maxThreadsPerBlock);

    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess ) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

        // Possibly: exit(-1) if program cannot continue....
    }

    unsigned long long total;
    cudaMemcpyFromSymbol(&total, totThr, sizeof(unsigned long long));
    printf("Total threads counted: %lu\n", total);

    // Stop timer measuring kernel time only
    cudaEventRecord(stopEventKernel, 0);

    // Stop timer measuring kernel + memory copy times
    cudaEventRecord(stopEvent, 0);
    
    // Calculate elapsed time
    cudaEventSynchronize(stopEvent);
    cudaEventSynchronize(stopEventKernel);
    cudaEventElapsedTime(&kernel_memcpy_runtime, startEvent, stopEvent);
    cudaEventElapsedTime(&kernel_runtime, startEventKernel, stopEventKernel);
}

int main(int argc, char *argv[]) {
    if(argc != 2)
    {
        printf("Usage: ./program passLen\n");
        exit(0);
    }

    int passLen = strtol(argv[1], NULL, 10);
    
    // Initialize time measurement
    float time_difference;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    
    BFS(passLen);

    // Stop time measurement
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time_difference, startEvent, stopEvent);
    
    printf("Single GPU: Required %f ms total time. Kernel and memcpy required %f ms. Kernel only required %f ms.\n", time_difference, kernel_memcpy_runtime, kernel_runtime);
}