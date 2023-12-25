#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>

// Use "module load mpi devtoolset/9 Cuda11.4" on huff

// salloc --partition GPU --qos gpu --nodes 1 --ntasks-per-node 1 --cpus-per-task 1 --mem 64G --exclude huff44 ./test 3 > output.txt

float kernel_runtime;
float kernel_memcpy_runtime;

typedef struct MD2 {
    unsigned char checksum[16];
    unsigned char buffer[16];
    unsigned char block[48];
    int length;
} MD2;

__device__ unsigned char inputChars[] = "abcdefghijklmnopqrstuvwxyz""ABCDEFGHIJKLMNOPQRSTUVWXYZ""0123456789";

__device__ int alphabetSize = 62;

__device__ unsigned char S[256] = {
	41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6, 19, 
    98, 167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188, 76, 130, 202,
    30, 155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24, 138, 23, 229, 18,
    190, 78, 196, 214, 218, 158, 222, 73, 160, 251, 245, 142, 187, 47, 238, 122, 
    169, 104, 121, 145, 21, 178, 7, 63, 148, 194, 16, 137, 11, 34, 95, 33,
    128, 127, 93, 154, 90, 144, 50, 39, 53, 62, 204, 231, 191, 247, 151, 3,
    255, 25, 48, 179, 72, 165, 181, 209, 215, 94, 146, 42, 172, 86, 170, 198,
    79, 184, 56, 210, 150, 164, 125, 182, 118, 252, 107, 226, 156, 116, 4, 241,
    69, 157, 112, 89, 100, 113, 135, 32, 134, 91, 207, 101, 230, 45, 168, 2,
    27, 96, 37, 173, 174, 176, 185, 246, 28, 70, 97, 105, 52, 64, 126, 15, 
    85, 71, 163, 35, 221, 81, 175, 58, 195, 92, 249, 206, 186, 197,	234, 38,
    44, 83, 13, 110, 133, 40, 132, 9, 211, 223, 205, 244, 65, 129, 77, 82,
    106, 220, 55, 200, 108, 193, 171, 250, 36, 225, 123, 8, 12, 189, 177, 74,
    120, 136, 149, 139, 227, 99, 232, 109, 233, 203, 213, 254, 59, 0, 29, 57,
    242, 239, 183, 14, 102, 88, 208, 228, 166, 119, 114, 248, 235, 117, 75, 10,
    49, 68, 80, 180, 143, 237, 31, 26, 219, 153, 141, 51, 159, 17, 131, 20
};

/***************************************************/
/*** WRITE THE CODE OF THE KERNEL FUNCTIONS HERE ***/
/***************************************************/

__device__ void transform(MD2 *context, unsigned char input[]) {
	int i,j,k,t;

	// this is going to update the block info based on any given input 
    // so in the case of the program it will first have the input buffer read into it which is our actual input
    // then it will have the check sum appended to it in th final call leading us to our output
	for (i=0; i < 16; i++) {
        // assigning the block our current data
        // then updating to get our encryption block from (context->block[16 + i] ^ context->block[i])
		context->block[16 + i] = input[i];
		context->block[32 + i] = (context->block[16 + i] ^ context->block[i]);
	}
	 
	t = 0;
	// this will do 18 rounds (the encrypting of the block)
	for (j = 0; j < 18; j++) {
		for (k = 0; k < 48; k++) {
			t = context->block[k] ^= S[t];
		}
		// setting t to (t+j) mod 256
		t = (t+j) % 256;
	}

	// updates the checksum for each call
	t = context->checksum[15];
	for (i=0; i < 16; i++) {
		context->checksum[i] = S[input[i] ^ t];
        t = context->checksum[i];
	}
}
// this will initialize the checksum and block values for the MD2
__device__ void initialize(MD2 *context) {
    // our block getting initialized
    memset(context->block, 0, 48);
    // our checksum getting initialized
    memset(context->checksum, 0, 16);
    // set out inital length to zero
	context->length = 0;
}

__device__ void update(MD2 *context, const unsigned char input[], size_t length) {

	// iterates through the length of our password and updates the input[i]
    // meaning we are putting each index of our password
	for (int i = 0; i < length; i++) {
		// input[i] is the current char in our password
		// so we are setting context->buffer to that char
		context->buffer[context->length] = input[i];
        // printf("Here: %d\n", context->length);
		// iterating our length
        context->length++;

		// once our input length is equal to 16 we will call the transform method
		if (context->length == 16) {
            // calls transform on the data array that is full before we start writing over the 
            // values with the continuation of the for loop
			transform(context, context->buffer);
            // need to zero out our index if it excedes the block size as we cannot go over 16
            context->length = 0;
		}
    }
}

__device__ void final(MD2 *context, unsigned char hash[]) {

    // determines the amount of padding that our password will need
    // and will always be less than 16 because of the catch in the update function that zeroes out context->length
	// printf("Here: %d\n", context->length);

    // the idea is that there is always going to be at least some padding to append to our data even if the 
    // length is 15 resulting in 1 pad length
    int padLength = 16 - context->length;

    // append the padding bytes for the final transformations 
    for(int i = context->length; i < 16; i++){
        context->buffer[i] = padLength;
    }

    // final transform of data
	transform(context, context->buffer);
    // then the final transform of our cheksum
	transform(context, context->checksum);

    // stores the final hash output of length 16
	memcpy(hash, context->block, 16);
}

// __device__ void bruteForce(unsigned char* output, int index, int passLen) {

//     // unsigned char* currentPass = (unsigned char*) malloc((passLen + 1) * sizeof(unsigned char));

//     int tid = (blockDim.x * blockIdx.x + threadIdx.x) % 62;

//     // blockIdx.x

//     for (int i = 0; i < alphabetSize; ++i) {
        
//         output[index] = inputChars[i];

//         // printf("Inner index: %d\n", index);

//         if (index == passLen - 1) {

//             // prints the current password guess
//             output[passLen] = '\0';
//             printf("Password Hello: %s\n", output);

//         }                                          
//         else {

//             // recursive call of the function so we can iterate the index
//             bruteForce(output, index + 1, passLen);

//         }
//     }
// }

// a hard coded brute force on the GPU kernel as there are memory restrictions 
// when doing recursive problems on it
__global__ void kernel_MD2_brute(int index, int passLen) {

    unsigned char* currentPass = (unsigned char*) malloc((passLen + 1) * sizeof(unsigned char));

    unsigned char* output = (unsigned char*) malloc(passLen * sizeof(unsigned char));

    int tid = (blockDim.x * blockIdx.x + threadIdx.x) % 62;

    // printf("Here is TID: %d\n", tid);

    for (int i = tid; i < tid + 1; i += 256) {
    
        //for (int i = 0; i < alphabetSize; ++i) {

        // printf("Index: %d\n", index);
        
        output[index] = inputChars[i];

        //output[index] = inputChars[i];
        for (int j = 0; j < alphabetSize; ++j) {
            index = 1;
            // index +=1;
            output[index] = inputChars[j];
            for (int k = 0; k < alphabetSize; ++k) {
                index = 2;
                output[index] = inputChars[k];
                // printf("Here: %s\n", output);
                // for (int l = 0; l < alphabetSize; l++) {
                //     index = 3;
                //     output[index] = inputChars[l];

                for (int j = 0; j < passLen; j++) {
                    currentPass[j] = output[j];
                }

                currentPass[passLen] = '\0';

                // prints the current password guess
                output[passLen] = '\0';
                // printf("Password: %s\n", output);

                // creating a buffer array that will store the output of the hash function
                unsigned char buf[16];
                // declaring our context that is going to be used for the MD2 calls 
                MD2 context;

                // calling of the MD2 hash function
                initialize(&context);
                update(&context, currentPass, passLen);
                final(&context, buf);

                // when you get the hash back from the function it needs to be converted to a char
                // so that we can then compare it to a hypothetical input hash

                // no sprintf on the GPU
                // printf("Hash: %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n",
                // buf[0], buf[1], buf[2], buf[3], 
                // buf[4], buf[5], buf[6], buf[7], 
                // buf[8], buf[9], buf[10], buf[11], 
                // buf[12], buf[13], buf[14], buf[15]);

                //}
            }
        }
    }
    free(output);
}

// this is the non hard coded brute force alg that is recursive but runs out of memory
// 10 million iterations into length 4 passwords
// __global__ void kernel_bruteForce(int index, int passLen) {

//     unsigned char* output = (unsigned char*) malloc(passLen * sizeof(unsigned char));

//     int tid = (blockDim.x * blockIdx.x + threadIdx.x) % 62;

//     // printf("Here is TID: %d\n", tid);

//     for (int i = tid; i < tid + 1; i += 256) {
    
//         //for (int i = 0; i < alphabetSize; i++) {
//         output[index] = inputChars[i];
//         // printf("Index: %d\n", index);
//         bruteforce(output, index + 1, passLen);
            
//     }
//     free(output);
// }
 
// Implements a kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
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

    // Execute the kernels
    /***********************************************/
    /*** WRITE THE CODE OF THE KERNEL CALLS HERE ***/
    /***********************************************/
    // the threads we can change to evaluate different run times

    // int alphabetSize = 62;
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (alphabetSize + alphabetSize - 1) / alphabetSize;

    // kernel_bruteForce<<<1,62>>>(0, passLen);
    kernel_MD2_brute<<<1,62>>>(0, passLen);

    // Wait for kernel completion
    cudaDeviceSynchronize();
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