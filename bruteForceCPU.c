#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <sys/time.h>
#include <math.h>

// the buffer is going to hold our current input 
// and the block is going to hold our data that will eventually be output 
typedef struct MD2 {
    unsigned char checksum[16];
    unsigned char buffer[16];
    unsigned char block[48];
    int length;
} MD2;

// Input Length Combinations (62^length)
// This is without special characters " !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
// 1: 62
// 2: 3,844
// 3: 238,328
// 4: 14,776,336
// 5: 9,161,322,832
// 6: 56,800,235,584

// Below is the S-table in the initial hex values 
//{ 0x29, 0x2E, 0x43, 0xC9, 0xA2, 0xD8, 0x7C, 0x01, 0x3D, 0x36, 0x54, 0xA1, 0xEC, 0xF0, 0x06, 0x13, 
//   0x62, 0xA7, 0x05, 0xF3, 0xC0, 0xC7, 0x73, 0x8C, 0x98, 0x93, 0x2B, 0xD9, 0xBC, 0x4C, 0x82, 0xCA, 
//   0x1E, 0x9B, 0x57, 0x3C, 0xFD, 0xD4, 0xE0, 0x16, 0x67, 0x42, 0x6F, 0x18, 0x8A, 0x17, 0xE5, 0x12, 
//   0xBE, 0x4E, 0xC4, 0xD6, 0xDA, 0x9E, 0xDE, 0x49, 0xA0, 0xFB, 0xF5, 0x8E, 0xBB, 0x2F, 0xEE, 0x7A, 
//   0xA9, 0x68, 0x79, 0x91, 0x15, 0xB2, 0x07, 0x3F, 0x94, 0xC2, 0x10, 0x89, 0x0B, 0x22, 0x5F, 0x21,
//   0x80, 0x7F, 0x5D, 0x9A, 0x5A, 0x90, 0x32, 0x27, 0x35, 0x3E, 0xCC, 0xE7, 0xBF, 0xF7, 0x97, 0x03, 
//   0xFF, 0x19, 0x30, 0xB3, 0x48, 0xA5, 0xB5, 0xD1, 0xD7, 0x5E, 0x92, 0x2A, 0xAC, 0x56, 0xAA, 0xC6, 
//   0x4F, 0xB8, 0x38, 0xD2, 0x96, 0xA4, 0x7D, 0xB6, 0x76, 0xFC, 0x6B, 0xE2, 0x9C, 0x74, 0x04, 0xF1, 
//   0x45, 0x9D, 0x70, 0x59, 0x64, 0x71, 0x87, 0x20, 0x86, 0x5B, 0xCF, 0x65, 0xE6, 0x2D, 0xA8, 0x02, 
//   0x1B, 0x60, 0x25, 0xAD, 0xAE, 0xB0, 0xB9, 0xF6, 0x1C, 0x46, 0x61, 0x69, 0x34, 0x40, 0x7E, 0x0F, 
//   0x55, 0x47, 0xA3, 0x23, 0xDD, 0x51, 0xAF, 0x3A, 0xC3, 0x5C, 0xF9, 0xCE, 0xBA, 0xC5, 0xEA, 0x26, 
//   0x2C, 0x53, 0x0D, 0x6E, 0x85, 0x28, 0x84, 0x09, 0xD3, 0xDF, 0xCD, 0xF4, 0x41, 0x81, 0x4D, 0x52, 
//   0x6A, 0xDC, 0x37, 0xC8, 0x6C, 0xC1, 0xAB, 0xFA, 0x24, 0xE1, 0x7B, 0x08, 0x0C, 0xBD, 0xB1, 0x4A, 
//   0x78, 0x88, 0x95, 0x8B, 0xE3, 0x63, 0xE8, 0x6D, 0xE9, 0xCB, 0xD5, 0xFE, 0x3B, 0x00, 0x1D, 0x39, 
//   0xF2, 0xEF, 0xB7, 0x0E, 0x66, 0x58, 0xD0, 0xE4, 0xA6, 0x77, 0x72, 0xF8, 0xEB, 0x75, 0x4B, 0x0A, 
//   0x31, 0x44, 0x50, 0xB4, 0x8F, 0xED, 0x1F, 0x1A, 0xDB, 0x99, 0x8D, 0x33, 0x9F, 0x11, 0x83, 0x14 }

// our S-tables turn to decimals from the initial hex

// the message is padded so that no matter its length it is a multiple of 16 bytes long
// it is always performed even if the input is already of the corrent length 
static const unsigned char S[256] = {
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

// this transform is going to use the same 256-byte permutation S as we have used before
// the basic idea of the math is given as pseudocode in the paper written by the creator as linked in my paper
void transform(MD2 *context, unsigned char input[]) {
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
void initialize(MD2 *context) {
    // our block getting initialized
    memset(context->block, 0, 48);
    // our checksum getting initialized
    memset(context->checksum, 0, 16);
    // set out inital length to zero
	context->length = 0;
}

void update(MD2 *context, const unsigned char input[], size_t length) {

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

void final(MD2 *context, unsigned char hash[]) {

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

void bruteForce(unsigned char* output, int index, int passLen) {

    // the possible chars to have in a password
    unsigned char inputChars[] = "abcdefghijklmnopqrstuvwxyz""ABCDEFGHIJKLMNOPQRSTUVWXYZ""0123456789";

    // how many chars we have in the inputChar[]
    int alphabetSize = 62;

    unsigned char* currentPass = malloc((passLen + 1) * sizeof(unsigned char));

    // will iterate the length of 62 and give each index every possible inputChar[i]
    for (int i = 0; i < alphabetSize; i++) {
        
        output[index] = inputChars[i];

        // printf("Index: %d\n", index);

        // this is going to compare the index to the password length as  we will always be in the 
        // iterating through this index when at a password combination that needs to be counted

        // starting at aaa we will iterate through all the inputChars to end at aa9
        // then our for loop will end and we will have the second for loop start to iterate 
        // shifting us to aba where again the last index will go through all of the input chars

        // basically recursivly calling the bruteForce function whenever we are in index 0 or 1 if the length is 3
        // the only time index 0 will shift is when we exhaust all combinations of index 1 and 2
        if (index == passLen - 1) {

            // This will fill the unsigned char array with what we have in the output 
            // while then putting a null value at the end to handle any memory issues 
            for (int j = 0; j < passLen; j++) {
                currentPass[j] = output[j];
            }
            
            // need to null out the last value for clarity
            currentPass[passLen] = '\0';

            // prints the current password guess
            output[passLen] = '\0';
            // printf("Password: %s\n", output);

            // creating a buffer array that will store the output of the hash function
            unsigned char buf[16];
            // declaring our context struct that is going to be used for the MD2 calls 
            MD2 context;

            // calling of the MD2 hash function
            initialize(&context);
            update(&context, currentPass, strlen(currentPass));
            final(&context, buf);

            // more variables for the below conversion so we can have our hash output in non byte terms
            unsigned int i;
            unsigned char hash[16];

            // when you get the hash back from the function it needs to be converted to a char
            // so that we can then compare it to a hypothetical input hash
            for (i = 0; i < 16; i++){
                
                if(i == 0) {
                    // printf("%02x", buf[i]);
                    sprintf(hash, "%02x", buf[i]);
                }
                
                else {

                    // printf("%02x", buf[i]);
                    sprintf(hash + strlen(hash), "%02x", buf[i]);
                }

            }

            // printf("Hash: %s\n", hash);

        }                                          
        else {

            // recursive call of the function so we can iterate the index
            bruteForce(output, index + 1, passLen);

        }
    }
}

int main(int argc, char *argv[]) {

    if(argc != 2) {
        printf("Usage: ./program passLen");
        exit(0);
    }

    int passLen = strtol(argv[1], NULL, 10);
    unsigned char* output = malloc(passLen * sizeof(unsigned char));

    struct timeval start, end;
    
    gettimeofday(&start, NULL);

    // will call the method and start the recursive brtue force
    bruteForce(output, 0, passLen);

    gettimeofday(&end, NULL);

    float time_difference = (end.tv_sec - start.tv_sec) * 1e6;
    time_difference = (time_difference + (end.tv_usec - start.tv_usec)) * 1e-6;

    printf("The Brute Force of password length %d took %fs CPU time.\n", passLen, time_difference);

    return 0;
}