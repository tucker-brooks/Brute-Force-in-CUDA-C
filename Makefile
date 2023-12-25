all: bruteForceCPU bruteForceGPU
bruteforceCPU: bruteForceCPU.c
	gcc -o bruteForceCPU bruteForceCPU.c 
bruteForceGPU: bruteForceGPU.c
	nvcc -std=c++11 -o bruteForceGPU bruteForceGPU.cu
clean:
	rm bruteForceCPU bruteForceGPU