# Brute-Force-in-CUDA-C
Password Brute Force Algorithm and MD2 Hashing in Cuda/C

Executed a CUDA-based one-to-one thread mapping approach for password combinations to threads, leveraging GPU parallelization for optimal performance in brute force problems.

Translated a recursive CPU-based brute force algorithm with a MD2 hash function to the GPU, achieving full parallelization by comprehensively mapping one password combination to singular threads and surpassing the basic implementation of outermost loop parallelization.

Demonstrated the profound impact of growing GPU computational power by significantly reducing time complexity in password brute forcing compared to the recursive CPU-based approach, achieving a 3.71x speedup with outermost loop parallelization and an extraordinary 18,570x speedup with one-to-one mapping.
