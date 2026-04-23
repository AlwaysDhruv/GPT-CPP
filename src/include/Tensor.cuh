#ifndef ADD_CUH
#define ADD_CUH

#include <vector>
#include <cuda_runtime.h>

namespace GPU
{
    __global__ void add_kernel(float *A, float *B, float *C, int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
            C[i] = A[i] + B[i];
    }

    inline std::vector<float> add(const std::vector<float>& A,
                                  const std::vector<float>& B)
    {
        int N = A.size();

        float *d_A, *d_B, *d_C;

        cudaMalloc(&d_A, N*sizeof(float));
        cudaMalloc(&d_B, N*sizeof(float));
        cudaMalloc(&d_C, N*sizeof(float));

        cudaMemcpy(d_A, A.data(), N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), N*sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

        std::vector<float> C(N);
        cudaMemcpy(C.data(), d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return C;
    }
}

#endif