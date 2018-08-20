// $ nvcc -std=c++14 --expt-extended-lambda demo.cu
#include "eager_cuda_executor.hpp"
#include <iostream>

int main()
{
  eager_cuda_executor eager_ex;

  eager_ex.execute([] __host__ __device__ ()
  {
    printf("Hello, world from eager task!\n");
  });

  // wait for everything
  cudaDeviceSynchronize();

  std::cout << "OK" << std::endl;

  return 0;
}

