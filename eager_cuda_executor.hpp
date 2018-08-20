#pragma once

#include <stdexcept>
#include <iostream>


template<class F>
__global__
void kernel(F f)
{
  f();
}


class eager_cuda_executor
{
  public:
    eager_cuda_executor()
      : s_(make_cuda_stream())
    {}

    eager_cuda_executor(const eager_cuda_executor&)
      : eager_cuda_executor()
    {}

    ~eager_cuda_executor()
    {
      destroy_cuda_stream(s_);
    }
    
    template<class Function>
    void execute(Function&& f) const
    {
      // decay copy the function
      auto g = std::forward<Function>(f);

      // launch the kernel
      kernel<<<1,1,0,s_>>>(g);
    }

    bool operator==(const eager_cuda_executor&) const { return true; }
    bool operator!=(const eager_cuda_executor& other) const { return !(*this == other); }

  private:
    static cudaStream_t make_cuda_stream()
    {
      cudaStream_t result{};
      if(auto error = cudaStreamCreate(&result))
      {
        throw std::runtime_error("eager_cuda_executor::make_cuda_stream(): CUDA error after cudaStreamCreate(): " + std::string(cudaGetErrorString(error)));
      }

      return result;
    }

    static void destroy_cuda_stream(cudaStream_t s)
    {
      if(auto error = cudaStreamDestroy(s))
      {
        std::cerr << "eager_cuda_executor::destroy_cuda_stream(): CUDA error after cudaStreamDestroy(): " << cudaGetErrorString(error) << std::endl;
        std::terminate();
      }
    }

    cudaStream_t s_;
};

