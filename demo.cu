// $ nvcc -std=c++14 --expt-extended-lambda demo.cu
#include "eager_cuda_executor.hpp"
#include "lazy_cuda_executor.hpp"
#include <iostream>


// this thing is like a future<void>
struct none_sender
{
  template<class NoneReceiver>
  __host__ __device__
  void submit(NoneReceiver nr)
  {
    nr.set_value();
  }
};


// this thing is like a future<T>
template<class T>
struct just
{
  template<class SingleReceiver>
  __host__ __device__
  void submit(SingleReceiver sr)
  {
    sr.set_value(value_);
  }

  T value_;
};


// this thing is like like a promise
struct sink_receiver
{
  template<class... Args>
  __host__ __device__
  void set_value(Args&&...) const {}
};


int main()
{
  // test eager one-way execution
  {
    eager_cuda_executor eager;

    eager.execute([] __host__ __device__ ()
    {
      printf("Hello, world from eager task!\n");
    });

    // wait for everything
    cudaDeviceSynchronize();
  }

  // test lazy void -> void execution
  {
    lazy_cuda_executor lazy;
  
    lazy.make_value_task(none_sender(), [] __host__ __device__ ()
    {
      printf("Hello, world from lazy task!\n");
    }).submit(sink_receiver());
  
    // wait for everything
    cudaDeviceSynchronize();
  }

  // test lazy int -> void execution
  {
    lazy_cuda_executor lazy;

    lazy.make_value_task(just<int>{13}, [] __host__ __device__ (int value)
    {
      printf("Received %d in lazy task.\n", (int)value);
    }).submit(sink_receiver());

    // wait for everything
    cudaDeviceSynchronize();
  }

  std::cout << "OK" << std::endl;

  return 0;
}

