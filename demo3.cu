// $ nvcc -std=c++14 --expt-extended-lambda demo.cu
#include "simpler_cuda_executor.hpp"
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


// this thing is like like a promise
struct sink_receiver
{
  template<class... Args>
  __host__ __device__
  void set_value(Args&&...) const {}
};


int main()
{
  // test lazy void -> void execution
  {
    simpler_cuda_executor ex;
  
    auto ex_a = ex.make_value_task(none_sender(), [] __host__ __device__ ()
    {
      printf("Hello, world from lazy task A!\n");
    });

    auto ex_b = ex.make_value_task(ex_a, [] __host__ __device__ ()
    {
      printf("Hello, world from lazy task B chained after task A!\n");
    });
    
    // submit lazy task b into a no-op function
    ex_b.submit(sink_receiver());
  
    // wait for lazy_task_b
    cudaEventSynchronize(ex_b.event());
  }

  std::cout << "OK" << std::endl;

  return 0;
}

