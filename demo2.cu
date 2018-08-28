// $ nvcc -std=c++14 --expt-extended-lambda demo.cu
#include "eager_cuda_executor.hpp"
#include "lazyish_cuda_executor.hpp"
#include <iostream>


// this thing is like a future<void>
struct none_sender
{
  template<class NoneReceiver>
  void submit(NoneReceiver nr)
  {
    nr.set_value();
  }
};


// this thing is like like a promise
struct sink_receiver
{
  template<class... Args>
  void set_value(Args&&...) const {}
};


int main()
{
  // test lazy void -> void execution
  {
    lazyish_cuda_executor lazyish;

    auto lazy_task_a = lazyish.make_value_task(none_sender(), [] __host__ __device__ ()
    {
      printf("Hello, world from lazy task A!\n");
    });

    auto lazy_task_b = lazyish.make_value_task(lazy_task_a, [] __host__ __device__ ()
    {
      printf("Hello, world from lazy task B chained after task A!\n");
    });

    // submit lazy task b into a no-op function
    detail::wait(lazy_task_b);
  }

  std::cout << "OK" << std::endl;

  return 0;
}
