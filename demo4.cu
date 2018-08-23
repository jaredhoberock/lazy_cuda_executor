// $ nvcc -std=c++14 --expt-extended-lambda demo.cu
#include "eager_cuda_executor.hpp"
#include "value_propagating_cuda_executor.hpp"
#include <iostream>


// this thing is like a ready future of T
template<class T>
struct just
{
  template<class SingleReceiver>
  __host__ __device__
  void submit(SingleReceiver sr)
  {
    sr.set_value(value_);
  }

  T get() const
  {
    return value_;
  }

  T value_;
};


// this thing is like a promise
struct printf_receiver
{
  __host__ __device__
  void set_value(int val) const
  {
    printf("Received %d in printf_receiver\n", val);
  }
};


int main()
{
  // test int -> int execution
  {
    value_propagating_cuda_executor ex;
  
    auto task_a = ex.make_value_task(just<int>{13}, [] __host__ __device__ (int val)
    {
      printf("Received %d in task a\n", val);
      return val + 1;
    });

    auto task_b = ex.make_value_task(task_a, [] __host__ __device__ (int val)
    {
      printf("Received %d in task b\n", val);
      return val + 1;
    });
    
    // submit task b into a print function
    task_b.submit(printf_receiver());
  
    // wait for task_b
    cudaEventSynchronize(task_b.event());
  }

  std::cout << "OK" << std::endl;

  return 0;
}

