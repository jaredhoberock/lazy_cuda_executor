# lazy_cuda_executor
Toy implementation of a lazy CUDA executor

## demo

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
    
    
    // this thing is like like a promise
    struct sink_receiver
    {
      template<class... Args>
      __host__ __device__
      void set_value(Args&&...) const {}
    };
    
    
    int main()
    {
      eager_cuda_executor eager_ex;
    
      eager_ex.execute([] __host__ __device__ ()
      {
        printf("Hello, world from eager task!\n");
      });
    
      lazy_cuda_executor lazy_ex;
    
      lazy_ex.make_value_task(none_sender(), [] __host__ __device__ ()
      {
        printf("Hello, world from lazy task!\n");
      }).submit(sink_receiver());
    
      // wait for everything
      cudaDeviceSynchronize();
    
      std::cout << "OK" << std::endl;
    
      return 0;
    }

Program output:

    $ nvcc -std=c++14 --expt-extended-lambda demo.cu
    $ ./a.out
    Hello, world from eager task!
    Hello, world from lazy task!
    OK
