#pragma once

#include "eager_cuda_executor.hpp"

#define __REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr


namespace detail
{


template<class F, class R>
struct function_receiver
{
  template<class... Args>
  __host__ __device__
  void set_value(Args&&... args)
  {
    *result_ptr_ = f_(std::forward<Args>(args)...);
  }

  F f_;
  R* result_ptr_;
};


template<class Sender, class Receiver>
struct submit_receiver
{
  __host__ __device__
  void operator()()
  {
    sender_.submit(receiver_);
  }

  Sender sender_;
  Receiver receiver_;
};


template<class Sender>
using sender_result_t = decltype(std::declval<Sender>().get());


template<class F, class Arg, class Result>
struct invoke_and_placement_new_result
{
  __host__ __device__
  void operator()()
  {
    new(result_ptr) Result(f(*argument_ptr));
  }

  F f;
  Arg* argument_ptr;
  Result* result_ptr;
};


} // end detail

class cuda_executor_with_values
{
  public:
    template<class T>
    class value_task
    {
      public:
        value_task(value_task&& other)
          : event_{},
            result_ptr_{}
        {
          std::swap(event_, other.event_);
          std::swap(result_ptr_, other.result_ptr_);
        }

        ~value_task()
        {
          destroy_cuda_event(event_);
        }

        T get() const
        {
          return *result_ptr_;
        }

        template<class Arg, class F>
        value_task(const value_task<Arg>& sender, F&& f, const eager_cuda_executor& executor)
          : event_(make_cuda_event()),
            executor_(executor),
            result_ptr_(allocate_result())
        {
          // make executor's stream wait on sender's event
          if(auto error = cudaStreamWaitEvent(executor_.stream(), sender.event_, 0))
          {
            throw std::runtime_error("cuda_executor_with_values::value_task ctor: CUDA error after cudaStreamWaitEvent(): " + std::string(cudaGetErrorString(error)));
          }

          // decay copy f
          Arg* argument_ptr = sender.result_ptr_;
          auto f_copy = std::forward<F>(f);

          // launch f_copy on executor_
          detail::invoke_and_placement_new_result<std::decay_t<F>, Arg, T> execute_me{std::forward<F>(f), argument_ptr, result_ptr_};
          executor_.execute(execute_me);

          // record a new event on executor_'s stream
          if(auto error = cudaEventRecord(event_, executor_.stream()))
          {
            throw std::runtime_error("cuda_executor_with_values::value_task ctor: CUDA error after cudaEventRecord(): " + std::string(cudaGetErrorString(error)));
          }

          // XXX argument_ptr can be deallocated once event_ completes
          //     for now, it leaks
        }

        template<class S, class F,
                 __REQUIRES(
                   !std::is_same<
                     value_task,
                     std::decay_t<S>
                   >::value
                 )>
        value_task(S&& sender, F&& f, const eager_cuda_executor& executor)
          : event_(make_cuda_event()),
            executor_(executor),
            result_ptr_(allocate_result())
        {
          // turn f into a receiver
          detail::function_receiver<std::decay_t<F>, T> f_receiver{std::forward<F>(f), result_ptr_};

          // create a function object that submits f_receiver to sender
          detail::submit_receiver<std::decay_t<S>, decltype(f_receiver)> execute_me{std::forward<S>(sender), std::move(f_receiver)};
          
          // execute on the executor
          executor_.execute(std::move(execute_me));

          // record a new event on executor_'s stream
          if(auto error = cudaEventRecord(event_, executor_.stream()))
          {
            throw std::runtime_error("cuda_executor_with_values::value_task ctor: CUDA error after cudaEventRecord(): " + std::string(cudaGetErrorString(error)));
          }
        }

        // XXX where is NoneReceiver meant to be executed?
        // XXX notice how the implementation of .submit() is essentially identical to make_value_task()'s ctor
        template<class NoneReceiver>
        void submit(NoneReceiver nr)
        {
          // make executor's stream wait on this's event
          if(auto error = cudaStreamWaitEvent(executor_.stream(), event_, 0))
          {
            throw std::runtime_error("cuda_executor_with_values::value_task::submit(): CUDA error after cudaStreamWaitEvent(): " + std::string(cudaGetErrorString(error)));
          }

          // launch nr on executor
          T* result_ptr = result_ptr_;
          executor_.execute([=] __host__ __device__ () mutable
          {
            nr.set_value(*result_ptr);
          });

          // record a new event on executor_'s stream
          if(auto error = cudaEventRecord(event_, executor_.stream()))
          {
            throw std::runtime_error("cuda_executor_with_values::value_task::submit(): CUDA error after cudaEventRecord(): " + std::string(cudaGetErrorString(error)));
          }

          // XXX result_ can be deallocated once event_ completes
          //     for now, it leaks
        }

        cudaEvent_t event() const
        {
          return event_;
        }
       
      private:
        static cudaEvent_t make_cuda_event()
        {
          cudaEvent_t result{};
          if(auto error = cudaEventCreateWithFlags(&result, cudaEventDisableTiming))
          {
            throw std::runtime_error("cuda_executor_with_values::value_task::make_cuda_event(): CUDA error after cudaEventCreate(): " + std::string(cudaGetErrorString(error)));
          }

          return result;
        }

        static void destroy_cuda_event(cudaEvent_t e)
        {
          if(auto error = cudaEventDestroy(e))
          {
            std::cerr << "cuda_executor_with_values::value_task::destroy_cuda_event(): CUDA error after cudaEventDestroy(): " + std::string(cudaGetErrorString(error));
            std::terminate();
          }
        }

        static T* allocate_result()
        {
          T* result_ptr{};
          if(auto error = cudaMallocManaged(&result_ptr, sizeof(T)))
          {
            throw std::runtime_error("cuda_executor_with_values::value_task::allocate_value(): CUDA error after cudaMallocManaged(): " + std::string(cudaGetErrorString(error)));
          }

          return result_ptr;
        }

        cudaEvent_t event_;
        eager_cuda_executor executor_;
        T* result_ptr_;
    };

    template<class SingleSender, class F>
    value_task<
      std::result_of_t<
        F(detail::sender_result_t<std::decay_t<SingleSender>>)
      >
    >
      make_value_task(SingleSender&& ns, F&& f) const
    {
      return {std::forward<SingleSender>(ns), std::forward<F>(f), executor_};
    }

  private:
    eager_cuda_executor executor_;
};


