#pragma once

#include "eager_cuda_executor.hpp"

#define __REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr


namespace detail
{


template<class F>
struct function_receiver
{
  template<class... Args>
  __host__ __device__
  void set_value(Args&&... args)
  {
    f_(std::forward<Args>(args)...);
  }

  F f_;
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


} // end detail


class simpler_cuda_executor
{
  public:
    simpler_cuda_executor()
      : simpler_cuda_executor(eager_cuda_executor())
    {}

    simpler_cuda_executor(simpler_cuda_executor&& other)
      : event_{}
    {
      std::swap(event_, other.event_);
    }

    ~simpler_cuda_executor()
    {
      destroy_cuda_event(event_);
    }

    // XXX where is NoneReceiver meant to be executed?
    // XXX notice how the implementation of .submit() is essentially identical to make_value_task()'s ctor
    template<class NoneReceiver>
    void submit(NoneReceiver nr)
    {
      // make executor's stream wait on this's event
      if(auto error = cudaStreamWaitEvent(executor_.stream(), event_, 0))
      {
        throw std::runtime_error("lazyish_cuda_executor::value_task::submit(): CUDA error after cudaStreamWaitEvent(): " + std::string(cudaGetErrorString(error)));
      }

      // launch nr on executor
      executor_.execute([=] __host__ __device__ () mutable
      {
        nr.set_value();
      });

      // record a new event on executor_'s stream
      if(auto error = cudaEventRecord(event_, executor_.stream()))
      {
        throw std::runtime_error("lazyish_cuda_executor::value_task::submit(): CUDA error after cudaEventRecord(): " + std::string(cudaGetErrorString(error)));
      }
    }

    template<class NoneSender, class F>
    simpler_cuda_executor
      make_value_task(NoneSender&& ns, F&& f) const
    {
      return {std::forward<NoneSender>(ns), std::forward<F>(f), executor_};
    }

    cudaEvent_t event() const
    {
      return event_;
    }

  private:
    explicit simpler_cuda_executor(const eager_cuda_executor& executor)
      : executor_(executor),
        event_(make_cuda_event())
    {}

    template<class F>
    simpler_cuda_executor(const simpler_cuda_executor& predecessor, F&& f, const eager_cuda_executor& executor)
      : simpler_cuda_executor(executor)
    {
      // make executor's stream wait on predecessor's event
      if(auto error = cudaStreamWaitEvent(executor_.stream(), predecessor.event_, 0))
      {
        throw std::runtime_error("simpler_cuda_executor ctor: CUDA error after cudaStreamWaitEvent(): " + std::string(cudaGetErrorString(error)));
      }

      // launch f on executor
      executor_.execute(std::forward<F>(f));

      // record a new event on executor_'s stream
      if(auto error = cudaEventRecord(event_, executor_.stream()))
      {
        throw std::runtime_error("simpler_cuda_executor ctor: CUDA error after cudaEventRecord(): " + std::string(cudaGetErrorString(error)));
      }
    }

    template<class S, class F,
             __REQUIRES(
               !std::is_same<
                 simpler_cuda_executor,
                 std::decay_t<S>
               >::value
             )>
    simpler_cuda_executor(S&& sender, F&& f, const eager_cuda_executor& executor)
      : simpler_cuda_executor()
    {
      // turn f into a receiver
      detail::function_receiver<std::decay_t<F>> f_receiver{std::forward<F>(f)};

      // create a function object that submits f_receiver to sender
      detail::submit_receiver<std::decay_t<S>, decltype(f_receiver)> execute_me{std::forward<S>(sender), std::move(f_receiver)};
      
      // execute on the executor
      executor_.execute(std::move(execute_me));

      // record a new event on executor_'s stream
      if(auto error = cudaEventRecord(event_, executor_.stream()))
      {
        throw std::runtime_error("simpler_cuda_executor::value_task ctor: CUDA error after cudaEventRecord(): " + std::string(cudaGetErrorString(error)));
      }
    }

    static cudaEvent_t make_cuda_event()
    {
      cudaEvent_t result{};
      if(auto error = cudaEventCreateWithFlags(&result, cudaEventDisableTiming))
      {
        throw std::runtime_error("simpler_cuda_executor::value_task::make_cuda_event(): CUDA error after cudaEventCreate(): " + std::string(cudaGetErrorString(error)));
      }

      return result;
    }

    static void destroy_cuda_event(cudaEvent_t e)
    {
      if(auto error = cudaEventDestroy(e))
      {
        std::cerr << "simpler_cuda_executor::value_task::destroy_cuda_event(): CUDA error after cudaEventDestroy(): " + std::string(cudaGetErrorString(error));
        std::terminate();
      }
    }

    cudaEvent_t event_;
    eager_cuda_executor executor_;
};

