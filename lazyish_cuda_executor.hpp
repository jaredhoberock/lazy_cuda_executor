#pragma once

#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>

template<class T>
using decay_t = typename std::decay<T>::type;

template<class T>
using result_of_t = typename std::result_of<T>::type;

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

template<class F>
struct function_execute_receiver
{
  template<class... Args>
  void set_value(Args&&... args)
  {
    // execute on the executor
    executor_.execute(std::move(f_));

    // record a new event on executor_'s stream
    if(auto error = cudaEventRecord(event_, executor_.stream()))
    {
      throw std::runtime_error("function_execute_receiver::set_value: CUDA error after cudaEventRecord(): " + std::string(cudaGetErrorString(error)));
    }
  }

  F f_;
  cudaEvent_t event_;
  eager_cuda_executor executor_;
};

template<class R>
struct set_value_caller_fn
{
  static void call(cudaStream_t stream, cudaError_t err, set_value_caller_fn<R>* caller) {
    auto owner = std::unique_ptr<set_value_caller_fn<R>>(caller);
    std::thread{&set_value_caller_fn<R>::call_set_value, std::move(owner)}.detach();
  }

  void call_set_value() {
    nr.set_value();
  }

  R nr;
};

struct waiter{
  std::mutex* lock_;
  std::condition_variable* wake_;
  bool* done_;
  template<class... Args>
  void set_value(Args&&...) {
    std::unique_lock<std::mutex> guard{*lock_};
    *done_ = true;
    wake_->notify_one();
  }
};

template<class Sender>
void wait(Sender& s) {
  std::mutex lock;
  std::condition_variable wake;
  bool done = false;
  s.submit(waiter{&lock, &wake, &done});
  std::unique_lock<std::mutex> guard{lock};
  wake.wait(guard, [&](){return done;});
}

} // end detail

class lazyish_cuda_executor
{
  public:
    class value_task
    {
      public:
        value_task(value_task&& other)
          : event_{}
        {
          std::swap(event_, other.event_);
        }

        ~value_task()
        {
          destroy_cuda_event(event_);
        }

        template<class F>
        value_task(const value_task& sender, F&& f, const eager_cuda_executor& executor)
          : event_(make_cuda_event()),
            executor_(executor)
        {
          // make executor's stream wait on sender's event
          if(auto error = cudaStreamWaitEvent(executor_.stream(), sender.event_, 0))
          {
            throw std::runtime_error("lazyish_cuda_executor::value_task ctor: CUDA error after cudaStreamWaitEvent(): " + std::string(cudaGetErrorString(error)));
          }

          // launch f on executor_
          executor_.execute(std::forward<F>(f));

          // record a new event on executor_'s stream
          if(auto error = cudaEventRecord(event_, executor_.stream()))
          {
            throw std::runtime_error("lazyish_cuda_executor::value_task ctor: CUDA error after cudaEventRecord(): " + std::string(cudaGetErrorString(error)));
          }
        }

        template<class S, class F,
                 __REQUIRES(
                   !std::is_same<
                     value_task,
                     decay_t<S>
                   >::value
                 )>
        value_task(S&& sender, F&& f, const eager_cuda_executor& executor)
          : event_(make_cuda_event()),
            executor_(executor)
        {
          detail::function_execute_receiver<decay_t<F>> f_receiver{std::forward<F>(f), event_, executor_};

          sender.submit(f_receiver);
        }

        template<class NoneReceiver>
        void submit(NoneReceiver nr)
        {
#if 1
// case 1. non-blocking - uses cudaStreamAddCallback + std::thread
          if(auto error = cudaStreamWaitEvent(executor_.stream(), event_, 0))
          {
            throw std::runtime_error("lazyish_cuda_executor::value_task::submit: CUDA error after cudaStreamWaitEvent(): " + std::string(cudaGetErrorString(error)));
          }

          std::unique_ptr<detail::set_value_caller_fn<NoneReceiver>> fn{new detail::set_value_caller_fn<NoneReceiver>{std::move(nr)}};

          if(auto error = cudaStreamAddCallback(executor_.stream(), (cudaStreamCallback_t)detail::set_value_caller_fn<NoneReceiver>::call, fn.release(), 0))
          {
            throw std::runtime_error("lazyish_cuda_executor::value_task::submit: CUDA error after cudaStreamAddCallback(): " + std::string(cudaGetErrorString(error)));
          }
#else
// case 3. blocking - uses cudaEventSynchronize and the context that called submit
          if(auto error = cudaEventSynchronize(event_))
          {
            throw std::runtime_error("lazyish_cuda_executor::value_task::submit: CUDA error after cudaEventSynchronize(): " + std::string(cudaGetErrorString(error)));
          }

          nr.set_value();
#endif
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
            throw std::runtime_error("lazyish_cuda_executor::value_task::make_cuda_event(): CUDA error after cudaEventCreate(): " + std::string(cudaGetErrorString(error)));
          }

          return result;
        }

        static void destroy_cuda_event(cudaEvent_t e)
        {
          if(auto error = cudaEventDestroy(e))
          {
            std::cerr << "lazyish_cuda_executor::value_task::destroy_cuda_event(): CUDA error after cudaEventDestroy(): " + std::string(cudaGetErrorString(error));
            std::terminate();
          }
        }

        cudaEvent_t event_;
        eager_cuda_executor executor_;
    };

    template<class NoneSender, class F>
    value_task
      make_value_task(NoneSender&& ns, F&& f) const
    {
      return {std::forward<NoneSender>(ns), std::forward<F>(f), executor_};
    }

  private:
    eager_cuda_executor executor_;
};
