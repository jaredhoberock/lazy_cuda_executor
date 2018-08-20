#pragma once

#include "lazy_cuda_executor.hpp"
#include <type_traits>

#define __REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr


template<class F, class SingleReceiver>
struct __invoke_and_set_value
{
  template<class... Args,
           __REQUIRES(
             !std::is_void<
               std::result_of_t<F(Args&&...)>
             >::value
           )>
  __host__ __device__
  void set_value(Args&&... args)
  {
    receiver_.set_value(f_(std::forward<Args>(args)...));
  }

  template<class... Args,
           __REQUIRES(
             std::is_void<
               std::result_of_t<F(Args&&...)>
             >::value
           )>
  __host__ __device__
  void set_value(Args&&... args)
  {
    f_(std::forward<Args>(args)...);
    receiver_.set_value();
  }

  F f_;
  SingleReceiver receiver_;
};


template<class Sender, class Receiver>
struct __submit_receiver
{
  __host__ __device__
  void operator()()
  {
    sender_.submit(receiver_);
  }

  Sender sender_;
  Receiver receiver_;
};


class lazy_cuda_executor
{
  public:
    template<class Sender, class Function>
    class value_task
    {
      public:
        template<class S, class F>
        value_task(S&& sender, F&& f, const eager_cuda_executor& executor)
          : sender_(std::forward<S>(sender)),
            f_(std::forward<F>(f)),
            executor_(executor)
        {}

        template<class SingleReceiver>
        void submit(SingleReceiver sr)
        {
          // wrap up f_ to send its result to sr
          __invoke_and_set_value<Function, SingleReceiver> wrapped_receiver{f_, sr};

          // XXX workaround nvbug 2338736
          //executor_.execute([=] __host__ __device__
          //{
          //  // submit the wrapped receiver to the sender
          //  sender_.submit(wrapped_receiver);
          //});

          __submit_receiver<Sender,decltype(wrapped_receiver)> workaround{sender_, wrapped_receiver};
          executor_.execute(workaround);
        }

      private:
        Sender sender_;
        Function f_;
        eager_cuda_executor executor_;
    };

    template<class S, class F>
    value_task<std::decay_t<S>, std::decay_t<F>>
      make_value_task(S&& sender, F&& f) const
    {
      return {std::forward<S>(sender), std::forward<F>(f), executor_};
    }

  private:
    eager_cuda_executor executor_;
};

