#pragma once

#include "eager_cuda_executor.hpp"
#include "lazy_executor_adaptor.hpp"

using lazy_cuda_executor = lazy_executor_adaptor<eager_cuda_executor>;

