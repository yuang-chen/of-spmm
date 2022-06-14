/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_COMMON_STEADY_VECTOR_H_
#define ONEFLOW_CORE_COMMON_STEADY_VECTOR_H_

#include <vector>
#include <array>
#include <mutex>
#include <cmath>
#include <glog/logging.h>

namespace oneflow {

template<typename T, int N = 20>
class SteadyVector {
 public:
  SteadyVector() : size_(0) {}
  ~SteadyVector() = default;

  using value_type = const T;
  using size_type = size_t;

  // thread safe.
  size_t size() const { return size_; }

  // thread safe.
  const T& at(size_t index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, size_);
    int gran = GetGranularity(index);
    int start = (1 << gran) - 1;
    return granularity2vector_[gran].data()[index - start];
  }

  // thread safe.
  const T& operator[](size_t index) const {
    int gran = GetGranularity(index);
    int start = (1 << gran) - 1;
    return granularity2vector_[gran].data()[index - start];
  }

  void push_back(const T& elem) {
    std::unique_lock<std::mutex> lock(mutex_);
    int granularity = GetGranularity(size_);
    if (size_ + 1 == (1 << granularity)) {
      CHECK_LT(granularity, N);
      granularity2vector_[granularity].reserve(1 << granularity);
    }
    auto* vec = &granularity2vector_[granularity];
    vec->push_back(elem);
    ++size_;
  }

 private:
#ifdef __GNUC__
#define LOG2(x) ((unsigned)(8 * sizeof(unsigned long long) - __builtin_clzll((x)) - 1))
#else
#define LOG2(x) std::log2(x)
#endif

  static int GetGranularity(size_t index) { return LOG2(index + 1); }

#undef LOG2

  std::atomic<size_t> size_;
  std::mutex mutex_;
  std::array<std::vector<T>, N> granularity2vector_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_STEADY_VECTOR_H_
