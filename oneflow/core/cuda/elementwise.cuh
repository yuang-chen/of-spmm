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
#ifndef ONEFLOW_CORE_CUDA_ELEMENTWISE_H_
#define ONEFLOW_CORE_CUDA_ELEMENTWISE_H_

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include "oneflow/core/common/stride.h"

namespace oneflow {

namespace cuda {

namespace elementwise {

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;

inline cudaError_t GetNumBlocks(int64_t n, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                   sm_count * tpm / kBlockSize * kNumWaves));
  return cudaSuccess;
}

__device__ __forceinline__ int64_t cuda_compute_offset(int64_t out_offset,
                                                       const StrideParam& in_stride,
                                                       const StrideParam& out_stride) {
  int64_t in_offset = 0;
  int64_t remaining = out_offset;

#pragma unroll
  for (size_t i = 0; i < in_stride.ndim; ++i) {
    const int64_t idx = static_cast<int64_t>(remaining / out_stride.stride[i]);
    remaining -= idx * out_stride.stride[i];
    in_offset += idx * in_stride.stride[i];
  }
  return in_offset;
}

template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template<typename T, int pack_size>
union Pack {
  static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, pack_size> storage;
  T elem[pack_size];
};

template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed {
  __device__ Packed() {
    // do nothing
  }
  union {
    T elem[pack_size];
  };
};

constexpr int kMaxPackBytes = 128 / 8;
constexpr int kMaxPackSize = 8;

constexpr int Min(int a, int b) { return a < b ? a : b; }

template<typename T>
constexpr int PackSize() {
  return Min(kMaxPackBytes / sizeof(T), kMaxPackSize);
}

template<typename T, typename U, typename... Args>
constexpr int PackSize() {
  return Min(PackSize<T>(), PackSize<U, Args...>());
}

template<typename T>
class HasApply2 {
  typedef char one;
  struct two {
    char x[2];
  };

  template<typename C>
  static one test(decltype(&C::Apply2));
  template<typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

template<int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == true && pack_size % 2 == 0,
                                   Packed<R, pack_size>>::type
ApplyPack(const FunctorT& functor, const IN... in[pack_size]) {
  Packed<R, pack_size> ret;
#pragma unroll
  for (int j = 0; j < pack_size; j += 2) { functor.Apply2(ret.elem + j, (in + j)...); }
  return ret;
}

template<int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == false || pack_size % 2 != 0,
                                   Packed<R, pack_size>>::type
ApplyPack(const FunctorT& functor, const IN... in[pack_size]) {
  Packed<R, pack_size> ret;
#pragma unroll
  for (int j = 0; j < pack_size; ++j) { ret.elem[j] = functor((in[j])...); }
  return ret;
}

template<int pack_size, bool tail, typename FactoryT, typename R, typename... IN>
__global__ void __launch_bounds__(kBlockSize)
    ApplyGeneric(FactoryT factory, int64_t n_pack, Packed<R, pack_size>* pack_r,
                 const Packed<IN, pack_size>*... pack_in, int64_t n_tail, R* tail_r,
                 const IN*... tail_in) {
  auto functor = factory();
  const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
  for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x) {
    pack_r[i] = ApplyPack<pack_size, decltype(functor), R, IN...>(functor, (pack_in[i].elem)...);
  }
  if (tail && global_tid < n_tail) { tail_r[global_tid] = functor((tail_in[global_tid])...); }
}

template<typename FactoryT, typename R, typename... IN>
__global__ void __launch_bounds__(kBlockSize)
    ApplyStridedGeneric(FactoryT factory, int64_t count, const StrideParam in_stride,
                        const StrideParam out_stride, R* r, const IN*... in) {
  auto functor = factory();
  const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
  for (int64_t i = global_tid; i < count; i += blockDim.x * gridDim.x) {
    int64_t in_offset = cuda_compute_offset(i, in_stride, out_stride);
    r[i] = functor((in[in_offset])...);
  }
}

template<typename FunctorT>
struct SimpleFactory {
  explicit SimpleFactory(FunctorT functor) : tpl(functor) {}
  __device__ FunctorT operator()() const { return tpl; }

 private:
  FunctorT tpl;
};

template<size_t pack_size>
bool IsAligendForPack() {
  return true;
}

template<size_t pack_size, typename T, typename... Args>
bool IsAligendForPack(const T* ptr, const Args*... others) {
  return reinterpret_cast<uintptr_t>(ptr) % sizeof(Pack<T, pack_size>) == 0
         && IsAligendForPack<pack_size, Args...>(others...);
}

template<size_t pack_size, typename FactoryT, typename R, typename... IN>
cudaError_t LaunchKernel(FactoryT factory, int64_t n, R* r, const IN*... in, cudaStream_t stream) {
  const int64_t n_pack = n / pack_size;
  const int64_t tail_offset = n_pack * pack_size;
  const int64_t n_tail = n - tail_offset;
  int num_blocks;
  {
    cudaError_t err = GetNumBlocks(n_pack, &num_blocks);
    if (err != cudaSuccess) { return err; }
  }
  auto func = n_tail > 0 ? ApplyGeneric<pack_size, true, FactoryT, R, IN...>
                         : ApplyGeneric<pack_size, false, FactoryT, R, IN...>;
  func<<<num_blocks, kBlockSize, 0, stream>>>(
      factory, n_pack, reinterpret_cast<Packed<R, pack_size>*>(r),
      (reinterpret_cast<const Packed<IN, pack_size>*>(in))..., n_tail, r + tail_offset,
      (in + tail_offset)...);
  return cudaPeekAtLastError();
}

template<typename FactoryT, typename R, typename... IN>
cudaError_t LaunchStridedKernel(FactoryT factory, int64_t n, const StrideParam& in_stride,
                                const StrideParam& out_stride, R* r, const IN*... in,
                                cudaStream_t stream) {
  int num_blocks = 1;
  {
    cudaError_t err = GetNumBlocks(n, &num_blocks);
    if (err != cudaSuccess) { return err; }
  }
  auto func = ApplyStridedGeneric<FactoryT, R, IN...>;
  func<<<num_blocks, kBlockSize, 0, stream>>>(factory, n, in_stride, out_stride, r, (in)...);
  return cudaPeekAtLastError();
}

template<typename FactoryT, typename R, typename... IN>
struct GenericLauncher {
  static cudaError_t Launch(FactoryT factory, int64_t n, R* r, const IN*... in,
                            cudaStream_t stream) {
    constexpr int max_pack_size = PackSize<R, IN...>();
    if (IsAligendForPack<max_pack_size, R, IN...>(r, in...)) {
      return LaunchKernel<max_pack_size, FactoryT, R, IN...>(factory, n, r, in..., stream);
    } else {
      return LaunchKernel<1, FactoryT, R, IN...>(factory, n, r, in..., stream);
    }
  }

  static cudaError_t Launch(FactoryT factory, int64_t n, const StrideParam& in_stride,
                            const StrideParam& out_stride, R* r, const IN*... in,
                            cudaStream_t stream) {
    return LaunchStridedKernel<FactoryT, R, IN...>(factory, n, in_stride, out_stride, r, in...,
                                                   stream);
  }
};

template<typename FactoryT, typename R, typename A>
inline cudaError_t UnaryWithFactory(FactoryT factory, int64_t n, R* r, const A* a,
                                    cudaStream_t stream) {
  return GenericLauncher<FactoryT, R, A>::Launch(factory, n, r, a, stream);
}

template<typename FactoryT, typename R, typename A>
inline cudaError_t UnaryWithFactory(FactoryT factory, int64_t n, const StrideParam& in_stride,
                                    const StrideParam& out_stride, R* r, const A* a,
                                    cudaStream_t stream) {
  return GenericLauncher<FactoryT, R, A>::Launch(factory, n, in_stride, out_stride, r, a, stream);
}

template<typename FunctorT, typename R, typename A>
inline cudaError_t Unary(FunctorT functor, int64_t n, R* r, const A* a, cudaStream_t stream) {
  return UnaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a, stream);
}

template<typename FunctorT, typename R, typename A>
inline cudaError_t Unary(FunctorT functor, int64_t n, const StrideParam& in_stride,
                         const StrideParam& out_stride, R* r, const A* a, cudaStream_t stream) {
  return UnaryWithFactory(SimpleFactory<FunctorT>(functor), n, in_stride, out_stride, r, a, stream);
}

template<typename FactoryT, typename R, typename A, typename B>
inline cudaError_t BinaryWithFactory(FactoryT factory, int64_t n, R* r, const A* a, const B* b,
                                     cudaStream_t stream) {
  return GenericLauncher<FactoryT, R, A, B>::Launch(factory, n, r, a, b, stream);
}

template<typename FunctorT, typename R, typename A, typename B>
inline cudaError_t Binary(FunctorT functor, int64_t n, R* r, const A* a, const B* b,
                          cudaStream_t stream) {
  return BinaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a, b, stream);
}

template<typename FactoryT, typename R, typename A, typename B, typename C>
inline cudaError_t TernaryWithFactory(FactoryT factory, int64_t n, R* r, const A* a, const B* b,
                                      const C* c, cudaStream_t stream) {
  return GenericLauncher<FactoryT, R, A, B, C>::Launch(factory, n, r, a, b, c, stream);
}

template<typename FunctorT, typename R, typename A, typename B, typename C>
inline cudaError_t Ternary(FunctorT functor, int64_t n, R* r, const A* a, const B* b, const C* c,
                           cudaStream_t stream) {
  return TernaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a, b, c, stream);
}

}  // namespace elementwise

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_ELEMENTWISE_H_
