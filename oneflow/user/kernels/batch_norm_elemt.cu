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
#include <limits>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/cuda/cuda_device.h"
#include "oneflow/user/kernels/batch_norm_kernel_utils.h"

// NOTE(Liang Depeng):
// The implementation of batch_norm_gather_stats_with_counts kernel is modified from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Normalization.cuh

namespace oneflow {

namespace {

template<typename T, typename IDX_TYPE>
__global__ void batch_norm_transform_input_kernel(const IDX_TYPE batch_size,
                                                  const IDX_TYPE channel_size,
                                                  const IDX_TYPE spatial_size, const T* input_ptr,
                                                  const T* mean_ptr, const T* invstd_ptr,
                                                  const T* weight_ptr, const T* bias_ptr,
                                                  T* output_ptr) {
  IDX_TYPE channel = blockIdx.x;
  IDX_TYPE channel_offset = channel * spatial_size;
  IDX_TYPE batch_step = channel_size * spatial_size;

  if (channel >= channel_size) { return; }

  T gamma = static_cast<T>(weight_ptr[channel]);
  T beta = static_cast<T>(bias_ptr[channel]);
  T mean = static_cast<T>(mean_ptr[channel]);
  T invstd = invstd_ptr[channel];

  IDX_TYPE bstep = blockDim.y * gridDim.y;
  for (IDX_TYPE batch = threadIdx.y + blockIdx.y * blockDim.y; batch < batch_size; batch += bstep) {
    IDX_TYPE offset = batch * batch_step + channel_offset;
    for (IDX_TYPE feature = threadIdx.x; feature < spatial_size; feature += blockDim.x) {
      output_ptr[offset + feature] =
          static_cast<T>(gamma * (input_ptr[offset + feature] - mean) * invstd + beta);
    }
  }
}

template<typename T>
struct BatchNormElemtFunctor final {
  void operator()(ep::Stream* stream, const int64_t batch_size, const int64_t channel_size,
                  const int64_t spatial_size, const T* input_ptr, const T* mean_ptr,
                  const T* invstd_ptr, const T* weight_ptr, const T* bias_ptr, T* output_ptr) {
    // The input_transform kernel is pointwise, but we need to balance reading parameters
    // (save_var/mean, weight/bias) - which we only do once and have a for loop afterwards - with
    // having many threads and blocks and good occupancy. Quiet likely, we could go with even more
    // blocks than 1024. The various planes are independent, so we use blocks for them.
    int tf = std::max<int>(getNumThreads(spatial_size / 4),
                           std::min<int>(getNumThreads(spatial_size), 64));
    int tb = std::max<int>(64 / tf, 1);
    dim3 blocks_trans(channel_size, std::max<int>(1, std::min<int>((256 * 1024) / channel_size,
                                                                   (batch_size + tb - 1) / tb)));
    blocks_trans.y = std::min(blocks_trans.y, MAX_GRID_SIZE);
    dim3 threads_trans(tf, tb);

    if (batch_size * channel_size * spatial_size < std::numeric_limits<int32_t>::max()) {
      batch_norm_transform_input_kernel<T, int32_t>
          <<<blocks_trans, threads_trans, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              static_cast<int32_t>(batch_size), static_cast<int32_t>(channel_size),
              static_cast<int32_t>(spatial_size), input_ptr, mean_ptr, invstd_ptr, weight_ptr,
              bias_ptr, output_ptr);
    } else {
      batch_norm_transform_input_kernel<T, int64_t>
          <<<blocks_trans, threads_trans, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              batch_size, channel_size, spatial_size, input_ptr, mean_ptr, invstd_ptr, weight_ptr,
              bias_ptr, output_ptr);
    }
  }
};

}  // namespace

template<typename T>
class GpuBatchNormElemtKernel final : public user_op::OpKernel {
 public:
  GpuBatchNormElemtKernel() = default;
  ~GpuBatchNormElemtKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const user_op::Tensor* invstd = ctx->Tensor4ArgNameAndIndex("invstd", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);

    const T* input_ptr = input->dptr<T>();
    const T* mean_ptr = mean->dptr<T>();
    const T* invstd_ptr = invstd->dptr<T>();
    const T* weight_ptr = weight->dptr<T>();
    const T* bias_ptr = bias->dptr<T>();
    T* output_ptr = output->mut_dptr<T>();
    const int32_t axis = ctx->Attr<int32_t>("axis");

    bool use_channels_last_kernel = axis == 1 ? false : true;
    if (use_channels_last_kernel) {  // NHWC format

    } else {  // NCHW format
      const int64_t batch_size = input->shape_view().At(0);
      const int64_t channel_size = input->shape_view().At(1);
      const int64_t spatial_size = input->shape_view().Count(2);

      BatchNormElemtFunctor<T>()(ctx->stream(), batch_size, channel_size, spatial_size, input_ptr,
                                 mean_ptr, invstd_ptr, weight_ptr, bias_ptr, output_ptr);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BATCH_NORM_ELEMT_KERNEL(dtype)                                            \
  REGISTER_USER_KERNEL("batch_norm_elemt")                                                 \
      .SetCreateFn<GpuBatchNormElemtKernel<dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("mean", 0) == GetDataType<dtype>::value)   \
                       && (user_op::HobDataType("invstd", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("weight", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("bias", 0) == GetDataType<dtype>::value))

REGISTER_BATCH_NORM_ELEMT_KERNEL(float);
REGISTER_BATCH_NORM_ELEMT_KERNEL(double);

}  // namespace oneflow
