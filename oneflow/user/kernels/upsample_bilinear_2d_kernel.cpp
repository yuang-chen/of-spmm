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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/user/kernels/upsample_kernel.h"

namespace oneflow {

namespace {

template<typename T>
static void UpsampleBilinear2DForward(const int64_t elem_cnt, const T* in_dptr,
                                      NdIndexOffsetHelper<int64_t, 4> in_helper,
                                      NdIndexOffsetHelper<int64_t, 4> out_helper,
                                      const int64_t in_height, const int64_t in_width,
                                      const T scale_h, const T scale_w, const bool align_corners,
                                      T* out_dptr) {
  for (int64_t index = 0; index < elem_cnt; ++index) {
    int64_t n, c, h, w;
    out_helper.OffsetToNdIndex(index, n, c, h, w);
    BilinearParam<T> params;
    GetBilinearParam(align_corners, h, w, in_height, in_width, scale_h, scale_w, &params);
    const int64_t top_offset = in_helper.NdIndexToOffset(n, c, params.top_h_index, 0);
    const int64_t bottom_offset = in_helper.NdIndexToOffset(n, c, params.bottom_h_index, 0);
    const T top_left = in_dptr[top_offset + params.left_w_index];
    const T top_right = in_dptr[top_offset + params.right_w_index];
    const T bottom_left = in_dptr[bottom_offset + params.left_w_index];
    const T bottom_right = in_dptr[bottom_offset + params.right_w_index];
    const T top = top_left + (top_right - top_left) * params.w_lerp;
    const T bottom = bottom_left + (bottom_right - bottom_left) * params.w_lerp;
    out_dptr[index] = top + (bottom - top) * params.h_lerp;
  }
}

template<typename T>
static void UpsampleBilinearBackward(const int64_t elem_cnt, const T* dy_dptr,
                                     NdIndexOffsetHelper<int64_t, 4> dy_helper,
                                     NdIndexOffsetHelper<int64_t, 4> dx_helper,
                                     const int64_t dx_height, const int64_t dx_width,
                                     const T scale_h, const T scale_w, const bool align_corners,
                                     T* dx_dptr) {
  for (int64_t index = 0; index < elem_cnt; ++index) {
    int64_t n, c, h, w;
    dy_helper.OffsetToNdIndex(index, n, c, h, w);
    BilinearParam<T> params;
    GetBilinearParam(align_corners, h, w, dx_height, dx_width, scale_h, scale_w, &params);
    const int64_t top_offset = dx_helper.NdIndexToOffset(n, c, params.top_h_index, 0);
    const int64_t bottom_offset = dx_helper.NdIndexToOffset(n, c, params.bottom_h_index, 0);
    const T dy = dy_dptr[index];
    const T dbottom = params.h_lerp * dy;
    T* dx_dptr_bottom_offset = dx_dptr + bottom_offset;
    *(dx_dptr_bottom_offset + params.left_w_index) += static_cast<T>((1 - params.w_lerp) * dbottom);
    *(dx_dptr_bottom_offset + params.right_w_index) += static_cast<T>(params.w_lerp * dbottom);
    const T dtop = dy - dbottom;
    T* dx_dptr_top_offset = dx_dptr + top_offset;
    *(dx_dptr_top_offset + params.left_w_index) += static_cast<T>((1 - params.w_lerp) * dtop);
    *(dx_dptr_top_offset + params.right_w_index) += static_cast<T>(params.w_lerp * dtop);
  }
}

}  // namespace

template<typename T>
class UpsampleBilinear2DCPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBilinear2DCPUKernel() = default;
  ~UpsampleBilinear2DCPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const bool align_corners = ctx->Attr<bool>("align_corners");
    const int64_t elem_cnt = y_tensor->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> in_helper(x_tensor->shape().At(0), x_tensor->shape().At(1),
                                              x_tensor->shape().At(2), x_tensor->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> out_helper(y_tensor->shape().At(0), y_tensor->shape().At(1),
                                               y_tensor->shape().At(2), y_tensor->shape().At(3));

    const int64_t nbatch = x_tensor->shape().At(0);
    const int64_t channels = x_tensor->shape().At(1);
    const int64_t in_height = x_tensor->shape().At(2);
    const int64_t in_width = x_tensor->shape().At(3);
    const int64_t out_height = y_tensor->shape().At(2);
    const int64_t out_width = y_tensor->shape().At(3);

    if (in_height == out_height && in_width == out_width) {
      memcpy(y_tensor->mut_dptr<void>(), x_tensor->dptr<void>(),
             sizeof(T) * nbatch * channels * in_height * in_width);
    } else {
      const T scale_height = GetAreaPixelScale(in_height, out_height, align_corners, height_scale);
      const T scale_width = GetAreaPixelScale(in_width, out_width, align_corners, width_scale);
      UpsampleBilinear2DForward<T>(elem_cnt, x_tensor->dptr<T>(), in_helper, out_helper, in_height,
                                   in_width, scale_height, scale_width, align_corners,
                                   y_tensor->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleBilinear2DGradCPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBilinear2DGradCPUKernel() = default;
  ~UpsampleBilinear2DGradCPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    Memset<DeviceType::kCPU>(ctx->device_ctx(), dx_tensor->mut_dptr<T>(), 0,
                             dx_tensor->shape().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const bool align_corners = ctx->Attr<bool>("align_corners");
    const int64_t elem_cnt = dy_tensor->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> dy_helper(dy_tensor->shape().At(0), dy_tensor->shape().At(1),
                                              dy_tensor->shape().At(2), dy_tensor->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> dx_helper(dx_tensor->shape().At(0), dx_tensor->shape().At(1),
                                              dx_tensor->shape().At(2), dx_tensor->shape().At(3));

    const int64_t nbatch = dx_tensor->shape().At(0);
    const int64_t channels = dx_tensor->shape().At(1);
    const int64_t in_height = dx_tensor->shape().At(2);
    const int64_t in_width = dx_tensor->shape().At(3);
    const int64_t out_height = dy_tensor->shape().At(2);
    const int64_t out_width = dy_tensor->shape().At(3);
    if (in_height == out_height && in_width == out_width) {
      memcpy(dx_tensor->mut_dptr<void>(), dy_tensor->dptr<void>(),
             sizeof(T) * nbatch * channels * in_height * in_width);
    } else {
      const T scale_height = GetAreaPixelScale(in_height, out_height, align_corners, height_scale);
      const T scale_width = GetAreaPixelScale(in_width, out_width, align_corners, width_scale);
      UpsampleBilinearBackward<T>(elem_cnt, dy_tensor->dptr<T>(), dy_helper, dx_helper, in_height,
                                  in_width, scale_height, scale_width, align_corners,
                                  dx_tensor->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_BILINEAR_2D_CPU_KERNEL(dtype)                                \
  REGISTER_USER_KERNEL("upsample_bilinear_2d")                                         \
      .SetCreateFn<UpsampleBilinear2DCPUKernel<dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("upsample_bilinear_2d_grad")                                    \
      .SetCreateFn<UpsampleBilinear2DGradCPUKernel<dtype>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_UPSAMPLE_BILINEAR_2D_CPU_KERNEL(float)
REGISTER_UPSAMPLE_BILINEAR_2D_CPU_KERNEL(double)
REGISTER_UPSAMPLE_BILINEAR_2D_CPU_KERNEL(int)

}  // namespace oneflow
