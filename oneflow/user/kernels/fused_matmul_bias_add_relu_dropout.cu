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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/user/kernels/cublas_fused_mlp_util.cuh"
#include "oneflow/user/kernels/dropout_kernel.h"
// CUBLAS_AUX_EPILOGUE only support in cuda11.4 or higher version, in cuda11.4 it need static link.
#if CUDA_VERSION >= 11060

namespace oneflow {

namespace {

constexpr int32_t kVecSize = 4;
constexpr int32_t kBlockSize = 256;

template<typename T>
constexpr int32_t GetDropoutPackSize() {
  // For float, bfloat16, half.
  return 4;
};

template<>
constexpr int32_t GetDropoutPackSize<half2>() {
  return 2;
};

template<>
constexpr int32_t GetDropoutPackSize<double>() {
  return 2;
};

union RandPack4 {
  float4 storage;
  float elem[4];
};

template<typename T>
struct GetPack2Type {
  using T2 = typename std::aligned_storage<2 * sizeof(T), 2 * sizeof(T)>::type;
};

template<>
struct GetPack2Type<half> {
  using T2 = half2;
};

#if CUDA_VERSION >= 11000
template<>
struct GetPack2Type<nv_bfloat16> {
  using T2 = nv_bfloat162;
};
#endif

template<typename T>
using Pack2Type = typename GetPack2Type<T>::T2;

using H2PackType = typename std::aligned_storage<4 * sizeof(half), 4 * sizeof(half)>::type;

template<typename T>
union H2Pack {
  cuda::elementwise::Pack<T, 4> pack_storage;
  Pack2Type<T> h2[2];
  __device__ H2Pack() {
    // do nothing
  }
};

template<>
union H2Pack<half> {
  cuda::elementwise::Pack<half, 4> pack_storage;
  half2 h2[2];
  __device__ H2Pack() {
    // do nothing
  }
};

#if CUDA_VERSION >= 11000
template<>
union H2Pack<nv_bfloat16> {
  cuda::elementwise::Pack<nv_bfloat16, 4> pack_storage;
  nv_bfloat162 h2[2];
  __device__ H2Pack() {
    // do nothing
  }
};
#endif

template<typename T>
__device__ Pack2Type<T> Make2(float v);

template<>
__device__ Pack2Type<half> Make2<half>(float v) {
  return __float2half2_rn(v);
}

#if CUDA_VERSION >= 11000
template<>
__device__ Pack2Type<nv_bfloat16> Make2<nv_bfloat16>(float v) {
  return __float2bfloat162_rn(v);
}
#endif

constexpr int32_t kWarpSize = 32; 

int32_t AlignReluAuxLd(int32_t aux_ld) {
  /*
  ReLu bit-mask matrix leading dimension in elements.
  Must be divisible by 128 and be no less than the number of rows in the output matrix.
  */
  int32_t old_aux_ld = aux_ld;
  return ((old_aux_ld + kAuxReluLdAlignRequirement - 1) / kAuxReluLdAlignRequirement)
         * kAuxReluLdAlignRequirement;
}

template<int32_t unroll>
__device__ void SetCublasBitMask(const int32_t aux_ld,
                           const int64_t row,  
                           const int64_t col, 
                           int32_t thread_bitmask,
                           int32_t* mask){
    int32_t linear_index = row * aux_ld + col; 
    int32_t mask_index = linear_index / kWarpSize; 
    int32_t mask_offset = linear_index - mask_index * kWarpSize; 
    int32_t bitmask = thread_bitmask << mask_offset; 
    for (int stride = kWarpSize / (unroll * 2); stride > 0; stride /= 2) {
      bitmask |= __shfl_down_sync(__activemask(), bitmask, stride, kWarpSize);
    }
    if (mask_offset == 0) {
      int32_t aux_ld_load_index = linear_index / 32; // cause we use int32 to store aux_ld
      mask[aux_ld_load_index] = bitmask;
    }
}

#if CUDA_VERSION >= 11000
#define RETURN_VOID_IF_HALF                                                                        \
  typename std::enable_if_t<(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value), \
                            void>
#else
#define RETURN_VOID_IF_HALF typename std::enable_if_t<std::is_same<T, half>::value, void>
#endif
#define RETURN_VOID_IF_FLOAT typename std::enable_if_t<std::is_same<T, float>::value, void>
#define RETURN_VOID_IF_DOUBLE typename std::enable_if_t<std::is_same<T, double>::value, void>

template<typename T, int pack_size, bool relu>
__global__ RETURN_VOID_IF_FLOAT FusedVectorizedReluDropoutKernel(uint64_t seed,
                                                    one::CUDAGeneratorState* cuda_gen_state,
                                                    uint64_t inc_offset, const int64_t elem_cnt,
                                                    const int32_t aux_ld, 
                                                    const int64_t cols,  
                                                    float rate, float scale, 
                                                    const T* x, int32_t* mask, T* y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  using MaskType = cuda::elementwise::PackType<bool, pack_size>;
  using MaskPack = cuda::elementwise::Pack<bool, pack_size>;

  T t_scale = static_cast<T>(scale);
  RandPack4 rand_uniform_pack4;
  T zero_val = static_cast<T>(0.0);
  for (int64_t linear_index = global_thread_id * pack_size,
               step = gridDim.x * blockDim.x * pack_size;
       linear_index < elem_cnt; linear_index += step) {
    const int64_t row = linear_index / cols; 
    const int64_t col = linear_index - row * cols; 
    int32_t thread_bitmask = 0; 

    rand_uniform_pack4.storage = curand_uniform4(&state);
    const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;

    MaskPack mask_vec;
    LoadPack y_vec;
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      bool relu_mask = true; 
      if(relu){
        // Relu
        relu_mask = x_vec.elem[i] > zero_val; 
      }
      // dropout
      bool mask_val = rand_uniform_pack4.elem[i] > rate;
      // Combined relu_mask, dropout_mask together. 
      bool combined_mask = relu_mask && mask_val; 
      T float_combined_mask = static_cast<float>(combined_mask);
      thread_bitmask |= (combined_mask << i); 
      y_vec.elem[i] = x_vec.elem[i] * float_combined_mask * t_scale;
    }
    *(reinterpret_cast<LoadType*>(y + linear_index)) = y_vec.storage;
    SetCublasBitMask<4>(aux_ld, row, col, thread_bitmask, mask);
  }

  if (threadIdx.x == 0) {
    int32_t new_counter = cuda::atomic::Add(&cuda_gen_state->dev_counter, 1) + 1;
    if (new_counter == gridDim.x) {
      cuda_gen_state->dev_counter = 0;           // reset counter to zero
      cuda_gen_state->dev_offset += inc_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<typename T, bool relu>
__global__ void FusedReluDropoutKernel(uint64_t seed,
                                       one::CUDAGeneratorState* cuda_gen_state,
                                       uint64_t inc_offset, const int64_t elem_cnt,
                                       const int32_t aux_ld, 
                                       const int64_t rows,  
                                       const int64_t cols,  
                                       float rate, float scale, 
                                       const T* x, int32_t* mask, T* y) {
  const int32_t lane_id = threadIdx.x; 
  const int32_t global_warp_id = blockIdx.x * blockDim.y + threadIdx.y; 
  const int32_t step = gridDim.x * blockDim.y; 
  const int32_t global_thread_id = global_warp_id * kWarpSize + lane_id; 

  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  T t_scale = static_cast<T>(scale);
  T zero_val = static_cast<T>(0.0);

  for(int32_t row = global_warp_id; row < rows; row += step){
    for(int32_t col = lane_id; col < cols; col += kWarpSize){
      int32_t thread_bitmask = 0; 
      const int64_t linear_index = row * cols + col; 

      T x_val = x[linear_index]; 
      float rand_uniform_val = curand_uniform(&state);
      bool relu_mask = true; 
      if(relu){
        // relu
        relu_mask = x_val > zero_val; 
      }
      // dropout
      bool mask_val = rand_uniform_val > rate;
      // Combined relu_mask, dropout_mask together. 
      bool combined_mask = relu_mask && mask_val;
      thread_bitmask = combined_mask; 

      T y_val = y[linear_index]; 
      T t_combined_mask = static_cast<T>(combined_mask);
      y_val = x_val * t_combined_mask * t_scale;
      y[linear_index] = y_val; 
      SetCublasBitMask<1>(aux_ld, row, col, thread_bitmask, mask);
    }
  }

  if (threadIdx.x == 0) {
    int32_t new_counter = cuda::atomic::Add(&cuda_gen_state->dev_counter, 1) + 1;
    if (new_counter == gridDim.x) {
      cuda_gen_state->dev_counter = 0;           // reset counter to zero
      cuda_gen_state->dev_offset += inc_offset;  // maintain the state of generator's dev_offset
    }
  }
}


// template<typename T, int pack_size, bool tail>
// __global__ RETURN_VOID_IF_HALF FusedVectorizedReluDropoutKernel(uint64_t seed,
//                                                    one::CUDAGeneratorState* cuda_gen_state,
//                                                    uint64_t inc_offset, const int64_t elem_cnt,
//                                                    float rate, float scale, int64_t n_tail,
//                                                    const T* x, int32_t* mask, T* y, const T* tail_x,
//                                                    int32_t* tail_mask, T* tail_y) {
//   int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//   curandStatePhilox4_32_10_t state;
//   curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
//   using LoadType = cuda::elementwise::PackType<T, pack_size>;
//   using LoadPack = cuda::elementwise::Pack<T, pack_size>;
//   using StoreType = cuda::elementwise::PackType<Pack2Type<T>, pack_size / 2>;
//   using StorePack = cuda::elementwise::Pack<Pack2Type<T>, pack_size / 2>;
//   using MaskType = cuda::elementwise::PackType<bool, pack_size>;
//   using MaskPack = cuda::elementwise::Pack<bool, pack_size>;

//   RandPack4 rand_uniform_pack4;
//   Pack2Type<T> h2_scale = Make2<T>(scale);
//   T zero_val = static_cast<T>(0.0);

//   for (int64_t linear_index = global_thread_id * pack_size,
//                step = gridDim.x * blockDim.x * pack_size;
//        linear_index < elem_cnt; linear_index += step) {
//     rand_uniform_pack4.storage = curand_uniform4(&state);
//     const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
//     H2Pack<T> x_vec{};
//     x_vec.pack_storage.storage = *x_load;

//     MaskPack mask_vec;
//     StorePack y_vec;
//     StorePack one_or_zero_h2;

//     mask_vec.elem[0] = rand_uniform_pack4.elem[0] > rate;
//     float tmp_float_mask = static_cast<float>(mask_vec.elem[0]);
//     one_or_zero_h2.elem[0].x = tmp_float_mask;
//     mask_vec.elem[1] = rand_uniform_pack4.elem[1] > rate;
//     tmp_float_mask = static_cast<float>(mask_vec.elem[1]);
//     one_or_zero_h2.elem[0].y = tmp_float_mask;

//     // relu
//     x_vec.h2[0].x = x_vec.h2[0].x > zero_val ? x_vec.h2[0].x : x_vec.h2[0].x;
//     x_vec.h2[0].y = x_vec.h2[0].y > zero_val ? x_vec.h2[0].y : x_vec.h2[0].y;
//     // dropout
//     y_vec.elem[0] = __hmul2(__hmul2(x_vec.h2[0], one_or_zero_h2.elem[0]), h2_scale);

//     mask_vec.elem[2] = rand_uniform_pack4.elem[2] > rate;
//     tmp_float_mask = static_cast<float>(mask_vec.elem[2]);
//     one_or_zero_h2.elem[1].x = tmp_float_mask;
//     mask_vec.elem[3] = rand_uniform_pack4.elem[3] > rate;
//     tmp_float_mask = static_cast<float>(mask_vec.elem[3]);
//     one_or_zero_h2.elem[1].y = tmp_float_mask;

//     // relu
//     x_vec.h2[1].x = x_vec.h2[1].x > zero_val ? x_vec.h2[1].x : x_vec.h2[1].x;
//     x_vec.h2[1].y = x_vec.h2[1].y > zero_val ? x_vec.h2[1].y : x_vec.h2[1].y;
//     // dropout
//     y_vec.elem[1] = __hmul2(__hmul2(x_vec.h2[1], one_or_zero_h2.elem[1]), h2_scale);

//     *(reinterpret_cast<StoreType*>(y + linear_index)) = y_vec.storage;
//     *(reinterpret_cast<MaskType*>(mask + linear_index)) = mask_vec.storage;
//   }

//   if (tail && global_thread_id < n_tail) {
//     // relu
//     T tail_x_val = tail_x[global_thread_id];
//     T tail_out = tail_x_val > zero_val ? tail_x_val : zero_val;
//     // dropout
//     const float rand_uniform = curand_uniform(&state);
//     const bool mask_val = rand_uniform > rate;
//     tail_mask[global_thread_id] = mask_val;
//     float tmp_half_mask = static_cast<float>(mask_val);
//     tail_out = tail_out * static_cast<T>(tmp_half_mask) * h2_scale.x;
//     tail_y[global_thread_id] = tail_out;
//   }

//   __syncthreads();
//   if (threadIdx.x == 0) {
//     int32_t new_counter = cuda::atomic::Add(&cuda_gen_state->dev_counter, 1) + 1;
//     if (new_counter == gridDim.x) {
//       cuda_gen_state->dev_counter = 0;           // reset counter to zero
//       cuda_gen_state->dev_offset += inc_offset;  // maintain the state of generator's dev_offset
//     }
//   }
// }

// template<typename T, int pack_size, bool tail>
// __global__ RETURN_VOID_IF_DOUBLE FusedVectorizedReluDropoutKernel(uint64_t seed,
//                                                      one::CUDAGeneratorState* cuda_gen_state,
//                                                      uint64_t inc_offset, const int64_t elem_cnt,
//                                                      float rate, float scale, int64_t n_tail,
//                                                      const T* x, int32_t* mask, T* y, const T* tail_x,
//                                                      int32_t* tail_mask, T* tail_y) {
//   int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//   curandStatePhilox4_32_10_t state;
//   curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
//   using LoadType = cuda::elementwise::PackType<T, pack_size>;
//   using LoadPack = cuda::elementwise::Pack<T, pack_size>;
//   using MaskType = cuda::elementwise::PackType<bool, pack_size>;
//   using MaskPack = cuda::elementwise::Pack<bool, pack_size>;

//   RandPack4 rand_uniform_pack4;
//   bool grid_loop_rand_state = 0;
//   T zero_val = static_cast<T>(0.0);

//   for (int64_t linear_index = global_thread_id * pack_size; linear_index < elem_cnt;
//        linear_index += gridDim.x * blockDim.x * pack_size) {
//     if (grid_loop_rand_state == 0) {
//       rand_uniform_pack4.storage = curand_uniform4(&state);
//       grid_loop_rand_state ^= 1;
//     } else {
//       // Use the last two random numbers we generated in previous iteration.
//       rand_uniform_pack4.elem[0] = rand_uniform_pack4.elem[2];
//       rand_uniform_pack4.elem[1] = rand_uniform_pack4.elem[3];
//       grid_loop_rand_state ^= 1;
//     }
//     const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
//     LoadPack x_vec;
//     x_vec.storage = *x_load;

//     MaskPack mask_vec;
//     LoadPack y_vec;
// #pragma unroll
//     for (int i = 0; i < pack_size; i++) {
//       // Relu
//       y_vec.elem[i] = x_vec.elem[i] > zero_val ? x_vec.elem[i] : zero_val;
//       // Dropout
//       mask_vec.elem[i] = rand_uniform_pack4.elem[i] > rate;
//       y_vec.elem[i] = y_vec.elem[i] * mask_vec.elem[i] * scale;
//     }
//     *(reinterpret_cast<LoadType*>(y + linear_index)) = y_vec.storage;
//     *(reinterpret_cast<MaskType*>(mask + linear_index)) = mask_vec.storage;
//   }

//   if (tail && global_thread_id < n_tail) {
//     // relu
//     T tail_x_val = tail_x[global_thread_id];
//     T tail_out = tail_x_val > zero_val ? tail_x_val : zero_val;
//     // dropout
//     const float rand_uniform = curand_uniform(&state);
//     const bool mask_val = rand_uniform > rate;
//     tail_mask[global_thread_id] = mask_val;
//     tail_out = tail_out * mask_val * scale;
//     tail_y[global_thread_id] = tail_out;
//   }

//   __syncthreads();
//   if (threadIdx.x == 0) {
//     int32_t new_counter = cuda::atomic::Add(&cuda_gen_state->dev_counter, 1) + 1;
//     if (new_counter == gridDim.x) {
//       cuda_gen_state->dev_counter = 0;           // reset counter to zero
//       cuda_gen_state->dev_offset += inc_offset;  // maintain the state of generator's dev_offset
//     }
//   }
// }

template<int pack_size>
unsigned int ComputeGridSize(ep::Stream* stream, const int32_t block_size, const int64_t elem_cnt) {
  auto* cuda_stream = stream->As<ep::CudaStream>();
  const int32_t max_threads_multi_process =
      cuda_stream->device_properties().maxThreadsPerMultiProcessor;
  const int32_t multi_processor_count = cuda_stream->device_properties().multiProcessorCount;
  unsigned int blocks_per_sm = max_threads_multi_process / block_size;
  unsigned int grid_size = ((elem_cnt + block_size - 1) / block_size);
  grid_size = std::min((unsigned int)multi_processor_count * blocks_per_sm, grid_size);
  return grid_size;
}

inline cudaError_t GetWarpImplNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves,
  int* num_blocks) {
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
  *num_blocks =
  std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
  return cudaSuccess;
}

template<typename T, bool relu>
void LaunchFusedReluDropoutKernel(ep::CudaStream* stream, uint64_t seed,
                                  one::CUDAGeneratorState* cuda_gen_state, const int64_t elem_cnt,
                                  const int32_t aux_ld, const int64_t rows, const int64_t cols, 
                                  float rate, float scale, const T* x, int32_t* mask, T* y) {
  printf("Rows is: %ld, Cols is: %ld \n", rows, cols); 
  uint64_t inc_offset = 0;

  if(cols % 4 == 0){
    constexpr int pack_size = GetDropoutPackSize<T>();
    const int64_t pack_num = elem_cnt / pack_size;
    unsigned int grid_size = ComputeGridSize<4>(stream, kBlockSize, elem_cnt);
    FusedVectorizedReluDropoutKernel<T, pack_size, relu><<<grid_size, kBlockSize, 0, stream->cuda_stream()>>>(
      seed, cuda_gen_state, inc_offset, elem_cnt, aux_ld, cols, rate, scale, x, mask, y
    );   
  } else {
    printf("Launch this \n"); 
    constexpr int block_size = 128;
    constexpr int waves = 32;
    dim3 block_dim(32, 4);
    const int64_t num_blocks = (rows + 4 - 1) / 4; // since each block has 4 warps. 
    int grid_dim_x = 0;
    cudaError_t err = GetWarpImplNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
    // if (err != cudaSuccess) { return err; }
    FusedReluDropoutKernel<T, relu><<<grid_dim_x, block_dim, 0, stream->cuda_stream()>>>(
      seed, cuda_gen_state, 
      inc_offset, elem_cnt, aux_ld, 
      rows, cols, rate, scale, x, mask, y
    ); 
  }
}

template<typename T>
class FusedMatmulBiasAddReluDropoutKernel final : public user_op::OpKernel,
                                                  public user_op::CudaGraphSupport {
 public:
  FusedMatmulBiasAddReluDropoutKernel() = default;
  ~FusedMatmulBiasAddReluDropoutKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateCublasFusedMLPKernelCache();
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* dropout_state,
               const user_op::OpKernelCache* cache) const override {
    /*
    Fused DenseActivation Layer. Assume we have two layers:
    A: (m, k)
    B: (n, k) need transpose
    C: (j, n) need transpose
    tmp: A matmul B(transpose), its shape is (m, n)
    out: tmp matmul C(transpose), its shape is (m, j)
    */
    const int32_t weight_size = ctx->input_size("weights");
    const int32_t bias_size = ctx->input_size("biases");
    CHECK_EQ(weight_size, bias_size) << "The number of weight and bias is not equal!. ";
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    const auto* matmul_cache = CHECK_NOTNULL(dynamic_cast<const CublasFusedMLPKernelCache*>(cache));

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    bool skip_final_activation = ctx->Attr<bool>("skip_final_activation");

    auto* fused_dropout_kernel_state = dynamic_cast<FusedDropoutKernelState*>(dropout_state);
    CHECK_NOTNULL(fused_dropout_kernel_state);
    const auto& generator = fused_dropout_kernel_state->generator();
    CHECK_NOTNULL(generator);
    const auto device_index = ctx->stream()->device()->device_index();
    std::shared_ptr<one::CUDAGeneratorImpl> cuda_generator =
        CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>(device_index));
    uint64_t seed = cuda_generator->current_seed();
    const std::vector<float> dropout_rate_list = ctx->Attr<std::vector<float>>("dropout_rate_list");
    one::CUDAGeneratorState* cuda_gen_state = cuda_generator->cuda_gen_state();

    const DataType data_type = out->data_type();
    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);
    size_t cublas_m = 0, cublas_n = 0, cublas_k = 0;
    int64_t cublas_lda = 0, cublas_ldb = 0, cublas_ldc = 0;

    const double alpha = 1.0;
    const auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
    const double beta = 0.0;
    const auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    // Currently only support 2D matmul.
    DimVector in_shape(2);
    x->shape().ToDimVector(&in_shape);
    DimVector weight_shape(2);

    const void* in_buf_ptr = x->dptr();
    size_t offset = 0;
    for (int idx = 0; idx < weight_size; idx++) {
      const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weights", idx);
      const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("biases", idx);
      user_op::Tensor* cublas_aux = ctx->Tensor4ArgNameAndIndex("cublas_aux", idx);

      const int64_t batchsize = in_shape.at(0);
      const int64_t out_feature = weight->shape().At(0);
      weight->shape().ToDimVector(&weight_shape);
      size_t matmul_out_elem_cnt = batchsize * out_feature;
      size_t matmul_out_size = GetCudaAlignedSize(matmul_out_elem_cnt * sizeof(T));

      InferMatmulCublasMNK(in_shape, weight_shape,
                           /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                           /*transpose_b=*/ep::primitive::BlasTransposeType::T, &cublas_m,
                           &cublas_n, &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);

      cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
      T* relu_dropout_out_buf;
      void* matmul_out_ptr = (void*)(reinterpret_cast<char*>(tmp_buffer->mut_dptr()) + offset);
      offset += matmul_out_size;
      if (idx == weight_size - 1) {
        // relu_dropout_out_buf =
        //     reinterpret_cast<T*>(ctx->Tensor4ArgNameAndIndex("out", 0)->mut_dptr());
        matmul_out_ptr = ctx->Tensor4ArgNameAndIndex("out", 0)->mut_dptr(); 
      } else {
        relu_dropout_out_buf =
            reinterpret_cast<T*>(ctx->Tensor4ArgNameAndIndex("hidden", idx)->mut_dptr());
      }
      SetCublasAttr(matmul_cache, cublas_compute_dtype, cuda_data_type, /*need_aux=*/false,
                    /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                    /*transpose_b=*/ep::primitive::BlasTransposeType::T, epilogue, bias->dptr(),
                    /*aux_ptr=*/nullptr, cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb,
                    cublas_ldc);

      OF_CUBLAS_CHECK(cublasLtMatmul(
          cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, &sp_alpha, weight->dptr(),
          matmul_cache->cublas_a_desc, in_buf_ptr, matmul_cache->cublas_b_desc, &sp_beta,
          matmul_out_ptr, matmul_cache->cublas_c_desc, matmul_out_ptr, matmul_cache->cublas_c_desc,
          nullptr, cuda_stream->cublas_workspace(), cuda_stream->cublas_workspace_size(),
          cuda_stream->cuda_stream()));

      printf("Matmul out elemcnt is: %ld \n", matmul_out_elem_cnt);
      if (idx != weight_size - 1) {
        float rate = dropout_rate_list.at(idx);
        float scale = 0.0;
        const int32_t aux_ld = AlignReluAuxLd(out_feature); 
        printf("Aux ld is: %d \n", aux_ld); 
        printf("out feature is: %d \n", out_feature); 

        if (rate < 1.0f) { scale = 1.0f / (1.0f - rate); }
        LaunchFusedReluDropoutKernel<T, true>(cuda_stream, seed, cuda_gen_state, matmul_out_elem_cnt, 
                                              aux_ld, batchsize, out_feature, rate,
                                              scale, reinterpret_cast<T*>(matmul_out_ptr),
                                              reinterpret_cast<int32_t*>(cublas_aux->mut_dptr()),
                                              relu_dropout_out_buf);
        // Set relu_droput_out ptr as next layer's input.
        in_buf_ptr = relu_dropout_out_buf;
        // Set hidden_layer shape as next layer's input shape.
        in_shape.at(1) = out_feature;
      }
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MATMUL_BIAS_ADD_RELU_DROPOUT_KERNEL_GPU(cpp_type, data_type)     \
  REGISTER_USER_KERNEL("fused_matmul_bias_add_relu_dropout")                            \
      .SetCreateFn<FusedMatmulBiasAddReluDropoutKernel<cpp_type>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("out", 0) == data_type))                \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                               \
        const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);                    \
        const int64_t batchsize = x.shape().At(0);                                      \
        const int32_t weight_size = ctx->input_size("weights");                         \
        size_t tmp_size = 0;                                                            \
        for (int i = 0; i < weight_size - 1; i++) {                                     \
          const int64_t out_feature = ctx->InputTensorDesc("weights", i).shape().At(0); \
          tmp_size += GetCudaAlignedSize(batchsize * out_feature * sizeof(cpp_type));   \
        }                                                                               \
        return tmp_size;                                                                \
      });

// REGISTER_FUSED_MATMUL_BIAS_ADD_RELU_DROPOUT_KERNEL_GPU(double, DataType::kDouble)
REGISTER_FUSED_MATMUL_BIAS_ADD_RELU_DROPOUT_KERNEL_GPU(float, DataType::kFloat)
// REGISTER_FUSED_MATMUL_BIAS_ADD_RELU_DROPOUT_KERNEL_GPU(half, DataType::kFloat16)
// REGISTER_FUSED_MATMUL_BIAS_ADD_RELU_DROPOUT_KERNEL_GPU(nv_bfloat16, DataType::kBFloat16)

}  // namespace

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11060