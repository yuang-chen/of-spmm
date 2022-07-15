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
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/gather_kernel_util.h"
#include "oneflow/user/kernels/unsorted_segment_sum_kernel_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/embedding/embedding_manager.h"
#include "oneflow/core/control/ctrl_client.h"

namespace oneflow {

namespace {

template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

template<typename T, int32_t pack_size>
__device__ __inline__ void AtomicAdd(Pack<T, pack_size>* address, Pack<T, pack_size> val) {
  for (int i = 0; i < pack_size; ++i) {
    atomicAdd(reinterpret_cast<T*>(address) + i, static_cast<T>(val.elem[i]));
  }
}

template<>
__device__ __inline__ void AtomicAdd<half, 2>(Pack<half, 2>* address, Pack<half, 2> val) {
  half2 h2_val;
  h2_val.x = static_cast<half>(val.elem[0]);
  h2_val.y = static_cast<half>(val.elem[1]);
  atomicAdd(reinterpret_cast<half2*>(address), h2_val);
}

template<typename T, typename IDX, int pack_size, int N>
struct Param {
  IDX* cur_rank_inverse_indices[N];
  const Pack<T, pack_size>* unique_partitioned_embedding_grads[N];
  int32_t* is_kernel_start[N];
  const IDX* num_unique_matrix;
  Pack<T, pack_size>* cur_rank_unique_embedding_grad_ptr;
};

template<typename T, typename IDX, int pack_size, int N>
__global__ void EmbeddingGraidientShuffleCudaKernel(int64_t parallel_id, int64_t parallel_num,
                                                    int64_t embedding_num_pack,
                                                    Param<T, IDX, pack_size, N> param) {
#pragma unroll 1
  for (int i = 0; i < parallel_num; ++i) {
    int rank_id = (parallel_id + i) % parallel_num;
    IDX cur_rank_index_offset = 0;
    for (int k = 0; k < rank_id; ++k) {
      cur_rank_index_offset += param.num_unique_matrix[k * parallel_num + parallel_id];
    }
    IDX in_index_offset = 0;
    for (int k = 0; k < parallel_id; ++k) {
      in_index_offset += param.num_unique_matrix[rank_id * parallel_num + k];
    }
    const IDX* cur_rank_inverse_indices_ptr =
        param.cur_rank_inverse_indices[rank_id] + cur_rank_index_offset;
    const Pack<T, pack_size>* unique_partitioned_embedding_grad_ptr =
        param.unique_partitioned_embedding_grads[rank_id] + in_index_offset * embedding_num_pack;
    Pack<T, pack_size>* cur_rank_unique_embedding_grad_ptr =
        param.cur_rank_unique_embedding_grad_ptr;
    CUDA_1D_KERNEL_LOOP_T(
        int, j,
        param.num_unique_matrix[rank_id * parallel_num + parallel_id] * embedding_num_pack) {
      int in_row_id = j / embedding_num_pack;
      int col_id = j - in_row_id * embedding_num_pack;
      int out_row_id = cur_rank_inverse_indices_ptr[in_row_id];
      Pack<T, pack_size> grad_val = unique_partitioned_embedding_grad_ptr[j];
      AtomicAdd(cur_rank_unique_embedding_grad_ptr + out_row_id * embedding_num_pack + col_id,
                grad_val);
    }
  }
}

template<typename T, typename IDX, int pack_size, int N>
__global__ void BarrierKernel(int32_t parallel_id, int32_t parallel_num,
                              Param<T, IDX, pack_size, N> param) {
  int count = *param.is_kernel_start[parallel_id];
  volatile int32_t* start_f = param.is_kernel_start[parallel_id];
  *start_f = count + 1;
  // printf("\nparallel_id %d set to %d\n", parallel_id, *start_f);
  for (int k = 0; k < parallel_num; ++k) {
    volatile int32_t* is_kernel_start_ptr = param.is_kernel_start[k];
    while (*is_kernel_start_ptr < count + 1)
      ;
  }
}

void GetPtrs(user_op::KernelComputeContext* ctx,
             std::vector<void*>* unique_partitioned_embedding_grad_ptr,
             std::vector<void*>* cur_rank_inverse_indices_ptr,
             std::vector<void*>* is_kernel_start_ptr) {
  const int64_t num_ids =
      ctx->TensorDesc4ArgNameAndIndex("inverse_unique_partition_indices", 0)->shape().elem_cnt();
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
  std::string name =
      ctx->op_name()
      + std::to_string(num_ids);  // train and eval same op name. do it in pass? or use newUniqueId

  std::vector<cudaIpcMemHandle_t> handle;
  handle.resize(3);
  OF_CUDA_CHECK(
      cudaIpcGetMemHandle(&handle.at(0), unique_partitioned_embedding_grad_ptr->at(parallel_id)));
  OF_CUDA_CHECK(cudaIpcGetMemHandle(&handle.at(1), cur_rank_inverse_indices_ptr->at(parallel_id)));
  OF_CUDA_CHECK(cudaIpcGetMemHandle(&handle.at(2), is_kernel_start_ptr->at(parallel_id)));

  Singleton<CtrlClient>::Get()->PushKV(
      name + std::to_string(parallel_id),
      std::string(reinterpret_cast<const char*>(handle.data()), 3 * sizeof(cudaIpcMemHandle_t)));
  for (int64_t i = 0; i < parallel_num; ++i) {
    std::string key = name + std::to_string(i);
    if (parallel_id != i) {
      std::vector<cudaIpcMemHandle_t> handle;
      handle.resize(3);
      Singleton<CtrlClient>::Get()->PullKV(key, [i, &handle](const std::string& val) {
        memcpy(handle.data(), val.data(), 3 * sizeof(cudaIpcMemHandle_t));
      });
      OF_CUDA_CHECK(cudaIpcOpenMemHandle(&unique_partitioned_embedding_grad_ptr->at(i),
                                         handle.at(0), cudaIpcMemLazyEnablePeerAccess));
      OF_CUDA_CHECK(cudaIpcOpenMemHandle(&cur_rank_inverse_indices_ptr->at(i), handle.at(1),
                                         cudaIpcMemLazyEnablePeerAccess));
      OF_CUDA_CHECK(cudaIpcOpenMemHandle(&is_kernel_start_ptr->at(i), handle.at(2),
                                         cudaIpcMemLazyEnablePeerAccess));
    }
    // LOG(ERROR) << "rank " << parallel_id << " i " << i << " unique_partitioned_embedding_grad_ptr
    // "
    //           << unique_partitioned_embedding_grad_ptr->at(i) << " cur_rank_inverse_indices_ptr_
    //           "
    //           << cur_rank_inverse_indices_ptr->at(i) <<"is_kernel_start_ptr" <<
    //           is_kernel_start_ptr->at(i);
  }
}

template<typename IDX>
class DataShuffleKernelState final : public user_op::OpKernelState {
 public:
  explicit DataShuffleKernelState(user_op::KernelInitContext* ctx)
      : device_index_(-1), parallel_desc_(ctx->parallel_desc()) {
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    int64_t parallel_num = parallel_desc_.parallel_num();
    unique_partitioned_embedding_grad_ptr_.resize(parallel_num);
    cur_rank_inverse_indices_ptr_.resize(parallel_num);
    is_kernel_start_ptr_.resize(parallel_num);
    size_t is_kernel_start_size = GetCudaAlignedSize(sizeof(int32_t));
    OF_CUDA_CHECK(cudaMalloc(&is_kernel_start_ptr_.at(parallel_id), is_kernel_start_size));
    OF_CUDA_CHECK(cudaMemset(is_kernel_start_ptr_.at(parallel_id), 0, is_kernel_start_size));

    size_t unique_partitioned_embedding_grads_size =
        ctx->TensorDesc4ArgNameAndIndex("embedding_grad", 0)->shape().elem_cnt() * sizeof(half);
    OF_CUDA_CHECK(cudaMalloc(&unique_partitioned_embedding_grad_ptr_.at(parallel_id),
                             unique_partitioned_embedding_grads_size));
    size_t cur_rank_inverse_indices_size =
        ctx->TensorDesc4ArgNameAndIndex("cur_rank_inverse_indices", 0)->shape().elem_cnt()
        * sizeof(IDX);
    OF_CUDA_CHECK(
        cudaMalloc(&cur_rank_inverse_indices_ptr_.at(parallel_id), cur_rank_inverse_indices_size));
  }

  ~DataShuffleKernelState() {
    // free
  }

  std::vector<void*>* UniquePartitionedEmbeddingGrads() {
    return &unique_partitioned_embedding_grad_ptr_;
  }

  std::vector<void*>* CurRankInverseIndices() { return &cur_rank_inverse_indices_ptr_; }

  std::vector<void*>* IsKernelStart() { return &is_kernel_start_ptr_; }

 private:
  int device_index_;
  ParallelDesc parallel_desc_;
  std::vector<void*> unique_partitioned_embedding_grad_ptr_;
  std::vector<void*> cur_rank_inverse_indices_ptr_;
  std::vector<void*> is_kernel_start_ptr_;
};

constexpr int pack_size = 2;

}  // namespace

template<typename T, typename IDX>
class EmbeddingGraidientShuffleP2PKernel final : public user_op::OpKernel {
 public:
  EmbeddingGraidientShuffleP2PKernel() : current_iter_(0) {}
  ~EmbeddingGraidientShuffleP2PKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<DataShuffleKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    CHECK(!embedding::UseDynamicMemoryAllocation());
    CHECK(ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_FUSE_EMBEDDING_INTERACTION", false));
    auto* kernel_state = dynamic_cast<DataShuffleKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* embedding_grad = ctx->Tensor4ArgNameAndIndex("embedding_grad", 0);
    const user_op::Tensor* num_unique_matrix = ctx->Tensor4ArgNameAndIndex("num_unique_matrix", 0);
    const user_op::Tensor* cur_rank_inverse_indices =
        ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0);
    user_op::Tensor* cur_rank_unique_embedding_grad =
        ctx->Tensor4ArgNameAndIndex("cur_rank_unique_embedding_grad", 0);

    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const bool only_zero_valid_grad = ctx->Attr<bool>("only_zero_valid_grad");
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    const bool skip_first_scatter = ctx->Attr<bool>("skip_first_scatter");
    CHECK(skip_first_scatter);
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    if (current_iter_ == 0) {
      GetPtrs(ctx, kernel_state->UniquePartitionedEmbeddingGrads(),
              kernel_state->CurRankInverseIndices(), kernel_state->IsKernelStart());
    }
    OF_CUDA_CHECK(cudaMemcpyAsync(
        kernel_state->UniquePartitionedEmbeddingGrads()->at(parallel_id), embedding_grad->dptr(),
        embedding_grad->shape_view().elem_cnt() * sizeof(half), cudaMemcpyDefault, cuda_stream));
    OF_CUDA_CHECK(cudaMemcpyAsync(kernel_state->CurRankInverseIndices()->at(parallel_id),
                                  cur_rank_inverse_indices->dptr(),
                                  cur_rank_inverse_indices->shape_view().elem_cnt() * sizeof(IDX),
                                  cudaMemcpyDefault, cuda_stream));

    Param<T, IDX, pack_size, 8> param;
    CHECK_LE(parallel_num, 8);
    param.cur_rank_unique_embedding_grad_ptr =
        reinterpret_cast<Pack<T, pack_size>*>(cur_rank_unique_embedding_grad->mut_dptr<T>());
    for (int i = 0; i < parallel_num; ++i) {
      param.cur_rank_inverse_indices[i] =
          reinterpret_cast<IDX*>(kernel_state->CurRankInverseIndices()->at(i));
      param.unique_partitioned_embedding_grads[i] = reinterpret_cast<Pack<T, pack_size>*>(
          kernel_state->UniquePartitionedEmbeddingGrads()->at(i));
      param.is_kernel_start[i] = reinterpret_cast<int32_t*>(kernel_state->IsKernelStart()->at(i));
    }
    param.num_unique_matrix = reinterpret_cast<const uint32_t*>(num_unique_matrix->dptr());
    int64_t embedding_num_pack = embedding_size / pack_size;
    OF_CUDA_CHECK(cudaMemsetAsync(
        cur_rank_unique_embedding_grad->mut_dptr(), 0,
        cur_rank_unique_embedding_grad->shape_view().elem_cnt() * sizeof(T), cuda_stream));
    BarrierKernel<<<1, 1, 0, cuda_stream>>>(parallel_id, parallel_num, param);
    EmbeddingGraidientShuffleCudaKernel<<<216, kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
        parallel_id, parallel_num, embedding_num_pack, param);
    BarrierKernel<<<1, 1, 0, cuda_stream>>>(parallel_id, parallel_num, param);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

REGISTER_USER_KERNEL("embedding_gradient_shuffle")
    .SetCreateFn<EmbeddingGraidientShuffleP2PKernel<half, uint32_t>>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("embedding_grad", 0) == DataType::kFloat16)
                     && (user_op::HobDataType("num_unique_matrix", 0) == DataType::kUInt32)
                     && ParseBooleanFromEnv("EMBEDDING_GRADIENT_SHUFFLE_USE_P2P_KERNEL", false));

}  // namespace oneflow
