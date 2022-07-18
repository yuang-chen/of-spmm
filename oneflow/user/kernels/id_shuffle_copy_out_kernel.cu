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
#include "oneflow/core/embedding/embedding_manager.h"

namespace oneflow {

namespace {

class DataShuffleKernelState final : public user_op::OpKernelState {
 public:
  explicit DataShuffleKernelState(user_op::KernelInitContext* ctx) {
    const std::string& embedding_name = ctx->Attr<std::string>("embedding_name");
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    embedding_state_ = Singleton<embedding::EmbeddingManager>::Get()->GetEmbeddingState(
        embedding_name, parallel_id);
  }
  ~DataShuffleKernelState() override = default;

  embedding::EmbeddingState* EmbeddingState() { return embedding_state_; }

 private:
  embedding::EmbeddingState* embedding_state_;
};

template<typename K, typename U, typename IDX>
struct Param {
  uint32_t final_num_unique_ids;
  const K* cur_rank_unique_ids;
  K* out_cur_rank_unique_ids;
  const U* cur_rank_unique_table_ids;
  U* out_cur_rank_unique_table_ids;
  uint32_t cur_rank_num_ids;
  const IDX* cur_rank_inverse_indices;
  IDX* out_cur_rank_inverse_indices;
  uint32_t num_ids;
  const IDX* inverse_unique_partition_indices;
  IDX* out_inverse_unique_partition_indices;
  uint32_t num_unique_matrix_cnt;
  const IDX* num_unique_matrix;
  IDX* out_num_unique_matrix;
  const IDX* cur_rank_num_unique;
  IDX* out_cur_rank_num_unique;
};

template<typename K, typename U, typename IDX>
__global__ void CopyGpu(Param<K, U, IDX> param) {
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, param.final_num_unique_ids) {
    param.out_cur_rank_unique_ids[i] = param.cur_rank_unique_ids[i];
    param.out_cur_rank_unique_table_ids[i] = param.cur_rank_unique_table_ids[i];
  }
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, param.cur_rank_num_ids) {
    param.out_cur_rank_inverse_indices[i] = param.cur_rank_inverse_indices[i];
  }
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, param.num_ids) {
    param.out_inverse_unique_partition_indices[i] = param.inverse_unique_partition_indices[i];
  }
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, param.num_unique_matrix_cnt) {
    param.out_num_unique_matrix[i] = param.num_unique_matrix[i];
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    *param.out_cur_rank_num_unique = *param.cur_rank_num_unique;
  }
}

}  // namespace

template<typename K, typename U, typename IDX>
class IdShuffleCopyOutKernel final : public user_op::OpKernel {
 public:
  IdShuffleCopyOutKernel() : current_iter_(0){};
  ~IdShuffleCopyOutKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<DataShuffleKernelState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<DataShuffleKernelState*>(state);
    CHECK(kernel_state != nullptr);
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    embedding::EmbeddingState* embedding_state = kernel_state->EmbeddingState();
    const uint32_t num_unique = embedding_state->GetIdNumUnique(current_iter_);
    const std::vector<uint32_t>& num_unique_matrix_vec =
        embedding_state->GetIdNumUniqueMatrix(current_iter_);
    uint32_t cur_rank_num_ids = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      cur_rank_num_ids += num_unique_matrix_vec.at(i * parallel_num + parallel_id);
    }
    Param<K, U, IDX> param;
    param.final_num_unique_ids = num_unique;
    param.cur_rank_unique_ids =
        reinterpret_cast<const K*>(ctx->Tensor4ArgNameAndIndex("cur_rank_unique_ids", 0)->dptr());
    param.out_cur_rank_unique_ids =
        reinterpret_cast<K*>(ctx->Tensor4ArgNameAndIndex("out_cur_rank_unique_ids", 0)->mut_dptr());
    param.cur_rank_unique_table_ids = reinterpret_cast<const U*>(
        ctx->Tensor4ArgNameAndIndex("cur_rank_unique_table_ids", 0)->dptr());
    param.out_cur_rank_unique_table_ids = reinterpret_cast<U*>(
        ctx->Tensor4ArgNameAndIndex("out_cur_rank_unique_table_ids", 0)->mut_dptr());
    param.cur_rank_num_ids = cur_rank_num_ids;
    param.cur_rank_inverse_indices = reinterpret_cast<const IDX*>(
        ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0)->dptr());
    param.out_cur_rank_inverse_indices = reinterpret_cast<IDX*>(
        ctx->Tensor4ArgNameAndIndex("out_cur_rank_inverse_indices", 0)->mut_dptr());
    param.num_ids =
        ctx->Tensor4ArgNameAndIndex("inverse_unique_partition_indices", 0)->shape_view().elem_cnt();
    param.inverse_unique_partition_indices = reinterpret_cast<const IDX*>(
        ctx->Tensor4ArgNameAndIndex("inverse_unique_partition_indices", 0)->dptr());
    param.out_inverse_unique_partition_indices = reinterpret_cast<IDX*>(
        ctx->Tensor4ArgNameAndIndex("out_inverse_unique_partition_indices", 0)->mut_dptr());
    param.num_unique_matrix_cnt = parallel_num * parallel_num;
    param.num_unique_matrix =
        reinterpret_cast<const IDX*>(ctx->Tensor4ArgNameAndIndex("num_unique_matrix", 0)->dptr());
    param.out_num_unique_matrix =
        reinterpret_cast<IDX*>(ctx->Tensor4ArgNameAndIndex("out_num_unique_matrix", 0)->mut_dptr());
    param.cur_rank_num_unique =
        reinterpret_cast<const IDX*>(ctx->Tensor4ArgNameAndIndex("cur_rank_num_unique", 0)->dptr());
    param.out_cur_rank_num_unique = reinterpret_cast<IDX*>(
        ctx->Tensor4ArgNameAndIndex("out_cur_rank_num_unique", 0)->mut_dptr());

    CopyGpu<K, U, IDX><<<BlocksNum4ThreadsNum(param.num_ids), kCudaThreadsNumPerBlock, 0,
                         ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(param);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

#define ID_DATA_TYPE_SEQ                            \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, DataType::kUInt64) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)   \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define TABLE_ID_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(uint8_t, DataType::kUInt8)

#define IDX_DATA_TYPE_SEQ                           \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

#define REGISTER_CUDA_ID_SHUFFLE_COPY_OUT_KERNEL(k_dtype_pair, table_id_dtype_pair,              \
                                                 idx_dtype_pair)                                 \
  REGISTER_USER_KERNEL("id_shuffle_copy_out")                                                    \
      .SetCreateFn<IdShuffleCopyOutKernel<OF_PP_PAIR_FIRST(k_dtype_pair),                        \
                                          OF_PP_PAIR_FIRST(table_id_dtype_pair),                 \
                                          OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                   \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                        \
          && (user_op::HobDataType("cur_rank_unique_ids", 0) == OF_PP_PAIR_SECOND(k_dtype_pair)) \
          && (user_op::HobDataType("cur_rank_unique_table_ids", 0)                               \
              == OF_PP_PAIR_SECOND(table_id_dtype_pair))                                         \
          && (user_op::HobDataType("num_unique_matrix", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_ID_SHUFFLE_COPY_OUT_KERNEL, ID_DATA_TYPE_SEQ,
                                 TABLE_ID_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

}  // namespace oneflow
