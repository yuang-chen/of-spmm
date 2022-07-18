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
#include "oneflow/core/embedding/hash_functions.cuh"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/embedding/embedding_manager.h"
#include "oneflow/core/control/ctrl_client.h"

namespace oneflow {

namespace {

template<typename K>
struct TableEntry {
  K key;
  uint32_t value;
};

template<typename K, typename V, typename IDX, typename HASH>
__global__ void HashTableUniqueAndPartitionPairs(
    const uint32_t table_capacity, const uint32_t num_keys, int32_t num_partition,
    IDX* unique_counts, TableEntry<K>* table, const K* keys, const V* values,
    K* partitioned_unique_keys, V* partitioned_unique_values, IDX* reverse_index,
    bool need_process_values, int32_t* is_kernel_start) {
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, num_keys) {
    IDX r_index_plus_one = 0;
    const K key = keys[i];
    size_t key_hash = HASH()(key);
    uint32_t partition_id = key_hash % num_partition;
    IDX* unique_count = unique_counts + partition_id;
    K* unique_keys = partitioned_unique_keys + partition_id * num_keys;
    uint32_t pos = key_hash % table_capacity;
    const K key_hi = (key | 0x1);
    const K key_lo = (key & 0x1);
    uint32_t counter = 0;
    while (r_index_plus_one == 0) {
      bool prob_next = false;
      K* key_ptr = &table[pos].key;
      volatile uint32_t* table_value_ptr = &table[pos].value;
      const K old_key = cuda::atomic::CAS(key_ptr, 0, key_hi);
      if (old_key == 0) {
        IDX unique_pos = cuda::atomic::Add(unique_count, 1);
        r_index_plus_one = unique_pos + 1;
        unique_keys[unique_pos] = key;
        if (need_process_values) {
          partitioned_unique_values[partition_id * num_keys + unique_pos] = values[i];
        }
        *table_value_ptr = ((r_index_plus_one << 1U) | key_lo);
      } else if (old_key == key_hi) {
        const uint32_t value = *table_value_ptr;
        if (value == 0) {
          // do nothing
        } else if ((value & 0x1) == key_lo) {
          r_index_plus_one = (value >> 1U);
        } else {
          prob_next = true;
        }
      } else {
        prob_next = true;
      }
      if (prob_next) {
        pos += 1;
        counter += 1;
        if (pos >= table_capacity) { pos -= table_capacity; }
        if (counter >= table_capacity) { __trap(); }
      }
    }
    reverse_index[i] = partition_id * num_keys + r_index_plus_one - 1;
  }
}

template<typename K, typename U, typename IDX, int N>
struct Param {
  IDX* num_unique[N];
  K* unique_ids[N];
  U* unique_table_ids[N];
  int32_t* is_kernel_start[N];
  IDX* num_unique_matrix;
  int32_t* counter;
};

template<typename K, typename V, typename IDX, int N>
__global__ void BarrierKernel(int32_t parallel_id, int32_t parallel_num,
                              Param<K, V, IDX, N> param) {
  int count = param.is_kernel_start[parallel_id][parallel_id];
  if (threadIdx.x < parallel_num) {
    volatile int32_t* start_f = param.is_kernel_start[parallel_id];
    volatile int32_t* remote_start_f = param.is_kernel_start[threadIdx.x];
    start_f[threadIdx.x] = count + 1;
    while (remote_start_f[parallel_id] < count + 1) {}
  }
}

template<typename K, typename V, typename IDX, typename HASH, int N>
__global__ void HashTableUniquePairs(const uint32_t table_capacity, const uint32_t num_ids,
                                     int32_t parallel_num, int32_t parallel_id, IDX* unique_count,
                                     TableEntry<K>* table, Param<K, V, IDX, N> param,
                                     K* unique_keys, V* unique_values, IDX* reverse_index,
                                     bool need_process_values) {
#pragma unroll 1
  for (int i = 0; i < parallel_num; ++i) {
    int rank_id = (parallel_id + i) % parallel_num;
    const IDX* num_uniques = param.num_unique[rank_id];
    CUDA_1D_KERNEL_LOOP_T(int, rank_index, num_uniques[parallel_id]) {
      const IDX* num_uniques = param.num_unique[rank_id];
      // if (rank_index >= num_uniques[parallel_id]) { continue; }
      const K* keys = param.unique_ids[rank_id];
      const V* values = param.unique_table_ids[rank_id];
      IDX index_offset = 0;
      for (int k = 0; k < rank_id; ++k) { index_offset += param.num_unique[k][parallel_id]; }
      IDX r_index_plus_one = 0;
      const K key = keys[rank_index];
      size_t key_hash = HASH()(key);
      uint32_t pos = key_hash % table_capacity;
      const K key_hi = (key | 0x1);
      const K key_lo = (key & 0x1);
      uint32_t counter = 0;
      while (r_index_plus_one == 0) {
        bool prob_next = false;
        K* key_ptr = &table[pos].key;
        volatile uint32_t* table_value_ptr = &table[pos].value;
        const K old_key = cuda::atomic::CAS(key_ptr, 0, key_hi);
        if (old_key == 0) {
          IDX unique_pos = cuda::atomic::Add(unique_count, 1);
          r_index_plus_one = unique_pos + 1;
          unique_keys[unique_pos] = key;
          if (need_process_values) { unique_values[unique_pos] = values[rank_index]; }
          *table_value_ptr = ((r_index_plus_one << 1U) | key_lo);
        } else if (old_key == key_hi) {
          const uint32_t value = *table_value_ptr;
          if (value == 0) {
            // do nothing
          } else if ((value & 0x1) == key_lo) {
            r_index_plus_one = (value >> 1U);
          } else {
            prob_next = true;
          }
        } else {
          prob_next = true;
        }
        if (prob_next) {
          pos += 1;
          counter += 1;
          if (pos >= table_capacity) { pos -= table_capacity; }
          if (counter >= table_capacity) { __trap(); }
        }
      }
      reverse_index[rank_index + index_offset] = r_index_plus_one - 1;
      if (rank_index < parallel_num) {
        param.num_unique_matrix[i * parallel_num + rank_index] = param.num_unique[i][rank_index];
      }
    }
  }
}

template<typename U>
__global__ void GenerateTableIds(int32_t elem_cnt, int32_t num_tables, U* table_ids) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { table_ids[i] = i % num_tables; }
}

template<typename K, typename V, typename IDX, typename HASH>
void UniqueAndPartition(cudaStream_t cuda_stream, int64_t num_ids, size_t capacity,
                        int64_t num_partition, const K* ids, const V* table_ids,
                        IDX* num_partitioned_unique_ids_ptr, K* partitioned_unique_ids,
                        V* partitioned_unique_table_ids, IDX* inverse_unique_partition_indices,
                        void* workspace_ptr, size_t workspace_bytes, bool need_process_table_ids,
                        int32_t* is_kernel_start_ptr) {
  size_t table_capacity_bytes = capacity * sizeof(TableEntry<K>);
  CHECK_GE(workspace_bytes, table_capacity_bytes);
  OF_CUDA_CHECK(cudaMemsetAsync(workspace_ptr, 0, table_capacity_bytes, cuda_stream));
  OF_CUDA_CHECK(
      cudaMemsetAsync(num_partitioned_unique_ids_ptr, 0, num_partition * sizeof(IDX), cuda_stream));
  HashTableUniqueAndPartitionPairs<K, V, IDX, HASH>
      <<<BlocksNum4ThreadsNum(num_ids), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
          capacity, num_ids, num_partition, num_partitioned_unique_ids_ptr,
          reinterpret_cast<TableEntry<K>*>(workspace_ptr), ids, table_ids, partitioned_unique_ids,
          partitioned_unique_table_ids, inverse_unique_partition_indices, need_process_table_ids,
          is_kernel_start_ptr);
}

enum class IdShuffleBufferType {
  kNumPartitionedUnique = 0,
  kPartitionedUniqueIds,
  kReceivedIds,
  kTableIds,
  kPartitionedUniqueTableIds,
  kReceivedTableIds,
  kWorkspace,
  kMaxType
};

template<typename K, typename U, typename IDX>
class IdShuffleTmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdShuffleTmpBufferManager);
  IdShuffleTmpBufferManager(void* ptr, const int64_t num_ids, const int64_t parallel_num,
                            bool need_table_ids, bool need_process_table_ids)
      : offset_(0),
        offsets_(static_cast<size_t>(IdShuffleBufferType::kMaxType), -1),
        sizes_(static_cast<size_t>(IdShuffleBufferType::kMaxType)),
        ptr_(ptr) {
    const int64_t num_table_ids = need_process_table_ids ? num_ids : 0;
    const size_t table_ids_bytes = need_table_ids ? num_ids * sizeof(U) : 0;
    AllocBuffer(IdShuffleBufferType::kNumPartitionedUnique, parallel_num * sizeof(IDX));
    size_t partitioned_ids_bytes = parallel_num * num_ids * sizeof(K);
    AllocBuffer(IdShuffleBufferType::kPartitionedUniqueIds, partitioned_ids_bytes);
    AllocBuffer(IdShuffleBufferType::kReceivedIds, partitioned_ids_bytes);
    AllocBuffer(IdShuffleBufferType::kTableIds, table_ids_bytes);
    size_t partitioned_table_ids_bytes = parallel_num * num_table_ids * sizeof(U);
    AllocBuffer(IdShuffleBufferType::kPartitionedUniqueTableIds, partitioned_table_ids_bytes);
    AllocBuffer(IdShuffleBufferType::kReceivedTableIds, partitioned_table_ids_bytes);
    const size_t hash_table_capacity = parallel_num * num_ids;
    AllocBuffer(IdShuffleBufferType::kWorkspace, hash_table_capacity * sizeof(TableEntry<K>));
  }

  template<typename T = void>
  T* Ptr(IdShuffleBufferType type) {
    CHECK(ptr_ != nullptr);
    int64_t offset = offsets_.at(static_cast<size_t>(type));
    CHECK_NE(offset, -1);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + offset);
  }

  int64_t Size(IdShuffleBufferType type) { return sizes_.at(static_cast<size_t>(type)); }

  size_t TotalBufferSize() const { return offset_; }

 private:
  void AllocBuffer(IdShuffleBufferType type, size_t size) {
    const size_t type_id = static_cast<size_t>(type);
    CHECK_EQ(offsets_.at(type_id), -1);
    offsets_.at(type_id) = offset_;
    sizes_.at(type_id) = size;
    offset_ += GetCudaAlignedSize(size);
  }
  size_t offset_;
  std::vector<int64_t> offsets_;
  std::vector<int64_t> sizes_;
  void* ptr_;
};

template<typename K, typename U, typename IDX>
class DataShuffleKernelState final : public user_op::OpKernelState {
 public:
  explicit DataShuffleKernelState(user_op::KernelInitContext* ctx)
      : device_index_(-1),
        stream_name_(EagerNcclCommMgr::kDefaultStreamName),
        parallel_desc_(ctx->parallel_desc()),
        parallel_id_(ctx->parallel_ctx().parallel_id()) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    if (ctx->op_conf().has_stream_name_hint()) { stream_name_ = ctx->op_conf().stream_name_hint(); }
    int64_t parallel_num = parallel_desc_.parallel_num();
    OF_CUDA_CHECK(
        cudaMallocHost(&host_num_unique_matrix_, parallel_num * parallel_num * sizeof(IDX)));
    const std::string& embedding_name = ctx->Attr<std::string>("embedding_name");
    const int64_t parallel_id = parallel_id_;
    embedding_state_ = Singleton<embedding::EmbeddingManager>::Get()->GetEmbeddingState(
        embedding_name, parallel_id);
    const int64_t num_ids = ctx->TensorDesc4ArgNameAndIndex("ids", 0)->shape().elem_cnt();
    num_partitioned_unique_size_ = GetCudaAlignedSize(parallel_num * sizeof(IDX));
    partitioned_unique_ids_size_ = GetCudaAlignedSize(parallel_num * num_ids * sizeof(K));
    partitioned_unique_table_ids_size_ = GetCudaAlignedSize(parallel_num * num_ids * sizeof(U));
    is_kernel_start_size_ = GetCudaAlignedSize(parallel_num * sizeof(int32_t));
    size_t buffer_size = num_partitioned_unique_size_ + partitioned_unique_ids_size_
                         + partitioned_unique_table_ids_size_ + is_kernel_start_size_;
    buffer_ptrs_.resize(parallel_num);
    cudaMalloc(&buffer_ptrs_.at(parallel_id), buffer_size);
    cudaMemset(buffer_ptrs_.at(parallel_id), 0, buffer_size);
    std::string name = ctx->op_name();
    for (int64_t i = 0; i < parallel_num; ++i) {
      std::string key = name + std::to_string(i);
      if (parallel_id == i) {
        cudaIpcMemHandle_t handle;
        OF_CUDA_CHECK(cudaIpcGetMemHandle(&handle, buffer_ptrs_.at(parallel_id)));
        Singleton<CtrlClient>::Get()->PushKV(
            key, std::string(reinterpret_cast<const char*>(&handle), sizeof(cudaIpcMemHandle_t)));
      } else {
        cudaIpcMemHandle_t handle;
        Singleton<CtrlClient>::Get()->PullKV(key, [&handle](const std::string& val) {
          memcpy(&handle, val.data(), sizeof(cudaIpcMemHandle_t));
        });
        OF_CUDA_CHECK(
            cudaIpcOpenMemHandle(&buffer_ptrs_.at(i), handle, cudaIpcMemLazyEnablePeerAccess));
      }
    }
  }
  ~DataShuffleKernelState() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFreeHost(host_num_unique_matrix_));
    OF_CUDA_CHECK(cudaFree(buffer_ptrs_.at(parallel_id_)));
  }

  IDX* HostNumUniqueMatrix() { return host_num_unique_matrix_; }

  embedding::EmbeddingState* EmbeddingState() { return embedding_state_; }

  IDX* NumPartitionedUnique(int64_t parallel_id) {
    return reinterpret_cast<IDX*>(buffer_ptrs_.at(parallel_id));
  }

  K* PartitionedUniqueIds(int64_t parallel_id) {
    return reinterpret_cast<K*>(reinterpret_cast<char*>(buffer_ptrs_.at(parallel_id))
                                + num_partitioned_unique_size_);
  }

  U* PartitionedUniqueTableIds(int64_t parallel_id) {
    return reinterpret_cast<U*>(reinterpret_cast<char*>(buffer_ptrs_.at(parallel_id))
                                + num_partitioned_unique_size_ + partitioned_unique_ids_size_);
  }

  int32_t* IsKernelStart(int64_t parallel_id) {
    return reinterpret_cast<int32_t*>(reinterpret_cast<char*>(buffer_ptrs_.at(parallel_id))
                                      + num_partitioned_unique_size_ + partitioned_unique_ids_size_
                                      + partitioned_unique_table_ids_size_);
  }

 private:
  int device_index_;
  bool has_independent_stream_;
  std::string stream_name_;
  ParallelDesc parallel_desc_;
  int64_t parallel_id_;
  IDX* host_num_unique_matrix_;
  std::vector<void*> buffer_ptrs_;
  size_t num_partitioned_unique_size_;
  size_t partitioned_unique_ids_size_;
  size_t partitioned_unique_table_ids_size_;
  size_t is_kernel_start_size_;
  embedding::EmbeddingState* embedding_state_;
};

template<typename IDX>
__global__ void ComputeOffset(int32_t n, IDX* value) {
  IDX sum = 0;
  for (int i = 0; i < n; ++i) {
    IDX count = value[i];
    value[i] = sum;
    sum += count;
  }
}

template<typename IDX>
__global__ void ContiguousInverseUniquePartitionIndices(const int32_t num_ids, IDX* indices_offset,
                                                        IDX* inverse_ptr) {
  CUDA_1D_KERNEL_LOOP(i, num_ids) {
    int inverse_indice = inverse_ptr[i];
    int partition_id = inverse_indice / num_ids;
    int partition_indice = inverse_indice - partition_id * num_ids;
    int new_offset = indices_offset[partition_id];
    inverse_ptr[i] = new_offset + partition_indice;
  }
}
}  // namespace

template<typename K, typename U, typename IDX>
class IdShuffleP2PKernel final : public user_op::OpKernel {
 public:
  IdShuffleP2PKernel() : current_iter_(0){};
  ~IdShuffleP2PKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<DataShuffleKernelState<K, U, IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<DataShuffleKernelState<K, U, IDX>*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* ids = ctx->Tensor4ArgNameAndIndex("ids", 0);
    user_op::Tensor* num_unique_matrix = ctx->Tensor4ArgNameAndIndex("num_unique_matrix", 0);
    user_op::Tensor* inverse_unique_partition_indices =
        ctx->Tensor4ArgNameAndIndex("inverse_unique_partition_indices", 0);
    user_op::Tensor* cur_rank_num_unique = ctx->Tensor4ArgNameAndIndex("cur_rank_num_unique", 0);
    user_op::Tensor* cur_rank_unique_ids = ctx->Tensor4ArgNameAndIndex("cur_rank_unique_ids", 0);
    user_op::Tensor* cur_rank_unique_table_ids =
        ctx->Tensor4ArgNameAndIndex("cur_rank_unique_table_ids", 0);
    user_op::Tensor* cur_rank_inverse_indices =
        ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int32_t num_tables = ctx->Attr<int32_t>("num_tables");
    const bool has_table_ids = ctx->has_input("table_ids", 0);
    const bool need_gen_table_ids = (!has_table_ids && num_tables > 1);
    const bool need_process_table_ids = (has_table_ids || num_tables > 1);
    const int64_t num_ids = ids->shape_view().elem_cnt();
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    IdShuffleTmpBufferManager<K, U, IDX> buffer_manager(
        tmp_buffer->mut_dptr(), num_ids, parallel_num, need_gen_table_ids, need_process_table_ids);
    CHECK_GE(tmp_buffer->shape_view().elem_cnt(), buffer_manager.TotalBufferSize());

    const U* table_ids_ptr;
    if (has_table_ids) {
      const user_op::Tensor* table_ids = ctx->Tensor4ArgNameAndIndex("table_ids", 0);
      table_ids_ptr = reinterpret_cast<const U*>(table_ids->dptr());
    } else if (need_gen_table_ids) {
      GenerateTableIds<<<BlocksNum4ThreadsNum(num_ids), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
          num_ids, num_tables, buffer_manager.template Ptr<U>(IdShuffleBufferType::kTableIds));
      table_ids_ptr = buffer_manager.template Ptr<U>(IdShuffleBufferType::kTableIds);
    } else {
      table_ids_ptr = nullptr;
    }
    IDX* num_partitioned_unique = kernel_state->NumPartitionedUnique(parallel_id);
    K* partitioned_unique_ids = kernel_state->PartitionedUniqueIds(parallel_id);
    U* partitioned_unique_table_ids = kernel_state->PartitionedUniqueTableIds(parallel_id);
    IDX* num_unique_matrix_ptr = reinterpret_cast<IDX*>(num_unique_matrix->mut_dptr());
    size_t hash_table_capacity = parallel_num * num_ids;
    void* workspace_ptr = buffer_manager.Ptr(IdShuffleBufferType::kWorkspace);
    size_t workspace_size = buffer_manager.Size(IdShuffleBufferType::kWorkspace);
    Param<K, U, IDX, 8> param;
    CHECK_LE(parallel_num, 8);
    for (int i = 0; i < parallel_num; ++i) {
      param.num_unique[i] = kernel_state->NumPartitionedUnique(i);
      param.unique_ids[i] = kernel_state->PartitionedUniqueIds(i) + parallel_id * num_ids;
      param.unique_table_ids[i] =
          kernel_state->PartitionedUniqueTableIds(i) + parallel_id * num_ids;
      param.is_kernel_start[i] = kernel_state->IsKernelStart(i);
    }
    UniqueAndPartition<K, U, IDX, embedding::ShardingHash>(
        cuda_stream, num_ids, hash_table_capacity, parallel_num,
        reinterpret_cast<const K*>(ids->dptr()), table_ids_ptr, num_partitioned_unique,
        partitioned_unique_ids, partitioned_unique_table_ids,
        reinterpret_cast<IDX*>(inverse_unique_partition_indices->mut_dptr()), workspace_ptr,
        workspace_size, need_process_table_ids, kernel_state->IsKernelStart(parallel_id));

    OF_CUDA_CHECK(cudaMemsetAsync(workspace_ptr, 0, workspace_size, cuda_stream));
    IDX* cur_rank_num_unique_ids_ptr = reinterpret_cast<IDX*>(cur_rank_num_unique->mut_dptr());
    OF_CUDA_CHECK(cudaMemsetAsync(cur_rank_num_unique_ids_ptr, 0, sizeof(IDX), cuda_stream));

    param.num_unique_matrix = num_unique_matrix_ptr;
    BarrierKernel<<<1, parallel_num, 0, cuda_stream>>>(parallel_id, parallel_num, param);
    const int num_blocks =
        2 * ctx->stream()->As<ep::CudaStream>()->device_properties().multiProcessorCount;
    HashTableUniquePairs<K, U, IDX, embedding::LocalUniqueHash>
        <<<num_blocks, 1024, 0, cuda_stream>>>(
            hash_table_capacity, num_ids, parallel_num, parallel_id, cur_rank_num_unique_ids_ptr,
            reinterpret_cast<TableEntry<K>*>(workspace_ptr), param,
            reinterpret_cast<K*>(cur_rank_unique_ids->mut_dptr()),
            reinterpret_cast<U*>(cur_rank_unique_table_ids->mut_dptr()),
            reinterpret_cast<IDX*>(cur_rank_inverse_indices->mut_dptr()), need_process_table_ids);
    BarrierKernel<<<1, parallel_num, 0, cuda_stream>>>(parallel_id, parallel_num, param);
    if (parallel_num > 1) {
      // use num_partitioned_unique as indices_offset buffer, so should after ncclAllGather.
      ComputeOffset<<<1, 1, 0, cuda_stream>>>(parallel_num, num_partitioned_unique);
      ContiguousInverseUniquePartitionIndices<<<BlocksNum4ThreadsNum(num_ids),
                                                kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
          num_ids, num_partitioned_unique,
          reinterpret_cast<IDX*>(inverse_unique_partition_indices->mut_dptr()));
    }
    // if (!need_process_table_ids) {
    //  OF_CUDA_CHECK(cudaMemsetAsync(cur_rank_unique_table_ids->mut_dptr(), 0,
    //                                received_elem_cnt * sizeof(U), cuda_stream));
    //}
    embedding::EmbeddingState* embedding_state = kernel_state->EmbeddingState();
    std::vector<uint32_t> num_unique_matrix_vec(parallel_num * parallel_num);
    IDX* host_num_unique_matrix = kernel_state->HostNumUniqueMatrix();
    OF_CUDA_CHECK(cudaMemcpyAsync(host_num_unique_matrix, num_unique_matrix_ptr,
                                  parallel_num * parallel_num * sizeof(IDX), cudaMemcpyDefault,
                                  cuda_stream));
    CHECK_JUST(ctx->stream()->Sync());
    std::memcpy(num_unique_matrix_vec.data(), host_num_unique_matrix,
                parallel_num * parallel_num * sizeof(IDX));
    CHECK_EQ(sizeof(IDX), sizeof(uint32_t)) << "assume sizeof(IDX) equals to sizeof(uint32_t)";
    embedding_state->SetIdNumUniqueMatrix(num_unique_matrix_vec, current_iter_);
    // reuse HostNumUniqueMatrix ptr
    IDX* host_num_unique = host_num_unique_matrix;
    OF_CUDA_CHECK(cudaMemcpyAsync(host_num_unique, cur_rank_num_unique->dptr(), sizeof(IDX),
                                  cudaMemcpyDefault, cuda_stream));
    CHECK_JUST(ctx->stream()->Sync());
    uint32_t final_num_unique = *host_num_unique;
    embedding_state->SetIdFinalNumUnique(final_num_unique, current_iter_);
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
//OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  //OF_PP_MAKE_TUPLE_SEQ(uint64_t, DataType::kUInt64) \
  //OF_PP_MAKE_TUPLE_SEQ(int8_t, DataType::kInt8)     \
  //OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)   \
  //OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define IDX_DATA_TYPE_SEQ                           \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

#define REGISTER_CUDA_ID_SHUFFLE_P2P_KERNEL(k_dtype_pair, table_id_dtype_pair, idx_dtype_pair)   \
  REGISTER_USER_KERNEL("id_shuffle")                                                             \
      .SetCreateFn<IdShuffleP2PKernel<OF_PP_PAIR_FIRST(k_dtype_pair),                            \
                                      OF_PP_PAIR_FIRST(table_id_dtype_pair),                     \
                                      OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                       \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                        \
          && (user_op::HobDataType("ids", 0) == OF_PP_PAIR_SECOND(k_dtype_pair))                 \
          && (user_op::HobDataType("cur_rank_unique_table_ids", 0)                               \
              == OF_PP_PAIR_SECOND(table_id_dtype_pair))                                         \
          && (user_op::HobDataType("num_unique_matrix", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair)) \
          && ParseBooleanFromEnv("USE_P2P_KERNEL", false))                                       \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const user_op::TensorDesc& ids = ctx->InputTensorDesc("ids", 0);                         \
        const bool has_table_ids = ctx->has_input("table_ids", 0);                               \
        const int32_t num_tables = ctx->Attr<int32_t>("num_tables");                             \
        const bool need_gen_table_ids = (!has_table_ids && num_tables > 1);                      \
        const bool need_process_table_ids = (has_table_ids || num_tables > 1);                   \
        IdShuffleTmpBufferManager<OF_PP_PAIR_FIRST(k_dtype_pair),                                \
                                  OF_PP_PAIR_FIRST(table_id_dtype_pair),                         \
                                  OF_PP_PAIR_FIRST(idx_dtype_pair)>                              \
            buffer_manager(nullptr, ids.shape().elem_cnt(), ctx->parallel_desc().parallel_num(), \
                           need_gen_table_ids, need_process_table_ids);                          \
        return buffer_manager.TotalBufferSize();                                                 \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_ID_SHUFFLE_P2P_KERNEL, ID_DATA_TYPE_SEQ,
                                 TABLE_ID_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

}  // namespace oneflow
