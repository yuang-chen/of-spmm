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
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ Maybe<void> IdShuffleCopyOutOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(ctx->inputs(), 0)
      .Split(ctx->outputs(), 0)
      .Broadcast(user_op::OpArg("num_unique_matrix", 0))
      .Broadcast(user_op::OpArg("out_num_unique_matrix", 0))
      .Broadcast(user_op::OpArg("cur_rank_num_unique", 0))
      .Broadcast(user_op::OpArg("out_cur_rank_num_unique", 0))
      .Build();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> IdShuffleCopyOutOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("out_num_unique_matrix", 0) = ctx->InputShape("num_unique_matrix", 0);
  *ctx->OutputShape("out_inverse_unique_partition_indices", 0) =
      ctx->InputShape("inverse_unique_partition_indices", 0);
  *ctx->OutputShape("out_cur_rank_num_unique", 0) = ctx->InputShape("cur_rank_num_unique", 0);
  *ctx->OutputShape("out_cur_rank_unique_ids", 0) = ctx->InputShape("cur_rank_unique_ids", 0);
  *ctx->OutputShape("out_cur_rank_unique_table_ids", 0) =
      ctx->InputShape("cur_rank_unique_table_ids", 0);
  *ctx->OutputShape("out_cur_rank_inverse_indices", 0) =
      ctx->InputShape("cur_rank_inverse_indices", 0);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> IdShuffleCopyOutOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> IdShuffleCopyOutOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out_num_unique_matrix", 0) = ctx->InputDType("num_unique_matrix", 0);
  *ctx->OutputDType("out_inverse_unique_partition_indices", 0) =
      ctx->InputDType("inverse_unique_partition_indices", 0);
  *ctx->OutputDType("out_cur_rank_num_unique", 0) = ctx->InputDType("cur_rank_num_unique", 0);
  *ctx->OutputDType("out_cur_rank_unique_ids", 0) = ctx->InputDType("cur_rank_unique_ids", 0);
  *ctx->OutputDType("out_cur_rank_unique_table_ids", 0) =
      ctx->InputDType("cur_rank_unique_table_ids", 0);
  *ctx->OutputDType("out_cur_rank_inverse_indices", 0) =
      ctx->InputDType("cur_rank_inverse_indices", 0);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
