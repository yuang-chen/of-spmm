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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

oneflow::DataType InferGnParamDataType(const DataType x_data_type) {
  return x_data_type == DataType::kFloat16 ? DataType::kFloat : x_data_type;
}

}  // namespace

/* static */ Maybe<void> GroupNormOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* y = ctx->OutputTensorDesc("y", 0);
  user_op::TensorDesc* mean = ctx->OutputTensorDesc("mean", 0);
  user_op::TensorDesc* inv_variance = ctx->OutputTensorDesc("inv_variance", 0);
  const bool affine = ctx->Attr<bool>("affine");
  const int32_t num_groups = ctx->Attr<int32_t>("num_groups"); 
  const int64_t batch_num = x.shape().At(0); 
  const int64_t channel_num = x.shape().At(1); // Assueme channel first
  *y->mut_shape() = x.shape();
  *y->mut_is_dynamic() = x.is_dynamic();
  if(affine){
    const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
    CHECK_EQ_OR_RETURN(gamma.shape().At(0), channel_num); 
    const user_op::TensorDesc& beta = ctx->InputTensorDesc("beta", 0);
    CHECK_EQ_OR_RETURN(beta.shape().At(0), channel_num); 
  }
  CHECK_EQ_OR_RETURN(channel_num % num_groups, 0) << "Channels should be divisble by num_groups. "; 
  *mean->mut_shape() = Shape({batch_num, num_groups});
  *inv_variance = *mean;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> GroupNormOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> GroupNormOp::GetSbp(user_op::SbpContext* ctx) {
  // TODO: Support More SBP 
  ctx->NewBuilder()
    .Split(ctx->inputs(), 0)
    .Split(ctx->outputs(), 0)
    .Broadcast(user_op::OpArg("gamma", 0))
    .Broadcast(user_op::OpArg("beta", 0))
    .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> GroupNormOp::InferDataType(user_op::InferContext* ctx) {
  const bool affine = ctx->Attr<bool>("affine");
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* y = ctx->OutputTensorDesc("y", 0);
  *y->mut_data_type() = x.data_type();
  if (affine) {
    const user_op::TensorDesc& gamma = ctx->InputTensorDesc("gamma", 0);
    CHECK_EQ_OR_RETURN(gamma.data_type(), x.data_type());
    const user_op::TensorDesc& beta = ctx->InputTensorDesc("beta", 0);
    CHECK_EQ_OR_RETURN(beta.data_type(), x.data_type());
  }
  user_op::TensorDesc* mean = ctx->OutputTensorDesc("mean", 0);
  user_op::TensorDesc* inv_variance = ctx->OutputTensorDesc("inv_variance", 0);
  *mean->mut_data_type() = InferGnParamDataType(x.data_type());
  *inv_variance->mut_data_type() = mean->data_type();
  return Maybe<void>::Ok();
}

}  // namespace oneflow