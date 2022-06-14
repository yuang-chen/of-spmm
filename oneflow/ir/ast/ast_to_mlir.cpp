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
#include "oneflow/api/python/graph_mlir/ast_to_mlir_api.h"
#include <cstdint>
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "mlir/IR/Builders.h"

namespace {

using namespace mlir;

static MLIRContext* mlir_ctx = nullptr;
static OwningOpRef<ModuleOp> mlir_module = nullptr;
static OpBuilder* mlir_op_builder = nullptr;
static uint32_t mlir_line = 0;

}  // namespace

namespace oneflow {

FileLineColLoc getNextLoc() {
  return FileLineColLoc::get(::mlir_ctx, "nn_graph", ::mlir_line++, /*column=*/0);
}

void initAstToMLIRContext() {
  ::mlir_ctx = new MLIRContext();
  ::mlir_ctx->getOrLoadDialect<mlir::oneflow::OneFlowDialect>();
  ::mlir_ctx->loadDialect<mlir::LLVM::LLVMDialect>();

  ::mlir_module = ModuleOp::create(getNextLoc());
  ::mlir_op_builder = new OpBuilder(::mlir_ctx);

  emitMLIRConstantInt(1);
}

void finishAstToMLIRContext() {
  std::cout << "Finish AST To MLIR Context" << std::endl;
  ::mlir_module->print(llvm::outs());

  delete ::mlir_op_builder;
  ::mlir_module.release();
  delete ::mlir_ctx;

  ::mlir_op_builder = nullptr;
  ::mlir_module = nullptr;
  ::mlir_ctx = nullptr;
  ::mlir_line = 0;
}

void emitMLIRConstantInt(int value) {
  auto int_type = ::mlir_op_builder->getIntegerType(32);
  auto int_value = ::mlir_op_builder->getIntegerAttr(int_type, value);
  auto int_op = ::mlir_op_builder->create<LLVM::ConstantOp>(getNextLoc(), int_type, int_value);

  ::mlir_module->push_back(int_op);
}

void emitMLIRConstantBool(bool value) {
  auto bool_type = ::mlir_op_builder->getIntegerType(1);
  auto bool_value = ::mlir_op_builder->getBoolAttr(value);
  auto bool_op = ::mlir_op_builder->create<LLVM::ConstantOp>(getNextLoc(), bool_type, bool_value);

  ::mlir_module->push_back(bool_op);
}

void emitMLIRConstantFloat(float value) {
  auto float_type = ::mlir_op_builder->getF32Type();
  auto float_value = ::mlir_op_builder->getF32FloatAttr(value);
  auto float_op =
      ::mlir_op_builder->create<LLVM::ConstantOp>(getNextLoc(), float_type, float_value);

  ::mlir_module->push_back(float_op);
}

}  // namespace oneflow