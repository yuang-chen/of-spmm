/*
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
#include <iostream>
#include <string>
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace oneflow {

namespace {

class OutlineJitFunctionPass : public OutlineJitFunctionPassBase<OutlineJitFunctionPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateFuserPasses(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

class KernelLaunchWithLLVMPass : public KernelLaunchWithLLVMPassBase<KernelLaunchWithLLVMPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateKernelWithLLVMPasses(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

class KernelLaunchFunctionPass : public KernelLaunchFunctionPassBase<KernelLaunchFunctionPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateKernelWrapperPasses(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

class FuseIntoExistingOpPass : public FuseIntoExistingOpPassBase<FuseIntoExistingOpPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateFuserForExistingOp(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<Pass> createOutlineJitFunctionPass() {
  return std::make_unique<OutlineJitFunctionPass>();
}

std::unique_ptr<Pass> createKernelLaunchFunctionPass() {
  return std::make_unique<KernelLaunchFunctionPass>();
}

std::unique_ptr<mlir::Pass> createKernelLaunchWithLLVMPass() {
  return std::make_unique<KernelLaunchWithLLVMPass>();
}

std::unique_ptr<Pass> createFuseIntoExistingOpPass() {
  return std::make_unique<FuseIntoExistingOpPass>();
}

}  // namespace oneflow
}  // namespace mlir
