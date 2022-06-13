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
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <iostream>
#include <string>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/graph_mlir/ast_to_mlir.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace oneflow {

#ifndef WITH_MLIR

Maybe<void> initMLIRContext() {
  UNIMPLEMENTED_THEN_RETURN() << "initMLIRContext is only supported WITH_MLIR";
}

Maybe<void> finishMLIRContext() {
  UNIMPLEMENTED_THEN_RETURN() << "finishMLIRContext is only supported WITH_MLIR";
}

#endif

Maybe<void> GraphAstToMLIR(py::object build_method_ast) {
  py::object ast_module = py::module_::import("ast");
  py::str ast_str = ast_module.attr("dump")(build_method_ast, "indent"_a = 4);
  std::cout << std::string(ast_str) << std::endl;

  py::handle method = build_method_ast.attr("body")[0];

  py::str s = ast_module.attr("dump")(method, "indent"_a = 4);

  std::cout << std::string(s) << std::endl;

  return Maybe<void>::Ok();
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("GraphAstToMLIR", [](const py::object& graph_build_method_ast) -> Maybe<void> {
    return initMLIRContext();
    ;
  });
}

}  // namespace oneflow
