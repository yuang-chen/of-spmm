"""
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
"""
from typing import List
from oneflow.nn.graph.graph import Graph
import ast
import textwrap
import oneflow._oneflow_internal
from fileinput import filename
import inspect
from typing import Callable, List, Optional, Tuple
from pyparsing import Any
import mlir 


def get_func_source_strs(func: Callable) -> Tuple[List[str], int, Optional[str]]:
    file_name = None
    try:
        file_name = inspect.getsourcefile(func)
        sourcelines, file_lineno = inspect.getsourcelines(func)
    except OSError as e:
        raise OSError(f"Can't get source for {func}.") from e

    return sourcelines, file_lineno, file_name


class GraphASTVisitor(ast.NodeVisitor):
    def visit_Assign(self, node: ast.Assign) -> Any:
        print(type(node).__name__)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        print("Constant", node.value)
        self.generic_visit(node)


class ScriptedGraph(Graph):
    def __init__(self):
        super().__init__()
        self._build_method_ast = self._get_ast(self._get_build_method_source_str())
        # print(ast.dump(self._build_method_ast, indent=4))
        # visitor = GraphASTVisitor()
        # visitor.visit(self._build_method_ast)
        # oneflow._oneflow_internal.GraphAstToMLIR(self._build_method_ast)
        # oneflow._oneflow_internal.finish()

    def _get_build_method_source_str(self) -> str:
        sourcelines, file_lineno, file_name = get_func_source_strs(self.build)
        return "".join(sourcelines)

    def _get_ast(self, func_str: str):
        return ast.parse(textwrap.dedent(func_str))
