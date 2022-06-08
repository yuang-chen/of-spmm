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
from oneflow.nn.graph.ast import get_func_source_strs
from oneflow.nn.graph.graph import Graph
import ast
import textwrap

class ScriptedGraph(Graph):
    def __init__(self):
        super().__init__()
        self._build_method_ast = self._get_ast(self._get_build_method_source_str())

    def _get_build_method_source_str(self) -> str:
        sourcelines, file_lineno, file_name = get_func_source_strs(self.build)
        print(sourcelines)
        return ''.join(sourcelines)

    def _get_ast(self, func_str: str):
        return ast.parse(textwrap.dedent(func_str))
