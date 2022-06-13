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
import ast
from fileinput import filename
import inspect
from typing import Callable, List, Optional, Tuple

from pyparsing import Any


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
