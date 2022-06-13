oneflow_add_mlir_library(ast_to_mlir ast_to_mlir.cpp)
target_link_libraries(ast_to_mlir PRIVATE 
glog::glog
pybind11::headers
${Python3_LIBRARIES}
${oneflow_third_party_libs})
