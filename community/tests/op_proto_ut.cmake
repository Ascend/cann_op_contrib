# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
set(CMAKE_COMMON_SOURCE_DIR ${CANN_ROOT_DIR}/community/common)
set(OPENSDK ${ASCEND_DIR}/opensdk/opensdk)
set(METADEF_INCLUDE ${OPENSDK}/include/metadef)
set(GRAPHENGINE_INCLUDE ${OPENSDK}/include/air)
set(UTILS_DIR ${CMAKE_COMMON_SOURCE_DIR}/utils)

set(_op_proto_utils
  ${UTILS_DIR}/array_ops_shape_fns.cc
  ${UTILS_DIR}/axis_util.cc
  ${UTILS_DIR}/common_shape_fns.cc
  ${UTILS_DIR}/linalg_ops_shape_fns.cc
  ${CMAKE_COMMON_SOURCE_DIR}/src/error_util.cc
  ${UTILS_DIR}/images_ops_shape_fns.cc
  ${UTILS_DIR}/ragged_conversion_ops_shape_fns.cc
  ${UTILS_DIR}/random_ops_shape_fns.cc
  ${UTILS_DIR}/reduce_infer_util.cc
  ${UTILS_DIR}/transfer_shape_according_to_format.cc
  ${UTILS_DIR}/util.cc
)

file(GLOB OP_PROTO_SRC_FILES ${CANN_ROOT_DIR}/community/ops/**/op_proto/*.cc)

file(GLOB OP_PROTO_PATH ${CANN_ROOT_DIR}/community/ops/**/op_proto/inc)
set(_op_proto_srcs  
   ${OP_PROTO_SRC_FILES}
)

set(_op_proto_include
  ${OP_PROTO_PATH}
  ${CMAKE_COMMON_SOURCE_DIR}
  ${CMAKE_COMMON_SOURCE_DIR}/inc
  ${METADEF_INCLUDE}/exe_graph
  ${GRAPHENGINE_INCLUDE}
  ${GRAPHENGINE_INCLUDE}/external
)

set(_op_proto_link_libs
  -Wl,--no-as-needed
    exe_graph
    graph
    graph_base
    register
    alog
    error_manager
  -Wl,--as-needed
    c_sec
)

add_library(opsproto_llt STATIC
  ${_op_proto_srcs}
  ${_op_proto_utils}
)
if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
  set(compile_opt_mode ${CMAKE_BUILD_MODE})
else()
  set(compile_opt_mode -O0)
endif()
target_include_directories(opsproto_llt PUBLIC
  ${_op_proto_include}
)
target_compile_options(opsproto_llt PUBLIC
  ${compile_opt_mode}
  -Dgoogle=ascend_private
)
target_link_libraries(opsproto_llt
  PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
  PUBLIC
    ${_op_proto_link_libs}
)

file(GLOB OP_PROTO_TEST_FILES ${CANN_ROOT_DIR}/community/tests/**/ut/op_proto/*.cpp)

add_executable(ops_cpp_proto_utest
        ${OP_PROTO_TEST_FILES}
        ${UTILS_DIR}/op_proto_test_util.cpp
        )

target_include_directories(ops_cpp_proto_utest PRIVATE
  ${UTILS_DIR}
)

target_compile_definitions(ops_cpp_proto_utest PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
        $<$<STREQUAL:${BUILD_OPEN_PROJECT},TRUE>:ONLY_COMPILE_OPEN_SRC>
        )
if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
   set(compile_opt_mode ${CMAKE_BUILD_MODE})
else()
   set(compile_opt_mode -g)
endif()
target_compile_options(ops_cpp_proto_utest PUBLIC
        -fPIC
        ${compile_opt_mode}
        -Dgoogle=ascend_private
        )
target_link_libraries(ops_cpp_proto_utest PRIVATE
        -Wl,--whole-archive
        opsproto_llt
        -Wl,--no-whole-archive
        GTest::gtest
        GTest::gtest_main
        )

set_target_properties(ops_cpp_proto_utest PROPERTIES CXX_STANDARD 17)

