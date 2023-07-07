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
set(METADEF_INCLUDE ${OPENSDK_DIR}/include/metadef)
set(GRAPHENGINE_INCLUDE ${OPENSDK_DIR}/include/air)
set(UTILS_DIR ${CMAKE_COMMON_SOURCE_DIR}/utils)

file(GLOB OP_PROTO_SRC_FILES ${CANN_ROOT_DIR}/community/ops/**/ai_core/op_tiling/*.cc)
set(_optiling_files
  ${OP_PROTO_SRC_FILES}
  ${UTILS_DIR}/tiling_util.cc
  ${UTILS_DIR}/fp16_t.cc
)

set(_op_tiling_include
  ${CMAKE_COMMON_SOURCE_DIR}
  ${METADEF_INCLUDE}/external/graph
  ${CMAKE_COMMON_SOURCE_DIR}/inc
  ${GRAPHENGINE_INCLUDE}
  ${GRAPHENGINE_INCLUDE}/external
)
set(_op_tiling_link_libs
    -Wl,--no-as-needed
    graph
    graph_base
    register
    alog
    error_manager
    -Wl,--as-needed
    c_sec
    json
)
  
add_library(optiling_llt STATIC ${_optiling_files})
target_include_directories(optiling_llt PUBLIC
  ${_op_tiling_include}
)
if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
  set(compile_opt_mode ${CMAKE_BUILD_MODE})
else()
  set(compile_opt_mode -O0)
endif()
target_compile_options(optiling_llt PUBLIC
  ${compile_opt_mode}
  -Dgoogle=ascend_private
)
target_compile_definitions(optiling_llt PRIVATE
  ASCEND_OPTILING_UT
)
target_link_libraries(optiling_llt
  PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
  PUBLIC
    ${_op_tiling_link_libs}
    platform
)

file(GLOB OP_TILING_TC_FILES ${CANN_ROOT_DIR}/community/tests/**/ut/tiling/*.cc)

add_executable(ops_cpp_op_tiling_utest
        ${OP_TILING_TC_FILES}
        )

target_compile_definitions(ops_cpp_op_tiling_utest PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
        $<$<STREQUAL:${BUILD_OPEN_PROJECT},TRUE>:ONLY_COMPILE_OPEN_SRC>
        )
if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
   set(compile_opt_mode ${CMAKE_BUILD_MODE})
else()
   set(compile_opt_mode -g)
endif()
target_compile_options(ops_cpp_op_tiling_utest PUBLIC
        ${compile_opt_mode}
        -Dgoogle=ascend_private
        )
target_link_libraries(ops_cpp_op_tiling_utest PRIVATE
        -Wl,--whole-archive
        optiling_llt
        -Wl,--no-whole-archive
        GTest::gtest
        GTest::gtest_main
        )

