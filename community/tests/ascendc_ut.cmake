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

project(ascendc_ut)

set(CMAKE_CXX_STANDARD 17)

file(GLOB ASCENDC_OPP_KERNEL_SRC ${CANN_ROOT_DIR}/community/ops/**/ai_core/op_kernel/*.cpp)

add_library(ascendc_llt STATIC
  ${ASCENDC_OPP_KERNEL_SRC}
)

target_include_directories(ascendc_llt PUBLIC
  ${ASCEND_DIR}/compiler/tikcpp
)

target_compile_options(ascendc_llt PUBLIC
  -D_GLIBCXX_USE_CXX11_ABI=0
  -g
  -Dgoogle=ascend_private
  -DRUN_TEST
)

list(APPEND CMAKE_PREFIX_PATH ${ASCEND_DIR}/tools/tikicpulib/lib/cmake)
find_package(tikicpulib REQUIRED)

target_link_libraries(ascendc_llt
  PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
  PUBLIC
    -ldl
    -Wl,--no-as-needed
    tikicpulib::${product_type}
)

file(GLOB UT_SRC_CC ${CANN_ROOT_DIR}/community/tests/**/ut/ascendc/*.cc
)

set(_ascendc_ut_files
  ${UT_SRC_CC}
  ${CANN_ROOT_DIR}/community/common/utils/ascendc_ut_util.cc
)

add_executable(ascendc_ut
  ${_ascendc_ut_files}
)

add_definitions(-DASCENDC_UT="ASCENDC_UT")

target_include_directories(ascendc_ut PRIVATE
  ${CANN_ROOT_DIR}/community/common/utils
  ${ASCEND_DIR}/include
  ${ASCEND_DIR}/tools/tikicpulib/lib/include
)

target_link_libraries(ascendc_ut
  PRIVATE
    -Wl,--whole-archive
     ascendc_llt
    -Wl,--no-whole-archive
    -Wl,--no-as-needed
    GTest::gtest
    GTest::gtest_main
)

target_compile_definitions(ascendc_ut PUBLIC
  D_GLIBCXX_USE_CXX11_ABI=0
  google=ascend_private
  RUN_TEST
)

