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

project(tik2_ut)

set(CMAKE_CXX_STANDARD 17)

file(GLOB TIK2_OPP_KERNEL_SRC ${CANN_ROOT_DIR}/community/ops/**/ai_core/op_kernel/*.cpp)

add_library(tik2_llt STATIC
  ${TIK2_OPP_KERNEL_SRC}
)

target_include_directories(tik2_llt PUBLIC
  ${ASCEND_DIR}/compiler/tikcpp
)

target_compile_options(tik2_llt PUBLIC
  -D_GLIBCXX_USE_CXX11_ABI=0
  -g
  -Dgoogle=ascend_private
  -DRUN_TEST
)

list(APPEND CMAKE_PREFIX_PATH ${ASCEND_DIR}/tools/tikicpulib/lib/cmake)
find_package(tikicpulib REQUIRED)

target_link_libraries(tik2_llt
  PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
  PUBLIC
    -ldl
    -Wl,--no-as-needed
    tikicpulib::${product_type}
)

file(GLOB UT_SRC_CC ${CANN_ROOT_DIR}/community/tests/**/ut/tik2/*.cc
)

set(_tik2_ut_files
  ${UT_SRC_CC}
  ${CANN_ROOT_DIR}/community/common/utils/tik2_ut_util.cc
)

add_executable(tik2_ut
  ${_tik2_ut_files}
)

add_definitions(-DTIK2_UT="TIK2_UT")

target_include_directories(tik2_ut PRIVATE
  ${CANN_ROOT_DIR}/community/common/utils
  ${ASCEND_DIR}/include
  ${ASCEND_DIR}/tools/tikicpulib/lib/include
)

target_link_libraries(tik2_ut
  PRIVATE
    -Wl,--whole-archive
     tik2_llt
    -Wl,--no-whole-archive
    -Wl,--no-as-needed
    GTest::gtest
    GTest::gtest_main
)

target_compile_definitions(tik2_ut PUBLIC
  D_GLIBCXX_USE_CXX11_ABI=0
  google=ascend_private
  RUN_TEST
)

