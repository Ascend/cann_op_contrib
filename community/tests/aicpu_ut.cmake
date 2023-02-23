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

project(cpu_kernels_ut)

set(CMAKE_CXX_STANDARD 11)

if(DEFINED ENV{ASCEND_CUSTOM_PATH})
  set(AICPU_OPP_ENV $ENV{ASCEND_CUSTOM_PATH}/opp/built-in/op_impl/aicpu/aicpu_kernel)
else()
  set(AICPU_OPP_ENV /usr/local/Ascend/opp/built-in/op_impl/aicpu/aicpu_kernel)
endif()

set(CMAKE_COMMON_SOURCE_DIR ${CANN_ROOT_DIR}/community/common)

set(_proto_srcs
  "${CMAKE_COMMON_SOURCE_DIR}/proto/cpu_tensor.proto"
  "${CMAKE_COMMON_SOURCE_DIR}/proto/cpu_attr.proto"
  "${CMAKE_COMMON_SOURCE_DIR}/proto/cpu_tensor_shape.proto"
  "${CMAKE_COMMON_SOURCE_DIR}/proto/cpu_node_def.proto"
)

# the output path of generated files

set(_proto_include "${CMAKE_BINARY_DIR}/proto/opp")

protobuf_generate(opp _proto_cc _proto_h ${_proto_srcs})

file(GLOB CPU_KERNELS_DIR_SRC ${CANN_ROOT_DIR}/community/ops/**/aicpu/impl/*.cc)

set(UTILS_DIR ${CMAKE_COMMON_SOURCE_DIR}/utils)

set(_cpu_context
  ${UTILS_DIR}/node_def.cc
  ${UTILS_DIR}/node_def_impl.cc
  ${UTILS_DIR}/tensor.cc
  ${UTILS_DIR}/tensor_impl.cc
  ${UTILS_DIR}/tensor_shape.cc
  ${UTILS_DIR}/tensor_shape_impl.cc
  ${UTILS_DIR}/attr_value.cc
  ${UTILS_DIR}/attr_value_impl.cc
  ${UTILS_DIR}/device.cc
  ${UTILS_DIR}/context.cc
  ${UTILS_DIR}/device_cpu_kernel.cc
  ${UTILS_DIR}/cpu_kernel_register.cc
  ${UTILS_DIR}/cpu_kernel_utils.cc
  ${UTILS_DIR}/host_sharder.cc
  ${UTILS_DIR}/device_sharder.cc
  ${UTILS_DIR}/eigen_threadpool.cc
  ${UTILS_DIR}/cpu_kernel_cache.cc
  ${UTILS_DIR}/async_cpu_kernel.cc
  ${UTILS_DIR}/async_event_util.cc
  ${UTILS_DIR}/aicpu_sharder.cc
  ${UTILS_DIR}/cust_op_log.cc
)

set(_cpu_kernels_src
  ${CPU_KERNELS_DIR_SRC}
  ${CMAKE_COMMON_SOURCE_DIR}/utils/allocator_utils.cc
  ${CMAKE_COMMON_SOURCE_DIR}/utils/bcast.cc
  ${CMAKE_COMMON_SOURCE_DIR}/utils/resource_mgr.cc
  ${CMAKE_COMMON_SOURCE_DIR}/utils/broadcast_iterator.cc
  ${CMAKE_COMMON_SOURCE_DIR}/utils/eigen_tensor.cc
  ${CMAKE_COMMON_SOURCE_DIR}/utils/kernel_util.cc
  ${CMAKE_COMMON_SOURCE_DIR}/utils/range_sampler.cc
  ${CMAKE_COMMON_SOURCE_DIR}/utils/sampling_kernels.cc
  ${CMAKE_COMMON_SOURCE_DIR}/utils/sparse_group.cc
  ${CMAKE_COMMON_SOURCE_DIR}/utils/sparse_tensor.cc
)

set(cpu_kernels_llt_src
  ${_cpu_context}
  ${_cpu_kernels_src}
  ${_proto_cc}
  ${_proto_h}
)

add_library(cpu_kernels_llt STATIC
  ${cpu_kernels_llt_src}
)

set(OPENSDK ${ASCEND_DIR}/opensdk/opensdk)
set(METADEF_INCLUDE ${OPENSDK}/include/metadef)
set(GRAPHENGINE_INCLUDE ${OPENSDK}/include/air)

set(cpu_kernels_llt_include
  ${OPENSDK}/include/air/framework/common
  ${CMAKE_COMMON_SOURCE_DIR}
  ${CMAKE_COMMON_SOURCE_DIR}/inc
  ${AICPU_OPP_ENV}/inc
  ${_proto_include}
  ${OPENSDK}/include/slog
  ${GRAPHENGINE_INCLUDE}
  ${GRAPHENGINE_INCLUDE}/external
)

target_include_directories(cpu_kernels_llt PUBLIC
  ${cpu_kernels_llt_include}
)

target_compile_options(cpu_kernels_llt PUBLIC
  -D_GLIBCXX_USE_CXX11_ABI=0
  -g
  -Dgoogle=ascend_private
  -DRUN_TEST
)

target_link_libraries(cpu_kernels_llt
  PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
  PUBLIC
    ascend_protobuf
    c_sec
    -ldl
    -Wl,--no-as-needed
    register
    Eigen3::Eigen
)

file(GLOB UT_SRC_CC ${CANN_ROOT_DIR}/community/tests/**/ut/aicpu/*.cc
)

file(GLOB UT_SRC_CPP ${CANN_ROOT_DIR}/community/tests/**/ut/aicpu/*.cpp
)

set(_cpu_kernels_ut_files
  ${UT_SRC_CC}
  ${UT_SRC_CPP}
  ${CANN_ROOT_DIR}/community/common/utils/node_def_builder.cpp
  ${CANN_ROOT_DIR}/community/common/utils/aicpu_read_file.cpp
  ${CANN_ROOT_DIR}/community/common/utils/aicpu_test_utils.cpp
)

add_executable(cpu_kernels_ut
  ${_cpu_kernels_ut_files}
)

target_include_directories(cpu_kernels_ut PRIVATE
  ${CANN_ROOT_DIR}/community/common/utils
)

target_link_libraries(cpu_kernels_ut
  PRIVATE
    -Wl,--whole-archive
     cpu_kernels_llt
    -Wl,--no-whole-archive
    -Wl,--no-as-needed
    GTest::gtest
    GTest::gtest_main
)

target_compile_definitions(cpu_kernels_ut PUBLIC
  D_GLIBCXX_USE_CXX11_ABI=0
  google=ascend_private
  RUN_TEST
)

