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

set(_proto_srcs
  "${OPENSDK}/proto/insert_op.proto"
)

# the output path of generated files

protobuf_generate(opp _proto_cc _proto_h ${_proto_srcs})

set(_op_proto_utils
  ${UTILS_DIR}/array_ops_shape_fns.cc
  ${UTILS_DIR}/axis_util.cc
  ${UTILS_DIR}/candidate_sampling_shape_fns.cc
  ${UTILS_DIR}/common_shape_fns.cc
  ${CMAKE_COMMON_SOURCE_DIR}/src/error_util.cc
  ${CMAKE_COMMON_SOURCE_DIR}/src/op_util.cc
  ${UTILS_DIR}/images_ops_shape_fns.cc
  ${UTILS_DIR}/linalg_ops_shape_fns.cc
  ${UTILS_DIR}/lookup_ops_shape_fns.cc
  ${UTILS_DIR}/nn_shape_fns.cc
  ${UTILS_DIR}/ragged_conversion_ops_shape_fns.cc
  ${UTILS_DIR}/random_ops_shape_fns.cc
  ${UTILS_DIR}/resource_variable_ops_shape_fns.cc
  ${UTILS_DIR}/transfer_shape_according_to_format.cc
  ${UTILS_DIR}/util.cc
  ${UTILS_DIR}/reduce_infer_util.cc
)

file(GLOB OP_PROTO_SRC_FILES ${CANN_ROOT_DIR}/community/ops/**/op_proto/*.cc)

file(GLOB OP_PROTO_PATH ${CANN_ROOT_DIR}/community/ops/**/op_proto)
set(_op_proto_srcs  
   ${OP_PROTO_SRC_FILES}
)

set(_op_proto_include
  ${UTILS_DIR}
  ${OP_PROTO_PATH}
  ${METADEF_INCLUDE}
  ${METADEF_INCLUDE}/external
  ${METADEF_INCLUDE}/external/graph
  ${METADEF_INCLUDE}/exe_graph
  ${CMAKE_COMMON_SOURCE_DIR}
  ${CMAKE_COMMON_SOURCE_DIR}/inc
  ${CMAKE_COMMON_SOURCE_DIR}/utils
  ${GRAPHENGINE_INCLUDE}
  ${GRAPHENGINE_INCLUDE}/external
  ${CANN_ROOT_DIR}/build/proto/opp
  ${Protobuf_INCLUDE}
)

set(_op_proto_link_libs
  -Wl,--no-as-needed
    exe_graph
    graph
    graph_base
    register
    alog
    error_manager
    platform
    ascend_protobuf
    static_turbojpeg
  -Wl,--as-needed
    c_sec
    json
)

if(ENABLE_TEST STREQUAL "")
  add_library(opsproto SHARED
    ${_op_proto_srcs}
    ${_op_proto_utils}
    ${_proto_h}
  )
  target_include_directories(opsproto PRIVATE
    ${_op_proto_include}
  )
  if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
   set(compile_opt_mode ${CMAKE_BUILD_MODE})
  else()
   set(compile_opt_mode -O2)
  endif()
  target_compile_options(opsproto PRIVATE
    ${compile_opt_mode}
    -Dgoogle=ascend_private
  )
  target_link_libraries(opsproto PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:slog_headers>>
    ${_op_proto_link_libs}
  )
  if(BUILD_OPEN_PROJECT)
    set(OPS_PROTO_PATH "${INSTALL_PATH}/opp/built-in/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}")
    add_library(opsproto_rt2.0 SHARED
      ${_op_proto_srcs}
      ${_op_proto_utils}
      ${_proto_h}
    )
    target_include_directories(opsproto_rt2.0 PRIVATE
      ${_op_proto_include}
    )
    target_compile_options(opsproto_rt2.0 PRIVATE
      ${compile_opt_mode}
      -Dgoogle=ascend_private
      -fvisibility=hidden
      -DDISABLE_COMPILE_V1
      -fPIC
    )
    target_link_libraries(opsproto_rt2.0 PRIVATE
      $<BUILD_INTERFACE:intf_pub>
      $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:slog_headers>>
      -Wl,--whole-archive
        rt2_registry_static
      -Wl,--no-whole-archive
      ${_op_proto_link_libs}
    )
    cann_install(
      TARGET      opsproto
      FILES       $<TARGET_FILE:opsproto>
      DESTINATION "${OPS_PROTO_PATH}"
    )
    cann_install(
      TARGET      opsproto_rt2.0
      FILES       $<TARGET_FILE:opsproto_rt2.0>
      DESTINATION "${OPS_PROTO_PATH}"
    )
    add_custom_target(copy_op_proto_inc ALL)
    cann_install(
      TARGET      copy_op_proto_inc
      DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/inc
      DESTINATION "${OPS_PROTO_PATH}"
    )
  else()
    add_library(opsproto_rt2.0 SHARED
      ${_op_proto_srcs}
      ${_op_proto_utils}
      ${_proto_h}
    )
    target_include_directories(opsproto_rt2.0 PRIVATE
      ${_op_proto_include}
    )
    target_compile_options(opsproto_rt2.0 PRIVATE
      ${compile_opt_mode}
      -Dgoogle=ascend_private
      -fvisibility=hidden
      -DDISABLE_COMPILE_V1
      -fPIC
    )
    target_link_libraries(opsproto_rt2.0 PRIVATE
      $<BUILD_INTERFACE:intf_pub>
      $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:slog_headers>>
      -Wl,--whole-archive
        rt2_registry_static
      -Wl,--no-whole-archive
      ${_op_proto_link_libs}
    )
    install(
      TARGETS opsproto OPTIONAL
      LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    )
    install(
      TARGETS opsproto OPTIONAL
      EXPORT  opsproto-targets
      LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}
    )
    install(
      TARGETS opsproto_rt2.0 OPTIONAL
      EXPORT  opsproto_rt2.0-targets
      LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}
    )
    if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
      install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/inc
        DESTINATION "${INSTALL_LIBRARY_DIR}/op_proto_inc"
	PATTERN "OWNERS" EXCLUDE
        PATTERN "experiment_ops.h" EXCLUDE
      )
    else()
      install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/inc
        DESTINATION "${INSTALL_LIBRARY_DIR}/op_proto_inc"
        PATTERN "OWNERS" EXCLUDE
      )
    endif()
  endif()
elseif(UT_TEST_ALL OR PROTO_UT OR PASS_UT OR PLUGIN_UT OR ONNX_PLUGIN_UT)
  add_library(opsproto_llt STATIC
    ${_op_proto_srcs}
    ${_op_proto_utils}
    ${_proto_h}
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
endif()

file(GLOB OP_PROTO_TEST_FILES ${CANN_ROOT_DIR}/community/tests/**/ut/op_proto/*.cc)

add_executable(ops_cpp_proto_utest
        ${OP_PROTO_TEST_FILES}
        )
target_include_directories(ops_cpp_proto_utest PRIVATE
        ${GTEST_INCLUDE}
        ${CMAKE_COMMON_SOURCE_DIR}
        ${CMAKE_COMMON_SOURCE_DIR}/inc
        ${CMAKE_COMMON_SOURCE_DIR}/utils
        ${OPENSDK}/c_sec/include
        ${OPENSDK}/include/slog
        ${OPENSDK}/include/air
        ${OPENSDK}/include/air/external
        ${METADEF_INCLUDE}
        ${METADEF_INCLUDE}/graph
        ${METADEF_INCLUDE}/exe_graph
        ${METADEF_INCLUDE}/external
        ${METADEF_INCLUDE}/external/graph
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
        platform
        -Wl,--no-whole-archive
        GTest::gtest
        GTest::gtest_main
        )

set_target_properties(ops_cpp_proto_utest PROPERTIES CXX_STANDARD 17)

