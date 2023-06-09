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

set(PROTO "${METADEF_DIR}/proto/insert_op.proto")
protobuf_generate(opp _proto_cc _proto_h ${PROTO})

file(GLOB PROTO_SRC ./**/op_proto/*.cc)

set(PROTO_INC
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/exe_graph
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
    ${METADEF_DIR}/third_party/fwkacllib/inc/toolchain
    ${CANN_ROOT_DIR}/community/common
    ${CANN_ROOT_DIR}/community/common/inc
    #${Protobuf_INCLUDE}
    ${C_SEC_INCLUDE}
)

set(PROTO_LINK_LIB
    -Wl,--no-as-needed
    #exe_graph
    #graph
    #register
    #alog
    #error_manager
    #ascend_protobuf
    -Wl,--as-needed
    #c_sec
)

add_library(op_proto SHARED
    ${PROTO_SRC}
    ${_proto_h}
)

target_include_directories(op_proto PRIVATE
    ${PROTO_INC}
)

target_link_libraries(op_proto PRIVATE
    ${PROTO_LINK_LIB}
)

target_compile_options(op_proto PRIVATE
    ${compile_opt_mode}
    -Dgoogle=ascend_private
)

set(PROTO_PATH "${INSTALL_DIR}/community/op_proto/")
cann_install(
    TARGET      op_proto
    FILES       $<TARGET_FILE:op_proto>
    DESTINATION "${PROTO_PATH}"
)

add_custom_target(copy_proto_inc ALL)
cann_install(
    TARGET      copy_proto_inc
    DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/add/op_proto/inc
    DESTINATION "${PROTO_PATH}"
)

if(ALL_UT OR PROTO_UT)
    add_library(op_proto_llt STATIC
        ${PROTO_SRC}
        ${_proto_h}
    )
    if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
        set(compile_opt_mode ${CMAKE_BUILD_MODE})
    else()
        set(compile_opt_mode -O0)
    endif()
    target_include_directories(op_proto_llt PUBLIC
        ${PROTO_INC}
    )
    target_compile_options(op_proto_llt PUBLIC
        ${compile_opt_mode}
        -Dgoogle=ascend_private
    )
    target_link_libraries(op_proto_llt
        PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub>
        PUBLIC
        ${PROTO_LINK_LIB}
    )
endif()
# -----------------------------------------------------------------------
file(GLOB TILING_SRC ./**/ai_core/op_tiling/*.cc)

set(TILING_LINK_LIB
    -Wl,--no-as-needed
    #graph
    #exe_graph
    #register
    #alog
    #error_manager
    #ascend_protobuf
    -Wl,--as-needed
    #c_sec
)

set(TILING_INC
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
    ${C_SEC_INCLUDE}
    #${Protobuf_INCLUDE}
)

add_library(op_tiling SHARED
    ${TILING_SRC}
)

target_include_directories(op_tiling PRIVATE
    ${TILING_INC}
)

target_link_libraries(op_tiling PRIVATE
    ${TILING_LINK_LIB}
)

target_compile_options(op_tiling PRIVATE
    ${compile_opt_mode}
    -Dgoogle=ascend_private
    "-fno-strict-aliasing"
    "-fno-common"
)

set(TILING_PATH "${INSTALL_DIR}/community/op_tiling")
cann_install(
    TARGET      op_tiling
    FILES       $<TARGET_FILE:op_tiling>
    DESTINATION "${TILING_PATH}"
)


# -----------------------------------------------------------------------

file(GLOB AICPU_SRC ./**/aicpu/impl/*.cc)

set(AICPU_LINK_LIB
    -Wl,--no-as-needed
    #graph
    #exe_graph
    #register
    #alog
    #error_manager
    #ascend_protobuf
    -Wl,--as-needed
    #c_sec
)

set(AICPU_OPP_ENV $ENV{ASCEND_AICPU_PATH}/opp/op_impl/built-in/aicpu/aicpu_kernel)
set(AICPU_INCLUDE ${AICPU_OPP_ENV}/inc)

set(AICPU_INC
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
    ${CANN_ROOT_DIR}/community/common
    ${CANN_ROOT_DIR}/community/common/inc
    ${AICPU_INCLUDE}
    ${EIGEN_INCLUDE}
    ${C_SEC_INCLUDE}
    #${Protobuf_INCLUDE}
)

add_library(cpu_kernels SHARED
    ${AICPU_SRC}
)

target_include_directories(cpu_kernels PRIVATE
    ${AICPU_INC}
)

target_link_libraries(cpu_kernels PRIVATE
    ${AICPU_LINK_LIB}
)

target_compile_options(cpu_kernels PRIVATE
    ${compile_opt_mode}
    -Dgoogle=ascend_private
    "-fno-strict-aliasing"
    "-fno-common"
)


set(AICPU_PATH "${INSTALL_DIR}/community/aicpu/lib")
cann_install(
    TARGET      cpu_kernels
    FILES       $<TARGET_FILE:cpu_kernels>
    DESTINATION "${AICPU_PATH}"
)