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

file(GLOB PROTO_SRC
    ./**/op_proto/*.cc
    #${CANN_ROOT_DIR}/community/common/src/*.cc
    #${CANN_ROOT_DIR}/community/common/utils/*.cc
)

set(PROTO_INC
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/exe_graph
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
    ${METADEF_DIR}/third_party/fwkacllib/inc/toolchain
    ${CANN_ROOT_DIR}/community/common/inc
    ${CANN_ROOT_DIR}/community/common
    #${CANN_ROOT_DIR}/community/common/src
    #${Protobuf_INCLUDE}
    ${C_SEC_INCLUDE}
)

set(PROTO_LINK_LIB
    -Wl,--no-as-needed
    exe_graph
    graph
    register
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

add_dependencies(op_proto secure_c external_protobuf external_gtest)

target_include_directories(op_proto PRIVATE
    ${PROTO_INC}
)

target_link_libraries(op_proto PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ${PROTO_LINK_LIB}
)

target_compile_options(op_proto PRIVATE
    ${compile_opt_mode}
    -Dgoogle=ascend_private
)

set(PROTO_INSTALL_PATH "${INSTALL_DIR}/community/op_proto/")
cann_install(
    TARGET      op_proto
    FILES       $<TARGET_FILE:op_proto>
    DESTINATION "${PROTO_INSTALL_PATH}"
)

file(GLOB PROTO_INC_FILES ./**/op_proto/inc/*.h)
set(PROTO_INC_INSTALL_PATH "${INSTALL_DIR}/community/op_proto/inc")
add_custom_target(copy_proto_inc ALL)
cann_install(
    TARGET      copy_proto_inc
    FILES       ${PROTO_INC_FILES}
    DESTINATION "${PROTO_INC_INSTALL_PATH}"
)

if(ALL_UT OR PROTO_UT)
    add_library(op_proto_llt STATIC
        ${PROTO_SRC}
        ${_proto_h}
    )
    add_dependencies(op_proto_llt secure_c external_protobuf external_gtest)
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
file(GLOB TILING_SRC
    ./**/ai_core/op_tiling/*.cc
    ${CANN_ROOT_DIR}/community/common/utils/tiling_util.cc
    ${CANN_ROOT_DIR}/community/common/utils/fp16_t.cc
)

set(TILING_LINK_LIB
    -Wl,--no-as-needed
    graph
    #exe_graph
    register
    #alog
    #error_manager
    #ascend_protobuf
    -Wl,--as-needed
    #c_sec
)

set(TILING_INC
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/exe_graph
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
    ${CANN_ROOT_DIR}/community/common
    ${CANN_ROOT_DIR}/community/common/inc
    ${C_SEC_INCLUDE}
    ${JSON_INCLUDE}
    #${Protobuf_INCLUDE}
)

add_library(optiling SHARED
    ${TILING_SRC}
)

add_dependencies(optiling secure_c external_protobuf external_gtest nlohmann_json)

target_include_directories(optiling PRIVATE
    ${TILING_INC}
)

target_link_libraries(optiling PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ${TILING_LINK_LIB}
)

target_compile_options(optiling PRIVATE
    ${compile_opt_mode}
    -Dgoogle=ascend_private
    "-fno-strict-aliasing"
    "-fno-common"
)

set(TILING_INSTALL_PATH "${INSTALL_DIR}/community/op_tiling")
cann_install(
    TARGET      optiling
    FILES       $<TARGET_FILE:optiling>
    DESTINATION "${TILING_INSTALL_PATH}"
)

if(ALL_UT OR TILING_UT)
    add_library(op_tiling_llt STATIC
        ${TILING_SRC}
    )
    add_dependencies(op_tiling_llt secure_c external_protobuf external_gtest)
    if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
        set(compile_opt_mode ${CMAKE_BUILD_MODE})
    else()
        set(compile_opt_mode -O0)
    endif()
    target_include_directories(op_tiling_llt PUBLIC
        ${TILING_INC}
    )
    target_compile_options(op_tiling_llt PUBLIC
        ${compile_opt_mode}
        -Dgoogle=ascend_private
    )
    target_link_libraries(op_tiling_llt
        PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub>
        PUBLIC
        ${TILING_LINK_LIB}
    )
endif()

# -----------------------------------------------------------------------
file(GLOB OP_IMPL_FILES ./**/ai_core/cust_impl/*.py)
set(OP_IMPL_INSTALL_PATH "${INSTALL_DIR}/community/op_impl")
add_custom_target(copy_op_impl ALL)
cann_install(
    TARGET      copy_op_impl
    FILES       ${OP_IMPL_FILES}
    DESTINATION "${OP_IMPL_INSTALL_PATH}"
)
file(GLOB OP_IMPL_DYNAMIC_FILES ./**/ai_core/cust_impl/dynamic/*.py)
set(OP_IMPL_DYNAMIC_INSTALL_PATH "${INSTALL_DIR}/community/op_impl/dynamic")
add_custom_target(copy_op_impl_dynamic ALL)
cann_install(
        TARGET      copy_op_impl_dynamic
        FILES       ${OP_IMPL_DYNAMIC_FILES}
        DESTINATION "${OP_IMPL_DYNAMIC_INSTALL_PATH}"
)
# ----------------------------------------------------------------------------
set(INI_PARSER ${CANN_ROOT_DIR}/scripts/ai_core_parse_ini.py)

file(GLOB OP_310_INI_FILES ./**/ai_core/op_info_cfg/ascend310/*.ini)
file(MAKE_DIRECTORY ${INSTALL_DIR}/community/op_config/ascend310)
set(OP_310_INI_INSTALL_PATH ${INSTALL_DIR}/community/op_config/ascend310/aic_ops_info.json)
add_custom_command(OUTPUT ${OP_310_INI_INSTALL_PATH}
    COMMAND python3 ${INI_PARSER} ${OP_310_INI_FILES} ${OP_310_INI_INSTALL_PATH} 1>/dev/null
    COMMENT "Parsing ini file ${OP_310_INI_FILES}"
)
add_custom_target(aic_310_ops_info ALL
    DEPENDS ${OP_310_INI_INSTALL_PATH}
)

file(GLOB OP_310P_INI_FILES ./**/ai_core/op_info_cfg/ascend310p/*.ini)
file(MAKE_DIRECTORY ${INSTALL_DIR}/community/op_config/ascend310p)
set(OP_310P_INI_INSTALL_PATH ${INSTALL_DIR}/community/op_config/ascend310p/aic_ops_info.json)
add_custom_command(OUTPUT ${OP_310P_INI_INSTALL_PATH}
    COMMAND python3 ${INI_PARSER} ${OP_310P_INI_FILES} ${OP_310P_INI_INSTALL_PATH} 1>/dev/null
    COMMENT "Parsing ini file ${OP_310P_INI_FILES}"
)
add_custom_target(aic_310p_ops_info ALL
    DEPENDS ${OP_310P_INI_INSTALL_PATH}
)

file(GLOB OP_910_INI_FILES ./**/ai_core/op_info_cfg/ascend910/*.ini)
file(MAKE_DIRECTORY ${INSTALL_DIR}/community/op_config/ascend910)
set(OP_910_INI_INSTALL_PATH ${INSTALL_DIR}/community/op_config/ascend910/aic_ops_info.json)
add_custom_command(OUTPUT ${OP_910_INI_INSTALL_PATH}
    COMMAND python3 ${INI_PARSER} ${OP_910_INI_FILES} ${OP_910_INI_INSTALL_PATH} 1>/dev/null
    COMMENT "Parsing ini file ${OP_910_INI_FILES}"
)
add_custom_target(aic_910_ops_info ALL
    DEPENDS ${OP_910_INI_INSTALL_PATH}
)

file(GLOB OP_910B_INI_FILES ./**/ai_core/op_info_cfg/ascend910b/*.ini)
file(MAKE_DIRECTORY ${INSTALL_DIR}/community/op_config/ascend910b)
set(OP_910B_INI_INSTALL_PATH ${INSTALL_DIR}/community/op_config/ascend910b/aic_ops_info.json)
add_custom_command(OUTPUT ${OP_910B_INI_INSTALL_PATH}
    COMMAND python3 ${INI_PARSER} ${OP_910B_INI_FILES} ${OP_910B_INI_INSTALL_PATH} 1>/dev/null
    COMMENT "Parsing ini file ${OP_910B_INI_FILES}"
)
add_custom_target(aic_910b_ops_info ALL
    DEPENDS ${OP_910B_INI_INSTALL_PATH}
)
# --------------------------------aicpu---------------------------------------

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

if(EXISTS $ENV{ASCEND_AICPU_PATH}/opp/op_impl/built-in/aicpu/aicpu_kernel)
    set(AICPU_OPP_ENV $ENV{ASCEND_AICPU_PATH}/opp/op_impl/built-in/aicpu/aicpu_kernel)
else()
    set(AICPU_OPP_ENV $ENV{ASCEND_AICPU_PATH}/opp/built-in/op_impl/aicpu/aicpu_kernel)
endif()

set(AICPU_INCLUDE ${AICPU_OPP_ENV}/inc)

set(AICPU_INC
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
    ${CANN_ROOT_DIR}/community/common
    ${CANN_ROOT_DIR}/community/common/inc
    ${CANN_ROOT_DIR}/community/common/src
    ${AICPU_INCLUDE}
    ${EIGEN_INCLUDE}
    ${C_SEC_INCLUDE}
    #${Protobuf_INCLUDE}
)

add_library(cust_aicpu_kernels SHARED
    ${AICPU_SRC}
)

target_include_directories(cust_aicpu_kernels PRIVATE
    ${AICPU_INC}
)

target_link_libraries(cust_aicpu_kernels PRIVATE
    ${AICPU_LINK_LIB}
)

target_compile_options(cust_aicpu_kernels PRIVATE
    ${compile_opt_mode}
    -Dgoogle=ascend_private
    "-fno-strict-aliasing"
    "-fno-common"
)

set(AICPU_PATH "${INSTALL_DIR}/community/cpu/aicpu_kernel/impl")
cann_install(
    TARGET      cust_aicpu_kernels
    FILES       $<TARGET_FILE:cust_aicpu_kernels>
    DESTINATION "${AICPU_PATH}"
)
