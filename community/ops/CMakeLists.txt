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
set(OPENSDK ${ASCEND_DIR}/opensdk/opensdk)
set(METADEF_INCLUDE ${OPENSDK}/include/metadef)
set(GRAPHENGINE_INCLUDE ${OPENSDK}/include/air)

set(PROTO "${OPENSDK}/proto/insert_op.proto")
protobuf_generate(opp _proto_cc _proto_h ${PROTO})


set(_op_proto_utils
    ${CANN_ROOT_DIR}/community/common/utils/array_ops_shape_fns.cc
    ${CANN_ROOT_DIR}/community/common/utils/axis_util.cc
    ${CANN_ROOT_DIR}/community/common/utils/candidate_sampling_shape_fns.cc
    ${CANN_ROOT_DIR}/community/common/utils/common_shape_fns.cc
    ${CANN_ROOT_DIR}/community/common/src/error_util.cc
    ${CANN_ROOT_DIR}/community/common/src/op_util.cc
    ${CANN_ROOT_DIR}/community/common/utils/images_ops_shape_fns.cc
    ${CANN_ROOT_DIR}/community/common/utils/linalg_ops_shape_fns.cc
    ${CANN_ROOT_DIR}/community/common/utils/lookup_ops_shape_fns.cc
    ${CANN_ROOT_DIR}/community/common/utils/nn_shape_fns.cc
    ${CANN_ROOT_DIR}/community/common/utils/ragged_conversion_ops_shape_fns.cc
    ${CANN_ROOT_DIR}/community/common/utils/random_ops_shape_fns.cc
    ${CANN_ROOT_DIR}/community/common/utils/resource_variable_ops_shape_fns.cc
    ${CANN_ROOT_DIR}/community/common/utils/transfer_shape_according_to_format.cc
    ${CANN_ROOT_DIR}/community/common/utils/util.cc
    ${CANN_ROOT_DIR}/community/common/utils/reduce_infer_util.cc
)

file(GLOB PROTO_SRC
    ./**/op_proto/*.cc
)

set(PROTO_INC
    ${CANN_ROOT_DIR}/community/common/utils
    ${METADEF_INCLUDE}/exe_graph
    ${METADEF_INCLUDE}/external
    ${METADEF_INCLUDE}/external/graph
    ${GRAPHENGINE_INCLUDE}
    ${GRAPHENGINE_INCLUDE}/external
    ${OPENSDK}/include/slog
    ${OPENSDK}/c_sec/include
    ${CANN_ROOT_DIR}/community/common
    ${CANN_ROOT_DIR}/community/common/inc
    ${CANN_ROOT_DIR}/community/common/src
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
    ${_op_proto_utils}
    ${_proto_h}
)

#add_dependencies(op_proto secure_c external_protobuf external_gtest)

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
    #add_dependencies(op_proto_llt secure_c external_protobuf external_gtest)
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
    ${METADEF_INCLUDE}/exe_graph
    ${METADEF_INCLUDE}/external
    ${METADEF_INCLUDE}/external/graph
    ${OPENSDK}/include/slog
    ${OPENSDK}/c_sec/include
    ${GRAPHENGINE_INCLUDE}
    ${GRAPHENGINE_INCLUDE}/external
    ${CANN_ROOT_DIR}/community/common
    ${CANN_ROOT_DIR}/community/common/inc
    ${C_SEC_INCLUDE}
    ${JSON_INCLUDE}
    #${Protobuf_INCLUDE}
)

add_library(optiling SHARED
    ${TILING_SRC}
)

#add_dependencies(optiling secure_c external_protobuf external_gtest nlohmann_json)

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

# -----------------------------------------------------------------------
file(GLOB OP_IMPL_FILES ./**/ai_core/impl/*)
set(OP_IMPL_INSTALL_PATH "${INSTALL_DIR}/community/op_impl")
add_custom_target(copy_op_impl ALL)
cann_install(
    TARGET      copy_op_impl
    FILES       ${OP_IMPL_FILES}
    DESTINATION "${OP_IMPL_INSTALL_PATH}"
)
file(GLOB OP_IMPL_DYNAMIC_FILES ./**/ai_core/impl/dynamic/*.py)
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

# --------------------------- build tf_plugin --------------------------
set(TF_PROTO
    "${OPENSDK}/proto/tensorflow/versions.proto"
    "${OPENSDK}/proto/tensorflow/resource_handle.proto"
    "${OPENSDK}/proto/tensorflow/tensor_shape.proto"
    "${OPENSDK}/proto/tensorflow/function.proto"
    "${OPENSDK}/proto/tensorflow/node_def.proto"
    "${OPENSDK}/proto/tensorflow/graph.proto"
    "${OPENSDK}/proto/tensorflow/tensor.proto"
    "${OPENSDK}/proto/tensorflow/attr_value.proto"
    "${OPENSDK}/proto/tensorflow/graph_library.proto"
    "${OPENSDK}/proto/tensorflow/types.proto"
    "${OPENSDK}/proto/tensorflow/op_def.proto"
)
protobuf_generate(opp _proto_cc _proto_h ${TF_PROTO})
file(GLOB TF_PLUGIN_SRC ${CANN_ROOT_DIR}/community/ops/**/framework/tf/*.cc)

set(TF_PLUGIN_INC
    ${METADEF_INCLUDE}/external
    ${GRAPHENGINE_INCLUDE}
)

set(TF_PLUGIN_LINK_LIB
    -Wl,--no-as-needed
    #graph
    #register
    #alog
    #error_manager
    #te_fusion
    #_caffe_parser
    #ascend_protobuf
    -Wl,--as-needed
    #c_sec
)
add_library(tf_plugin SHARED
    ${TF_PLUGIN_SRC}
    ${_proto_h}
)

target_include_directories(tf_plugin PRIVATE
    ${TF_PLUGIN_INC}
)

target_link_libraries(tf_plugin PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ${TF_PLUGIN_LINK_LIB}
)

target_compile_options(tf_plugin PRIVATE
    "-fno-common"
    "-fno-strict-aliasing"
    ${compile_opt_mode}
    -Werror
    -Wno-deprecated-declarations
    -Dgoogle=ascend_private
)

set(TF_PLUGIN_PATH "${INSTALL_DIR}/community/framework/tf")
cann_install(
    TARGET      tf_plugin
    FILES       $<TARGET_FILE:tf_plugin>
    DESTINATION "${TF_PLUGIN_PATH}"
)

# --------------------------- build onnx_plugin --------------------------

set(ONNX_PROTO "${OPENSDK}/proto/onnx/ge_onnx.proto")
protobuf_generate(opp _proto_cc _proto_h ${ONNX_PROTO})

file(GLOB ONNX_PLUGIN_SRC ${CANN_ROOT_DIR}/community/ops/**/framework/onnx/*.cc)
set(ONNX_PLUGIN_INC
    ${METADEF_INCLUDE}
    ${METADEF_INCLUDE}/external
)

set(ONNX_PLUGIN_LINK_LIB
    -Wl,--no-as-needed
    #graph
    #register
    #alog
    #error_manager
    #te_fusion
    #_caffe_parser
    #ascend_protobuf
    -Wl,--as-needed
    #c_sec
)

add_library(onnx_plugin SHARED
    ${ONNX_PLUGIN_SRC}
    ${_proto_h}
)

target_include_directories(onnx_plugin PRIVATE
    ${ONNX_PLUGIN_INC}
)

target_link_libraries(onnx_plugin PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ${ONNX_PLUGIN_LINK_LIB}
)

target_compile_options(onnx_plugin PRIVATE
    "-fno-common"
    "-fno-strict-aliasing"
    ${compile_opt_mode}
    -Werror
    -Wno-deprecated-declarations
    -Dgoogle=ascend_private
)

set(ONNX_PLUGIN_PATH "${INSTALL_DIR}/community/framework/onnx")
cann_install(
    TARGET      onnx_plugin
    FILES       $<TARGET_FILE:onnx_plugin>
    DESTINATION "${ONNX_PLUGIN_PATH}"
)

# --------------------------- build caffe_plugin --------------------------
set(_proto_srcs
    "${OPENSDK}/proto/caffe/caffe.proto"
  )

  protobuf_generate(opp _proto_cc _proto_h ${_proto_srcs})

  file(GLOB CAFFE_PLUGIN_SRC ${CANN_ROOT_DIR}/community/ops/**/framework/caffe/*.cc)

  add_library(caffe_plugin SHARED
    ${CAFFE_PLUGIN_SRC}
    ${_proto_h}
  )
    set(CAFFE_PLUGIN_INC
        ${METADEF_INCLUDE}
        ${METADEF_INCLUDE}/external
        ${GRAPHENGINE_INCLUDE}
    )

  target_include_directories(caffe_plugin PRIVATE
    ${CAFFE_PLUGIN_INC}

  )

  target_compile_options(caffe_plugin PRIVATE
    "-fno-common"
    "-fno-strict-aliasing"
    -O2
    -Werror
    -Wno-deprecated-declarations
    -Dgoogle=ascend_private
  )

  target_link_libraries(caffe_plugin PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:slog_headers>>
    -Wl,--no-as-needed
      graph
      register
      alog
      error_manager
      te_fusion
      _caffe_parser
      ascend_protobuf
    -Wl,--as-needed
      c_sec
  )

set(CAFFE_PLUGIN_PATH "${INSTALL_DIR}/community/framework/caffe")
cann_install(
    TARGET      caffe_plugin
    FILES       $<TARGET_FILE:caffe_plugin>
    DESTINATION "${CAFFE_PLUGIN_PATH}"
)