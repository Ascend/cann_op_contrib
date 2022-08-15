# Copyright 2020 Huawei Technologies Co., Ltd
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

set(ONNX_PROTO "${METADEF_DIR}/proto/onnx/ge_onnx.proto")
protobuf_generate(opp _proto_cc _proto_h ${ONNX_PROTO})

aux_source_directory(. ONNX_PLUGIN_SRC)

set(ONNX_PLUGIN_INC
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
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

#(onnx_plugin PRIVATE "/home/cjl/Ascend/latest/compiler/lib64")

target_link_libraries(onnx_plugin PRIVATE
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