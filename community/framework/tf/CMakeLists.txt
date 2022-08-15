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

set(TF_PROTO
    "${METADEF_DIR}/proto/tensorflow/versions.proto"
    "${METADEF_DIR}/proto/tensorflow/resource_handle.proto"
    "${METADEF_DIR}/proto/tensorflow/tensor_shape.proto"
    "${METADEF_DIR}/proto/tensorflow/function.proto"
    "${METADEF_DIR}/proto/tensorflow/node_def.proto"
    "${METADEF_DIR}/proto/tensorflow/graph.proto"
    "${METADEF_DIR}/proto/tensorflow/tensor.proto"
    "${METADEF_DIR}/proto/tensorflow/attr_value.proto"
    "${METADEF_DIR}/proto/tensorflow/graph_library.proto"
    "${METADEF_DIR}/proto/tensorflow/types.proto"
    "${METADEF_DIR}/proto/tensorflow/op_def.proto"
)

protobuf_generate(opp _proto_cc _proto_h ${TF_PROTO})

aux_source_directory(. TF_PLUGIN_SRC)

set(TF_PLUGIN_INC
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/third_party/graphengine/inc
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

#target_link_directories(tf_plugin PRIVATE "/home/cjl/Ascend/latest/compiler/lib64")

target_link_libraries(tf_plugin PRIVATE
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