#
# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.  All rights reserved.
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
#-----------------------------------npu_supported_ops_json-------------------------------------------
set(HOST_DIR ${BUILD_DIR}/tik2)
set(OPENSDK ${ASCEND_DIR}/opensdk/opensdk)

file(GLOB tf_plugin_srcs ${CANN_ROOT_DIR}/community/ops/**/framework/tf/*.cc
)

add_library(cust_tf_parsers SHARED
    ${tf_plugin_srcs}
)
target_include_directories(cust_tf_parsers PRIVATE
    ${OPENSDK}/include/air
)
target_compile_definitions(cust_tf_parsers PRIVATE google=ascend_private)
target_link_libraries(cust_tf_parsers PRIVATE intf_pub graph)

set(TF_PATH ${HOST_DIR}/framework/tensorflow)

execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ${TF_PATH})
cann_install(
    TARGET      cust_tf_parsers
    FILES       $<TARGET_FILE:cust_tf_parsers>
    DESTINATION "${TF_PATH}"
)

execute_process( COMMAND ${CANN_ROOT_DIR}/scripts/gen_ops_filter.sh
        ${HOST_DIR}
        ${TF_PATH})