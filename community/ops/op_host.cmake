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

set(HOST_DIR ${BUILD_DIR}/tik2)

execute_process(COMMAND arch OUTPUT_VARIABLE CMAKE_SYSTEM_PROCESSOR OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ${HOST_DIR})


file(GLOB OP_HOST_SRC ${CANN_ROOT_DIR}/community/ops/**/ai_core/op_host/*.cpp
)
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -g -fPIC -shared -std=c++11 ${OP_HOST_SRC} -D_GLIBCXX_USE_CXX11_ABI=0
                -I  ${ASCEND_DIR}/include -L  ${ASCEND_DIR}/lib64 -lexe_graph -lregister
                -o ${HOST_DIR}/libascend_all_ops.so
                RESULT_VARIABLE EXEC_RESULT
                OUTPUT_VARIABLE EXEC_INFO
                ERROR_VARIABLE  EXEC_ERROR
)

set(ENV{LD_LIBRARY_PATH} "${ASCEND_DIR}/lib64:$ENV{LD_LIBRARY_PATH}")

execute_process(COMMAND ls  ${HOST_DIR}/libascend_all_ops.so  OUTPUT_VARIABLE build_file)

execute_process(COMMAND   ${ASCEND_DIR}/toolkit/tools/opbuild/op_build
                ${HOST_DIR}/libascend_all_ops.so ${HOST_DIR}
                RESULT_VARIABLE EXEC_RESULT
                OUTPUT_VARIABLE EXEC_INFO
                ERROR_VARIABLE  EXEC_ERROR
)


add_library(cust_op_proto SHARED ${OP_HOST_SRC} ${HOST_DIR}/op_proto.cc)
target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
target_link_libraries(cust_op_proto PRIVATE intf_pub exe_graph register)
set(OPENSDK $ENV{ASCEND_CUSTOM_PATH}/opensdk/opensdk)
target_include_directories(cust_op_proto PRIVATE
    ${OPENSDK}/c_sec/include
    ${OPENSDK}/include/air/external
)

SET(LIBRARY_OUTPUT_PATH ${HOST_DIR})
set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME
                      cust_opsproto_rt2.0
)

SET(OP_PROTO_PATH ${HOST_DIR}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
cann_install(
    TARGET      cust_op_proto
    FILES       $<TARGET_FILE:cust_op_proto>
    DESTINATION "${OP_PROTO_PATH}"
)


SET(OP_INC_PATH ${HOST_DIR}/op_proto/inc)
execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ${OP_INC_PATH})
execute_process( COMMAND cp
        ${HOST_DIR}/op_proto.h
        ${OP_INC_PATH})
#---------------------------------tiling-------------------------------------------------------------

add_library(cust_optiling SHARED ${OP_HOST_SRC})
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_link_libraries(cust_optiling PRIVATE intf_pub graph register)
target_include_directories(cust_optiling PRIVATE
    ${OPENSDK}/c_sec/include
    ${OPENSDK}/include/air/external
)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                      cust_opmaster_rt2.0
)

set(OPTILING_PATH ${HOST_DIR}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
cann_install(
    TARGET      cust_optiling
    FILES       $<TARGET_FILE:cust_optiling>
    DESTINATION "${OPTILING_PATH}"
)

add_custom_target(optiling_compat ALL
                  COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
                          ${HOST_DIR}/op_impl/ai_core/tbe/op_tiling/liboptiling.so
)

