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

if(PROTO_UT)
    file(GLOB PROTO_UT_SRC ./**/ut/*_proto_ut.cc)

    add_executable(op_proto_ut
        ${PROTO_UT_SRC}
    )
    target_include_directories(op_proto_ut PRIVATE
            #${GTEST_INCLUDE}
            #${OP_TEST_PLUFIN_AND_PROTO_INCLUDE}
            #${METADEF_INCLUDE}/graph
            #${METADEF_INCLUDE}/exe_graph
            #${METADEF_INCLUDE}/inc/external/graph
            )
    target_compile_definitions(op_proto_ut PRIVATE
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
            graph_dsl
            -Wl,--no-whole-archive
            )

    set_target_properties(ops_cpp_proto_utest PROPERTIES CXX_STANDARD 17)
    include(CTest)

    if(NOT UT_NO_EXEC)
        add_custom_command(
                TARGET ops_cpp_proto_utest POST_BUILD
                COMMAND ops_cpp_proto_utest
                COMMENT "Run ops proto utest"
        )
    endif()
endif()