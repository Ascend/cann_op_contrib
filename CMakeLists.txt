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

cmake_minimum_required(VERSION 3.13)

project(cann C CXX)

if(BUILD_MODE)
    message(${BUILD_MODE})
else()
    set(BUILD_MODE "FALSE")
endif()
if(${BUILD_MODE} STREQUAL "FALSE")
    set(CMAKE_BUILD_MODE "${BUILD_MODE}")
else()
    set(CMAKE_BUILD_MODE "-${BUILD_MODE}")
endif()
if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
    set(compile_opt_mode ${CMAKE_BUILD_MODE})
else()
    set(compile_opt_mode -O2)
endif()

if(DEFINED ENV{D_PKG_SERVER})
    set(CANN_PKG_SERVER $ENV{D_PKG_SERVER})
    message("Download packages from PKG server")
endif()
set(CANN_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(BUILD_DIR ${CANN_ROOT_DIR}/build)
set(INSTALL_DIR "${BUILD_DIR}/install")
set(PROTO_BINARY_DIR "${BUILD_DIR}/proto")

add_compile_options(
    -Werror=sign-compare -Werror=address -Werror=unused-variable
    -Werror=narrowing -Werror=parentheses -Werror=unused-but-set-variable
    -Werror=return-type -Werror=unused-value
    -Werror=deprecated-declarations)

include(cmake/common_funcs.cmake)
include(cmake/util/intf_pub_linux.cmake)
include(cmake/util/util.cmake)
include(cmake/dependencies.cmake)

if(ALL_UT OR PROTO_UT OR TILING_UT OR AICPU_UT OR TIK2_UT)
    include(cmake/util/intf_pub_llt_gccnative.cmake)
endif()

if(BUILD_MODE)
    message(${BUILD_MODE})
else()
    set(BUILD_MODE "FALSE")
endif()
if(${BUILD_MODE} STREQUAL "FALSE")
    set(CMAKE_BUILD_MODE "${BUILD_MODE}")
else()
    set(CMAKE_BUILD_MODE "-${BUILD_MODE}")
endif()

add_subdirectory(community)