#!/bin/bash
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
export BASE_PATH=$(cd "$(dirname $0)"; pwd)
export BUILD_PATH="${BASE_PATH}/build"
CMAKE_HOST_PATH="${BUILD_PATH}/cann"
TAR_DIR_PATH="${BASE_PATH}/CANN_OP_CONTRIB"

mk_dir() {
  local create_dir="$1"
  mkdir -pv "${create_dir}" > /dev/null
  echo "Created ${create_dir}"
}

# create build path
build_cann() {
  echo "Create build directory and build CANN"

  mk_dir "${BUILD_PATH}/install/community/aicpu/cfg" > /dev/null
  python scripts/parser_ini.py *.ini ${BUILD_PATH}/install/community/aicpu/cfg/aicpu_kernel.json

  mk_dir "${CMAKE_HOST_PATH}"
  cd "${CMAKE_HOST_PATH}" && cmake  ../..
  make ${VERBOSE} -j${THREAD_NUM}
  echo "CANN build success!"
}

change_dir() 
{
  mk_dir "${TAR_DIR_PATH}/framework/vendor/community" > /dev/null
  mk_dir "${TAR_DIR_PATH}/op_proto/vendor/community" > /dev/null
  mk_dir "${TAR_DIR_PATH}/op_impl/vendor/community/ai_core/tbe" > /dev/null

  cp -r ${BUILD_PATH}/install/community/framework ${TAR_DIR_PATH}/framework/vendor/community/ > /dev/null
  cp -r ${BUILD_PATH}/install/community/op_proto ${TAR_DIR_PATH}/op_proto/vendor/community/ > /dev/null
  cp -r ${BUILD_PATH}/install/community/op_tiling ${TAR_DIR_PATH}/op_impl/vendor/community/ai_core/tbe > /dev/null
}

main() {
  # CANN build start
  build_cann
  # Change dir
  change_dir
}
main
