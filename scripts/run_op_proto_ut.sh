#!/bin/bash
# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.  All rights reserved.
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
CURR_PATH=$(cd $(dirname $0); pwd)
export BASE_PATH=$(cd ${CURR_PATH}/..; pwd)
export BUILD_PATH="${BASE_PATH}/build"

build_dir="./build"
ut_exe="./community/tests/ops_cpp_proto_utest"

mk_dir() {
  local create_dir="$1"
  mkdir -pv "${create_dir}" > /dev/null
  echo "Created ${create_dir}"
}

build_ut() {
  echo "Build OP_PROTO UT"
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}" && cmake  .. -DAICPU_UT=False -DPROTO_UT=True -DTILING_UT=False -DASCENDC_UT=False
  make clean
  make ${VERBOSE} -j $1
  if [ $? -ne 0 ];then
    echo "OP_PROTO UT faild"
    exit 1
  else
    echo "OP_PROTO UT success!"
  fi
}

check_dir_exe(){
  if [ ! -d "$build_dir" ] || [ ! -f "$build_dir/$ut_exe" ];then
    echo "need compile first"
    exit 1
  fi
}

change_dir(){
  cd $1
}

run_ut(){
  ./$ut_exe
  if [ $? -ne 0 ];then
    echo "RUN OP_PROTO UT FAILD"
    exit 1
  else
    echo "RUN OP_PROTO UT SUCCESS!"
  fi
}

gen_cover(){
  lcov -d ./ -c -o init.info 
  lcov -a init.info -o total.info
  lcov --remove total.info '*/usr/include/*' "*/ai_core/*" "*/aicpu/impl/*" "*/community/common/*" '*/Ascend/ascend-toolkit/*' '*/ascend_protobuf/include/*' '*/eigen/include/*' '*/usr/lib/*' '*/usr/lib64/*' '*/src/log/*' '*/tests/*' '*/usr/local/include/*' '*/usr/local/lib/*' '*/usr/local/lib64/*' '*/third/*' 'testa.cpp' -o final.info
  genhtml -o cover_report_op_proto --legend --title "lcov"  --prefix=./ final.info
}

main() {
  build_ut $1
  change_dir $BASE_PATH
  check_dir_exe
  change_dir $build_dir
  run_ut
  if [ "$2" != "no_report" ];then
    gen_cover
  fi
}
main $1 $2