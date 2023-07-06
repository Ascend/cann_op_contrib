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
ut_exe="./community/tests/ascendc_ut"

mk_dir() {
  local create_dir="$1"
  mkdir -pv "${create_dir}" > /dev/null
  echo "Created ${create_dir}"
}

build_ut() {
  echo "Build UT"
  mk_dir "${BUILD_PATH}"
  product_type="ascend910"
  if [ $SOC_VERSION ];then
    product_type=${SOC_VERSION,,}
    product_910="ascend910"
    product_610="ascend610"
    product_310="ascend310p"
    if [[ $product_type =~ $product_910 ]];then
      product_type=$product_910
    elif [[ $product_type =~ $product_610 ]];then
      product_type=$product_610
    elif [[ $product_type =~ "ascend310" ]];then
      product_type=$product_310
    else
      product_type=$product_910
    fi
  fi
  cd "${BUILD_PATH}" && cmake  .. -DAICPU_UT=False -DPROTO_UT=False -DTILING_UT=False -DASCENDC_UT=True -Dproduct_type=${product_type}
  make clean
  make ${VERBOSE} -j $1
  if [ $? -ne 0 ];then
    echo "CANN Ascend C UT faild"
    exit 1
  else
    echo "CANN Ascend C UT success!"
  fi
}

check_dir_exe(){
  if [ ! -d "$build_dir" ] || [ ! -f "$build_dir/$ut_exe" ];then
    echo "need compile first"
    exit 1
  fi
}

gen_data() {
  python3 scripts/gen_test_data.py ascendc
}

change_dir(){
  cd $1
}

run_ut(){
  ./$ut_exe
  if [ $? -ne 0 ];then
    echo "RUN Ascend C UT FAILD"
    exit 1
  else
    echo "RUN Ascend C UT SUCCESS!"
  fi
}

gen_cover(){
  lcov -d ./ -c -o init.info 
  lcov -a init.info -o total.info
  lcov --remove total.info '*/usr/include/*' "*/op_proto/*" "*/community/common/*" "*/build/proto/*" '*/Ascend/ascend-toolkit/*' '*/ascend_protobuf/include/*' '*/eigen/include/*' '*/usr/lib/*' '*/usr/lib64/*' '*/src/log/*' '*/tests/*' '*/usr/local/include/*' '*/usr/local/lib/*' '*/usr/local/lib64/*' '*/third/*' 'testa.cpp' -o final.info
  genhtml -o cover_report_ascendc --legend --title "lcov"  --prefix=./ final.info
}

main() {
  build_ut $1
  change_dir $BASE_PATH
  check_dir_exe
  gen_data
  change_dir $build_dir
  run_ut
  if [ "$2" != "no_report" ];then
    gen_cover
  fi
}
main $1 $2