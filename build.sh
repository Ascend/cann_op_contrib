#!/bin/bash
# Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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
build_cann_tbe() {
  echo "Create build directory and build tbe"

  mk_dir "${BUILD_PATH}/install/community/cpu/config" > /dev/null
  python3 scripts/parser_ini.py *.ini ${BUILD_PATH}/install/community/cpu/config/cust_aicpu_kernel.json

  mk_dir "${CMAKE_HOST_PATH}"

  cd "${CMAKE_HOST_PATH}" && cmake  ../..
  make ${VERBOSE} -j${THREAD_NUM}
  if [ $? -ne 0 ];then
    echo "CANN tbe faild"
    exit 1
  else
    echo "CANN tbe success!"
  fi
}

build_cann_aicpu() {
  cd "${CMAKE_HOST_PATH}" && cmake ../.. -D BUILD_AICPU=True
  make ${VERBOSE} -j${THREAD_NUM}
  if [ $? -ne 0 ];then
    echo "CANN build aicpu faild"
    exit 1
  else
    echo "CANN build aicpu success!"
  fi
}

change_dir() 
{
  AI_CORE_PATH=""
  if [ -z $OPP_CUSTOM_VENDOR ];then
    AI_CORE_PATH="${TAR_DIR_PATH}/vendors/community/op_impl/ai_core/tbe/community_impl"
  else
    AI_CORE_PATH="${TAR_DIR_PATH}/vendors/community/op_impl/ai_core/tbe/${OPP_CUSTOM_VENDOR}_impl"
  fi
  mk_dir "${TAR_DIR_PATH}/vendors/community/op_impl/ai_core/tbe" > /dev/null
  mk_dir "${AI_CORE_PATH}" > /dev/null

  if [ -d ${BUILD_PATH}/install/community/framework ];then
    cp -r ${BUILD_PATH}/install/community/framework ${TAR_DIR_PATH}/vendors/community/ > /dev/null
  fi
  if [ -d ${BUILD_PATH}/install/community/op_proto ];then
    cp -r ${BUILD_PATH}/install/community/op_proto ${TAR_DIR_PATH}/vendors/community/ > /dev/null
  fi
  if [ -d ${BUILD_PATH}/install/community/op_tiling ];then
    cp -r ${BUILD_PATH}/install/community/op_tiling ${TAR_DIR_PATH}/vendors/community/op_impl/ai_core/tbe > /dev/null
  fi
  if [ -d ${BUILD_PATH}/install/community/op_impl ];then
    cp -r ${BUILD_PATH}/install/community/op_impl ${TAR_DIR_PATH}/vendors/community/op_impl/ai_core/tbe > /dev/null
    mv ${TAR_DIR_PATH}/vendors/community/op_impl/ai_core/tbe/op_impl/* ${AI_CORE_PATH}
    rm -rf ${TAR_DIR_PATH}/vendors/community/op_impl/ai_core/tbe/op_impl
  fi
  if [ -d ${BUILD_PATH}/install/community/op_config ];then
    cp -r ${BUILD_PATH}/install/community/op_config ${TAR_DIR_PATH}/vendors/community/op_impl/ai_core/tbe > /dev/null
    mv ${TAR_DIR_PATH}/vendors/community/op_impl/ai_core/tbe/op_config ${TAR_DIR_PATH}/vendors/community/op_impl/ai_core/tbe/config
  fi
}

change_dir_aicpu()
{
  if [ -d ${BUILD_PATH}/install/community/cpu ];then
    cp -r ${BUILD_PATH}/install/community/cpu ${TAR_DIR_PATH}/vendors/community/op_impl >/dev/null
  fi
}

ut_tbe() {
  mk_dir ${BUILD_PATH}
  cd ${BUILD_PATH}
  python3 ../scripts/run_tbe_ut_all.py
  if [ $? -ne 0 ];then
    echo "CANN build tbe ut faild"
    exit 1
  else
    echo "CANN build tbe ut success!"
  fi
  cd ${BASE_PATH}
}

ut_aicpu() {
  ./scripts/run_aicpu_ut.sh $1
  if [ $? -ne 0 ];then
    echo "CANN build aicpu ut faild"
    exit 1
  else
    echo "CANN build aicpu ut success!"
  fi
  cd ${BASE_PATH}
}

ut_proto(){
  ./scripts/run_op_proto_ut.sh $1
  if [ $? -ne 0 ];then
    echo "CANN build op_proto ut faild"
    exit 1
  else
    echo "CANN build op_proto ut success!"
  fi
  cd ${BASE_PATH}
}

ut_tiling(){
  ./scripts/run_tiling_ut.sh $1
  if [ $? -ne 0 ];then
    echo "CANN build tiling ut faild"
    exit 1
  else
    echo "CANN build tiling ut success!"
  fi
  cd ${BASE_PATH}
}

echo_help(){
  echo "eg: ./build.sh -u all            run all cases"
  echo "eg: ./build.sh -u tbe            run UT cases of tbe"
  echo "eg: ./build.sh -u aicpu          run UT cases of aicpu"
  echo "eg: ./build.sh -u proto          run UT cases of op proto"
  echo "eg: ./build.sh -u tiling         run UT cases of tiling"
  echo "eg: ./build.sh                   compile op of all"
}

main() {
  if [ $# == 0 ];then
    # CANN build start
    build_cann_tbe
    change_dir
    build_cann_aicpu
    change_dir_aicpu
  else
    if [ $1 == "tbe" ];then
      ut_tbe
    elif [ "$1" == "aicpu" ];then
      ut_aicpu $2
    elif [ "$1" == "proto" ];then
      ut_proto $2
    elif [ "$1" == "tiling" ];then
      ut_tiling $2
    elif [ "$1" == "all" ];then
      ut_tbe
      ut_aicpu
      ut_proto
      ut_tiling
    elif [ "$1" == "help" ] || [ "$1" == "h" ];then
      echo_help
    else
      echo "use ./build.sh h for get some help"
    fi
  fi
}

while getopts hj:u: OPTION;do
  case $OPTION in
  u)ut_type=$OPTARG
  ;;
  j)THREAD_NUM=$OPTARG
  ;;
  h)echo_help
  exit 0
  ;;
  ?)echo "get a non option $OPTARG and OPTION is $OPTION"
  ;;
  esac
done

main $ut_type $THREAD_NUM