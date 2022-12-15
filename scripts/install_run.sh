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
param_usage()
{
  echo "Usage: ./${0##*/} [options]"
  echo "Options:"
  echo "  --help | -h                       Print this message"
  echo "  --noexec                          Do not run embedded script"
  echo "  --extract=<path>                  Extract directly to a target directory (absolute or relative)"
  echo "                                    Usually used with --noexec to just extract files without running"
  echo "  --install                         Install run mode"
  echo "  --uninstall                       Uninstall"
  echo "  --upgrade                         Upgrade"
}
# 检查路径字符串
function check_path() {
    local path_str=${1}
    # 判断路径字符串长度
    if [ ${#path_str} -gt 4096 ]; then
        echo "[ERROR] parameter error $path_str, the length exceeds 4096."
        exit 1
    fi
    # 黑名单设置，不允许//，...这样的路径
    if echo "${path_str}" | grep -Eq '\/{2,}|\.{3,}'; then
        echo "[ERROR] The path ${path_str} is invalid, cannot contain the following characters: // ...!"
        exit 1
    fi
    # 白名单设置，只允许常见字符
    if ! echo "${path_str}" | grep -Eq '^\~?[a-zA-Z0-9./_-]*$'; then
        echo "[ERROR] The path ${path_str} is invalid, only [a-z,A-Z,0-9,-,_] is support!"
        exit 1
    fi
    echo "Creating directory ${path_str}"
    if [ ! -x ${path_str} ];then
        if [ ! -d ${path_str} ];then
            mkdir -p ${path_str}
            if [ $? -ne 0 ];then
                echo "[ERROR] The path ${path_str} create failed, please check the input!" 
                exit 1
            fi
        else
            echo "[ERROR] The path ${path_str} does not have executable permissions, please modify!" 
            exit 1
        fi
    fi
}
param_check()
{
    # 空参数判断
    if [[ x${!paramDict[@]} == x ]] ; then
        param_usage
	exit 1
    fi
    # 参数重复检查
    for key in "${!paramDict[@]}"; do
        if [ ${paramDict[$key]} -gt 1 ]; then
	    echo "[ERROR] parameter error ! param $key is repeat."
            exit 1
        fi
    done
    # 必选参数判断
    local args_num=0
    if [ x${paramDict["install"]} == x1 ]; then
        let 'args_num+=1'
	operate="install"
    fi
    if [ x${paramDict["upgrade"]} == x1 ]; then
        let 'args_num+=1'
	operate="upgrade"
    fi
    if [ x${paramDict["uninstall"]} == x1 ]; then
        let 'args_num+=1'
	operate="uninstall"
    fi
    if [ x${paramDict["noexec"]} == x1 ]; then
        let 'args_num+=1'
    fi
    if [ $args_num -lt 1 ]; then
        echo "[ERROR] parameter error ! Scene is neither install nor uninstall, upgrade, noexec."
        exit 1
    fi
    if [ $args_num -gt 1 ]; then
        echo "[ERROR] parameter error ! Scene conflict."
        exit 1
    fi
}
lines=0
md5="md5_value"
declare -A paramDict=()
ARGS=`getopt -q -o h --long help,install,uninstall,upgrade,noexec,extract: -n "$0" -- "$@"`
if [ $? != 0 ]; then
    param_usage
    exit 1
fi
while true
do
    case "$1" in
        -h|--help)
            param_usage
            exit 0 ;;
        --install)
	    let paramDict["install"]++
            shift
            ;;
        --uninstall)
	    let paramDict["uninstall"]++
            shift
            ;;
	--upgrade)
	    let paramDict["upgrade"]++
            shift
	    ;;
	--extract=*)
	    let paramDict["extract"]++
            extract_path=$(echo "${1}" | cut -d"=" -f2)
	    check_path $extract_path
	    shift
            ;;
	--noexec)
            let paramDict["noexec"]++
            shift
            ;;
        --)
	    param_usage
            shift
            exit 0 ;;
        *)
	    if [ "x$1" != "x" ]; then
	        param_usage
                exit 0
            fi
	    break
	    ;;
    esac
done
param_check

# 创建临时目录，解压文件
script_path="/tmp/CANN_OP_CONTRIB_`date "+%Y%m%d%H%M%S"`"
mkdir ${script_path}
tail -n +$lines $0 >${script_path}/CANN_OP_CONTRIB.tar.gz
# md5校验
md5_ver=`sha256sum ${script_path}/CANN_OP_CONTRIB.tar.gz | awk '{print $1}'`
if [[ $md5 != $md5_ver ]];then
    echo "[ERROR] SHA256 checksums error, please check package"
    rm -rf ${script_path}
    exit 1
else
    echo "SHA256 checksums are OK. All good."
fi
# 解压
tar -xf ${script_path}/CANN_OP_CONTRIB.tar.gz -C ${script_path} > /dev/null
rm ${script_path}/CANN_OP_CONTRIB.tar.gz
if [ x${extract_path} != x ]; then
    cp -r ${script_path}/* ${extract_path}
fi


if [ x${paramDict["noexec"]} != x1 ]; then
    bash ${script_path}/CANN_OP_CONTRIB/scripts/CANN_OP_CONTRIB_install.sh ${0##*/} $operate
fi

rm -rf ${script_path}
exit 0
