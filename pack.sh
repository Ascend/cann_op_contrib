#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

# 获取参数
#if [ $# -eq 2 ]; then
#    if [[ "$1" != "aarch64" && "$1" != "x86_64" ]]; then
#        echo "[Error] Input param error, Please input ./${0##*/} [aarch64/x86_64] [all/community/mdc/mindspore/other]"
#        exit
#    elif [[ "$2" != "community" && "$2" != "mdc" && "$2" != "mindspore" && "$2" != "other" && "$2" != "all" ]]; then
#        echo "[Error] Input param error, Please input ./${0##*/} [aarch64/x86_64] [all/community/mdc/mindspore/other]"
#        exit
#    else
#        targetkerner=$1
#        targetdir=$2
#    fi
#else
#    echo "[Error] Input param error, Please input ./${0##*/} [aarch64/x86_64] [all/community/mdc/mindspore/other]"
#    exit
#fi

# 生成编译结果
bash build.sh $targetkerner $targetdir
if [ $? -ne 0 ];then
    exit
fi

# 将脚本拷贝到编译后目录
mkdir -p ./CANN_OP_CONTRIB/scripts
cp ./scripts/install_run.sh ./CANN_OP_CONTRIB/scripts
cp ./scripts/CANN_OP_CONTRIB_install.sh ./CANN_OP_CONTRIB/scripts

# 判断编译后文件夹是否存在
if [ ! -d ./CANN_OP_CONTRIB/framework/vendor ] || [ ! -d ./CANN_OP_CONTRIB/op_proto/vendor ] || [ ! -d ./CANN_OP_CONTRIB/op_impl/vendor ];then
    echo "[ERROR] After compilation, the framework, op_proto and op_impl folders must exist in the ./CANN_OP_CONTRIB directory"
    exit
fi

# 设置run包前置脚本行数值
lines_num=`sed -n '$=' ./scripts/install_run.sh`
let lines_num++
sed -i "s/lines=0/lines=$lines_num/g" ./CANN_OP_CONTRIB/scripts/install_run.sh

# 打tar包
tar -zcf ./CANN_OP_CONTRIB.tar.gz ./CANN_OP_CONTRIB > /dev/null

# 设置md5值
md5=`sha256sum CANN_OP_CONTRIB.tar.gz | awk '{print $1}'`
sed -i "s/md5_value/$md5/g" ./CANN_OP_CONTRIB/scripts/install_run.sh

# 打run包
sed 's/^[ \t]*//g' ./CANN_OP_CONTRIB/scripts/install_run.sh > soft_install.sh
cat ./soft_install.sh ./CANN_OP_CONTRIB.tar.gz > ./CANN_OP_CONTRIB_linux${targetkerner}.run
chmod +x ./CANN_OP_CONTRIB_linux${targetkerner}.run

# 删除临时文件
rm -rf ./soft_install.sh ./CANN_OP_CONTRIB.tar.gz ./package.tar.gz ./CANN_OP_CONTRIB
