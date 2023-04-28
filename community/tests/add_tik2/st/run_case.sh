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
if [ -z $ASCEND_CUSTOM_PATH ];then
  ASCEND_CUSTOM_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi
export DDK_PATH=$ASCEND_CUSTOM_PATH
export NPU_HOST_LIB=$ASCEND_CUSTOM_PATH/runtime/lib64/stub
source $ASCEND_CUSTOM_PATH/../set_env.sh
$ASCEND_CUSTOM_PATH/python/site-packages/bin/msopst run -i add_tik2.json -soc Ascend910A -out tik2_st/out/