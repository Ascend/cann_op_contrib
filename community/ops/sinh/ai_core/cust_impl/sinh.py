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

import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check


@register_op_compute("sinh")
def sinh_compute(x, y, kernel_name="sinh"):
    two_tensor = tbe.broadcast(2.0, x.shape, output_dtype = x.dtype)
    one_unm_tensor = tbe.broadcast(-1, x.shape, output_dtype = x.dtype)
    res_exp = tbe.vexp(x)
    res_unm_exp = tbe.vexp(tbe.vdiv(x, one_unm_tensor))
    res_sub = tbe.vsub(res_exp, res_unm_exp)
    res = tbe.vdiv(res_sub, two_tensor)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sinh(x, y, kernel_name="sinh"):
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")

    res = sinh_compute(data_x, y, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    tbe.build(schedule, config)
