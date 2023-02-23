/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "inc/sinh_op.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(SinhInferShape)
{
    auto input_x = op.GetInputDescByName("x");
    auto input_type = input_x.GetDataType();
    auto input_shape = input_x.GetShape();
    auto out_y = op.GetOutputDescByName("y");
    out_y.SetDataType(input_type);
    out_y.SetShape(input_shape);
    op.UpdateOutputDesc("y", out_y);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Sinh, SinhVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Sinh, SinhInferShape);
VERIFY_FUNC_REG(Sinh, SinhVerify);
}
// namespace ge
