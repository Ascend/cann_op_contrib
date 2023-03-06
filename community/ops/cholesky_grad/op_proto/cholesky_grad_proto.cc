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

#include "inc/cholesky_grad_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/linalg_ops_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(CholeskyGrad, CholeskyGradInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);

  GeShape y_shape;
  if (MakeBatchSquareMatrix(x_desc, y_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(),
            "Op CholeskyGrad first input x tensor make batch square matrix "
            "failed.");
    return GRAPH_FAILED;
  }

  DataType type = x_desc->GetDataType();
  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(y_shape);
  y_desc->SetDataType(type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CholeskyGrad, CholeskyGradInfer);
}