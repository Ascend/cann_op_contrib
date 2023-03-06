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

#ifndef COMMUNITY_OPS_CHOLESKY_GRAD_OP_PROTO_INC_CHOLESKY_GRAD_OP_H
#define COMMUNITY_OPS_CHOLESKY_GRAD_OP_PROTO_INC_CHOLESKY_GRAD_OP_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the reverse mode backpropagated gradient of the Cholesky
algorithm . \n

* @par Inputs:
* The input x has to be symmetric and positive definite. Inputs include:
* @li x:A Tensor. Must be one of the following types: double, float32. Output
of batch Cholesky algorithm x = cholesky(A). Shape is [..., M, M]. Algorithm
depends only on lower triangular part of the innermost matrices of this tensor.
* @li grad:A Tensor. Must have the same type as l. df/dx where f is some
scalar function. Shape is [..., M, M]. Algorithm depends only on lower
triangular part of the innermost matrices of this tensor . \n

* @par Outputs:
* y:A Tensor. Has the same type as x . \n

* @attention Constraints:
* The input x is a tensor of shape [..., M, M] whose inner-most 2 dimensions
form square matrices.

* @par Third-party framework compatibility
* Compatible with tensorflow CholeskyGrad operator.
*/

REG_OP(CholeskyGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(CholeskyGrad)
}
# endif