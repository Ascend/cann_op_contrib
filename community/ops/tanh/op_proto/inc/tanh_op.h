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

#ifndef COMMUNITY_OPS_TANH_OP_PROTO_INC_TANH_OP_H
#define COMMUNITY_OPS_TANH_OP_PROTO_INC_TANH_OP_H

#include "graph/operator_reg.h"

namespace ge {
REG_OP(Tanh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Tanh)
} // namespace ge

#endif // COMMUNITY_OPS_ADD_OP_PROTO_INC_ADD_OP_H