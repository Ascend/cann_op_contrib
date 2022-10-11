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