#include "inc/tanh_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(OneInOneOutCommonInferShape) {
  static const int64_t input_x_idx = 0;
  static const int64_t output_y_idx = 0;
  if (OneInOneOutDynamicInfer(op, input_x_idx, {output_y_idx})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Tanh, OneInOneOutCommonInferShape);


}