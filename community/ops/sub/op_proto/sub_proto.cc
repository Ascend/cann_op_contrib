#include "inc/sub_op.h"
#include "register/op_impl_registry.h"
#include "register/infer_axis_slice_registry.h"
#include "register/infer_data_slice_registry.h"
#include "utils/util.h"
#include "graph/ge_attr_value.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
static void InferElewiseTwoInput(vector<vector<int64_t>>& in_data_slice, const vector<vector<int64_t>> out_data_slice,
                                 const vector<int64_t> in_dims, const vector<int64_t> out_dims) {
  if (in_dims.size() == out_dims.size()) {
    for (size_t i = 0UL; i < in_dims.size(); i++) {
      if (in_dims[i] == 1) {
        in_data_slice.push_back({0, 1});
      } else {
        in_data_slice.push_back(out_data_slice[i]);
      }
    }
  } else {
    for (size_t i = 0; i < in_dims.size(); i++) {
      if (in_dims[i] == 1) {
        in_data_slice.push_back({0, 1});
      } else {
        in_data_slice.push_back(out_data_slice[out_dims.size() - in_dims.size() + i]);
      }
    }
  }
}

static void InferElewiseTwoInputdif(vector<vector<int64_t>>& in_data_slice,
                                    const vector<vector<int64_t>> out_data_slice,
                                    const vector<int64_t> in_dims, const vector<int64_t> out_dims,
                                    const int64_t aixs) {
  if (in_dims.size() == out_dims.size()) {
    for (size_t i = 0UL; i < in_dims.size(); i++) {
      if (in_dims[i] == 1) {
        in_data_slice.push_back({0, 1});
      } else {
        in_data_slice.push_back(out_data_slice[i]);
      }
    }
  } else if (in_dims.size() == 1 && in_dims[0] != 1) {
    in_data_slice.push_back({out_data_slice[aixs][0] * 16, out_data_slice[aixs][1] * 16});
  }
}

IMPLEMT_COMMON_INFER_DATA_SLICE(ElewiseTwoInputInferDataSlice) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (!op_desc) {
    OP_LOGW(TbeGetName(op).c_str(), "GetOpDescFromOperator failed.");
    return GRAPH_FAILED;
  }

  auto tensor_desc_in_x1 = op_desc->MutableInputDesc("x1");
  if (!tensor_desc_in_x1) {
    OP_LOGW(TbeGetName(op).c_str(), "Get input desc x1 failed.");
    return GRAPH_FAILED;
  }
  auto x1_shape = tensor_desc_in_x1->MutableShape();
  auto x1_format = tensor_desc_in_x1->GetFormat();
  std::vector<int64_t> x1_dims = x1_shape.GetDims();

  auto tensor_desc_in_x2 = op_desc->MutableInputDesc("x2");
  if (!tensor_desc_in_x2) {
    OP_LOGW(TbeGetName(op).c_str(), "Get input desc x2 failed.");
    return GRAPH_FAILED;
  }
  auto x2_shape = tensor_desc_in_x2->MutableShape();
  auto x2_format = tensor_desc_in_x2->GetFormat();
  std::vector<int64_t> x2_dims = x2_shape.GetDims();

  auto tensor_desc_out_y = op_desc->MutableOutputDesc("y");
  if (!tensor_desc_out_y) {
    OP_LOGW(TbeGetName(op).c_str(), "Get input desc y failed.");
    return GRAPH_FAILED;
  }
  auto y_shape = tensor_desc_out_y->MutableShape();
  std::vector<int64_t> y_dims = y_shape.GetDims();

  vector<vector<int64_t>> y_data_slice = {};
  vector<vector<int64_t>> x1_data_slice = {};
  vector<vector<int64_t>> x2_data_slice = {};
  if (!ge::AttrUtils::GetListListInt(tensor_desc_out_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGW(TbeGetName(op).c_str(), "no data slice, use default as {}");
    return GRAPH_FAILED;
  }

if ((x1_format == FORMAT_NHWC and x2_format == FORMAT_ND) or (x1_format == FORMAT_ND and x2_format == FORMAT_NHWC) or
    (x1_format == x2_format)) {
    InferElewiseTwoInput(x1_data_slice, y_data_slice, x1_dims, y_dims);
    InferElewiseTwoInput(x2_data_slice, y_data_slice, x2_dims, y_dims);
  } else {
    if ((x1_format == FORMAT_NC1HWC0 && x2_dims.size() <= 1) ||
        (x1_dims.size() <= 1 && x2_format == FORMAT_NC1HWC0)) {
      // 5HD+ND
      InferElewiseTwoInputdif(x1_data_slice, y_data_slice, x1_dims, y_dims, 1);
      InferElewiseTwoInputdif(x2_data_slice, y_data_slice, x2_dims, y_dims, 1);
    } else if ((x1_format == FORMAT_FRACTAL_NZ && x2_dims.size() <= 1) ||
               (x1_dims.size() <= 1 && x2_format == FORMAT_FRACTAL_NZ)) {
      // NZ+ND
      InferElewiseTwoInputdif(x1_data_slice, y_data_slice, x1_dims, y_dims, y_dims.size() - 3);
      InferElewiseTwoInputdif(x2_data_slice, y_data_slice, x2_dims, y_dims, y_dims.size() - 3);
    } else if ((x1_format == FORMAT_FRACTAL_Z && x2_dims.size() <= 1) ||
               (x1_dims.size() <= 1 && x2_format == FORMAT_FRACTAL_Z)) {
      // F_Z+ND
      InferElewiseTwoInputdif(x1_data_slice, y_data_slice, x1_dims, y_dims, 0);
      InferElewiseTwoInputdif(x2_data_slice, y_data_slice, x2_dims, y_dims, 0);
    } else {
      x1_data_slice.assign(x1_dims.size(), {});
      x2_data_slice.assign(x2_dims.size(), {});
    }
  }

  if (!ge::AttrUtils::SetListListInt(tensor_desc_in_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice)) {
    OP_LOGW(TbeGetName(op).c_str(), "data slice set failed");
    return GRAPH_FAILED;
  }
  if (!ge::AttrUtils::SetListListInt(tensor_desc_in_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice)) {
    OP_LOGW(TbeGetName(op).c_str(), "data slice set failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TwoInOneOutCommonInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, 0, 1, 0, is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
INFER_DATA_SLICE_FUNC_REG(Sub, ElewiseTwoInputInferDataSlice);
COMMON_INFER_FUNC_REG(Sub, TwoInOneOutCommonInferShape);
INFER_AXIS_TYPE_INFO_REG(Sub, InferAxisType4BroadcastOp);


}