/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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

#include "inc/pr_roi_pooling_op.h"
#include "op_const.h"
#include "op_util.h"
#include "context_util.h"
//#include "utils/util.h"

namespace ge {
// ----------------PrRoIPooling Op Begin-------------------
static bool IsUnknownRankShape(const GeShape& input_shape) {
  return input_shape.IsUnknownDimNum();
}

void MakeUpShapeRange(const ge::GeShape& shape, std::vector<std::pair<int64_t, int64_t>>& range) {
  if (IsUnknownRankShape(shape)) {
    return;
  }

  if (range.empty()) {
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
      int64_t dim = shape.GetDim(i);
      if (dim == -1) {
        range.push_back(std::pair<int64_t, int64_t>(0, -1));
      } else {
        range.push_back(std::pair<int64_t, int64_t>(dim, dim));
      }
    }
  }
}

static bool IsRoiUnknownOutputShape(const GeShape& features_shape, const GeShape& rois_shape) {
  if (IsUnknownRankShape(features_shape) || IsUnknownRankShape(rois_shape)) {
    return true;
  }
  if (features_shape.GetDim(1) == -1 || rois_shape.GetDim(0) == -1) {
    return true;
  }
  return false;
}

IMPLEMT_COMMON_INFERFUNC(PrRoIPoolingInferShape) {
  const size_t POOLED_H_IDX = 2;
  const size_t POOLED_W_IDX = 3;
  const size_t NCHW_DIMENSION_NUM = 4;
  //OP_LOGD(TbeGetName(op).c_str(), "PrRoIPoolingInferShape Begin.");
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto features_desc = op_info->MutableInputDesc(0);
  auto rois_desc = op_info->MutableInputDesc(1);
  const GeShape& features_shape = features_desc->MutableShape();
  const GeShape& rois_shape = rois_desc->MutableShape();
  DataType features_dtype = features_desc->GetDataType();
  auto output_desc = op_info->MutableOutputDesc(0);
  GeShape &output_shape = output_desc->MutableShape();
  int64_t pool_h_shape;
  int64_t pool_w_shape;
  if (!AttrUtils::GetInt(op_info, "pooled_height", pool_h_shape)) {
    //OP_LOGE(TbeGetName(op).c_str(), "PrRoIPoolingInferShape, Get Attr pooled_height failed");
    return GRAPH_FAILED;
  }
  if (!AttrUtils::GetInt(op_info, "pooled_width", pool_w_shape)) {
    //OP_LOGE(TbeGetName(op).c_str(), "PrRoIPoolingInferShape, Get Attr pooled_width failed");
    return GRAPH_FAILED;
  }
  output_desc->SetDataType(features_dtype);
  output_shape.SetDimNum(NCHW_DIMENSION_NUM);
  output_shape.SetDim(POOLED_H_IDX, pool_h_shape);
  output_shape.SetDim(POOLED_W_IDX, pool_w_shape);
  // fixed shape case
  if (!IsRoiUnknownOutputShape(features_shape, rois_shape)) {
    output_shape.SetDim(0, rois_shape.GetDim(0));
    output_shape.SetDim(1, features_shape.GetDim(1));
    return GRAPH_SUCCESS;
  }
  // unknown shape case
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_shape.SetDim(0, -1);
  output_shape.SetDim(1, -1);
  MakeUpShapeRange(output_shape, output_shape_range);
  if (!IsUnknownRankShape(features_shape)) {
    int64_t channel_dim = features_shape.GetDim(1);
    if (channel_dim != -1) {
      output_shape_range[1] = std::pair<int64_t, int64_t>(channel_dim, channel_dim);
      output_shape.SetDim(1, channel_dim);
    } else {
      std::vector<std::pair<int64_t, int64_t>> features_shape_range;
      features_desc->GetShapeRange(features_shape_range);
      output_shape_range[1] = features_shape_range[1];
    }
  }
  if (!IsUnknownRankShape(rois_shape)) {
    int64_t batch_dim = rois_shape.GetDim(0);
    if (batch_dim != -1) {
      output_shape_range[0] = std::pair<int64_t, int64_t>(batch_dim, batch_dim);
      output_shape.SetDim(0, batch_dim);
    } else {
      std::vector<std::pair<int64_t, int64_t>> rois_shape_range;
      rois_desc->GetShapeRange(rois_shape_range);
      output_shape_range[0] = rois_shape_range[0];
    }
  }
  output_desc->SetShapeRange(output_shape_range);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(PrRoIPooling, PrRoIPoolingInferShape);
// ---------------- Op PrRoIPooling End-------------------
}
