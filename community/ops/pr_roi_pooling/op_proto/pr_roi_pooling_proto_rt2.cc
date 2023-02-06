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

#include "inc/pr_roi_pooling_op.h"
#include "register/op_impl_registry.h"
#include "op_const.h"
#include "op_util.h"
#include "context_util.h"


namespace ge {
static constexpr size_t INPUT_FEATURES_IDX = 0;
static constexpr size_t INPUT_ROIS_IDX = 1;
static constexpr size_t OUTPUT_Y_IDX = 0;
static constexpr size_t ROIALIGN_DIM_SIZE = 4;

static constexpr size_t OUTPUT_DIM0 = 0;
static constexpr size_t OUTPUT_DIM1 = 1;
static constexpr size_t OUTPUT_DIM2 = 2;
static constexpr size_t OUTPUT_DIM3 = 3;
static constexpr size_t ALIGN_ATTR_H_IDX = 1;
static constexpr size_t ALIGN_ATTR_W_IDX = 2;
// PrRoiPooling
static constexpr size_t POOLING_ATTR_H_IDX = 0;
static constexpr size_t POOLING_ATTR_W_IDX = 1;
// ROIAlignGrad
static constexpr size_t ATT_STRIDES_IDX = 0;
static constexpr size_t OUTPUT_XDIFF_IDX = 0;

ge::graphStatus RoiInferSahpe(gert::InferShapeContext* context, size_t pooled_h_attr_idx, size_t pooled_w_attr_idx) {
  const gert::Shape* input_features_shape = context->GetInputShape(INPUT_FEATURES_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_features_shape);
  const gert::Shape* input_rois_shape = context->GetInputShape(INPUT_ROIS_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_rois_shape);

  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t* pool_h_shape = attrs->GetAttrPointer<int64_t>(pooled_h_attr_idx);
  OPS_CHECK_NULL_WITH_CONTEXT(context, pool_h_shape);
  const int64_t* pool_w_shape = attrs->GetAttrPointer<int64_t>(pooled_w_attr_idx);
  OPS_CHECK_NULL_WITH_CONTEXT(context, pool_w_shape);

  gert::Shape* output_shape = context->GetOutputShape(OUTPUT_Y_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_shape);

  output_shape->SetDimNum(ROIALIGN_DIM_SIZE);
  output_shape->SetDim(OUTPUT_DIM0, input_rois_shape->GetDim(0));
  output_shape->SetDim(OUTPUT_DIM1, input_features_shape->GetDim(1));
  output_shape->SetDim(OUTPUT_DIM2, *pool_h_shape);
  output_shape->SetDim(OUTPUT_DIM3, *pool_w_shape);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShape4ROIAlign(gert::InferShapeContext* context) {
  return RoiInferSahpe(context, ALIGN_ATTR_H_IDX, ALIGN_ATTR_W_IDX);
}

ge::graphStatus InferShape4PrRoIPooling(gert::InferShapeContext* context) {
  return RoiInferSahpe(context, POOLING_ATTR_H_IDX, POOLING_ATTR_W_IDX);
}

ge::graphStatus InferShape4ROIAlignGrad(gert::InferShapeContext* context) {
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const gert::ContinuousVector* strides_ptr = attrs->GetAttrPointer<gert::ContinuousVector>(ATT_STRIDES_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, strides_ptr);

  gert::Shape* output_shape = context->GetOutputShape(OUTPUT_XDIFF_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_shape);

  int32_t dim_size = strides_ptr->GetSize();
  output_shape->SetDimNum(dim_size);
  const int64_t* strides_array = reinterpret_cast<const int64_t*>(strides_ptr->GetData());
  for (int32_t i = 0; i < dim_size; i++) {
    output_shape->SetDim(i, strides_array[i]);
  }

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(PrRoIPooling).InferShape(InferShape4PrRoIPooling);
}
