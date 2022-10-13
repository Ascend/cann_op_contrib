/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

/*!
 * \file pr_roi_pooling.cc
 * \brief tiling function of op
 */
#include "runtime2_util.h"
#include "op_util.h"
#include "register/op_compile_info_base.h"

namespace optiling {
using namespace ge;
using namespace std;

const int64_t TILING_MODE_1 = 1;
const int64_t SHAPE_DIM_0 = 0;
const int64_t SHAPE_DIM_1 = 1;
const int64_t HEIGHT_INDEX = 2;
const int64_t WIDTH_INDEX = 3;

static constexpr size_t INPUT_FEATURES_IDX = 0;
static constexpr size_t INPUT_ROIS_IDX = 1;
static constexpr size_t DIMNUM_2D = 2;
static constexpr size_t DIMNUM_5HD = 5;
static constexpr size_t C0_CONST = 16;

struct TilingPrepare4PrRoIPoolingCompileInfo {
  int32_t block_dim;
};

struct PrRoIPoolingTilingParams {
  int64_t tiling_mode;
  int64_t rois_n;
  int64_t c1_num;
  int64_t in_height;
  int64_t in_width;
  int64_t num_per_core;
  int64_t used_core_num;
  int64_t num_tail_core;
};

void InitPrRoIPoolingParams(PrRoIPoolingTilingParams* params) {
  params->tiling_mode = 1;
  params->rois_n = 1;
  params->c1_num = 1;
  params->in_height = 1;
  params->in_width = 1;
  params->num_per_core = 1;
  params->used_core_num = 1;
  params->num_tail_core = 1;
}

static bool CheckTensorShape(const gert::TilingContext* context, const gert::Shape* x_diff_shape,
                             const gert::Shape* rois_shape) {
  int64_t x_diff_shape_dims = x_diff_shape->GetDimNum();
  int64_t rois_shape_dims = rois_shape->GetDimNum();

  if (x_diff_shape_dims != DIMNUM_5HD) {
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                    "CheckTensorShape, shape of features must be 5, but is [%ld].", x_diff_shape_dims);
    return false;
  }

  if (rois_shape_dims != DIMNUM_2D) {
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "CheckTensorShape, dims of rois must be 2, but is [%ld]",
                                    rois_shape_dims);
    return false;
  }

  return true;
}

static void CalcBlockNum(const int64_t& core_num, const int64_t& rois_n, int64_t& num_per_core, int64_t& used_core_num,
                         int64_t& num_tail_core) {
  num_per_core = (rois_n + core_num - 1) / core_num;
  used_core_num = (rois_n + num_per_core - 1) / num_per_core;
  num_tail_core = rois_n - (used_core_num - 1) * num_per_core;
}

ge::graphStatus Tiling4PrRoIPooling(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "Tiling4PrRoIPooling running.");

  const TilingPrepare4PrRoIPoolingCompileInfo* compile_info =
      reinterpret_cast<const TilingPrepare4PrRoIPoolingCompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  PrRoIPoolingTilingParams* tilingdata = context->GetTilingData<PrRoIPoolingTilingParams>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, tilingdata);

  // get input shape info
  auto input_feature_map = context->GetInputShape(INPUT_FEATURES_IDX);
  OP_TILING_CHECK(input_feature_map == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get input_feature_map failed."),
                  return ge::GRAPH_FAILED);
  const gert::Shape& feature_map_shape = input_feature_map->GetStorageShape();

  auto input_rois = context->GetInputShape(INPUT_ROIS_IDX);
  OP_TILING_CHECK(input_rois == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get input_rois failed."),
                  return ge::GRAPH_FAILED);
  const gert::Shape& rois_shape = input_rois->GetStorageShape();

  int64_t core_num = compile_info->block_dim;
  OP_TILING_CHECK(!CheckTensorShape(context, &feature_map_shape, &rois_shape),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "PrRoIPoolingTiling: params check failed."),
                  return ge::GRAPH_FAILED);
  InitPrRoIPoolingParams(tilingdata);

  int64_t rois_n = rois_shape.GetDim(SHAPE_DIM_0);
  int64_t c1_num = feature_map_shape.GetDim(SHAPE_DIM_1);
  int64_t in_height = feature_map_shape.GetDim(HEIGHT_INDEX);
  int64_t in_width = feature_map_shape.GetDim(WIDTH_INDEX);
  int64_t num_per_core = 1;
  int64_t used_core_num = 1;
  int64_t num_tail_core = 1;

  CalcBlockNum(core_num, rois_n, num_per_core, used_core_num, num_tail_core);

  tilingdata->tiling_mode = TILING_MODE_1;
  tilingdata->rois_n = rois_n;
  tilingdata->c1_num = c1_num;
  tilingdata->in_height = in_height;
  tilingdata->in_width = in_width;
  tilingdata->num_per_core = num_per_core;
  tilingdata->used_core_num = used_core_num;
  tilingdata->num_tail_core = num_tail_core;
  // block_dim, core num used in tik op
  context->SetBlockDim(tilingdata->used_core_num);
  OP_LOGI(context->GetNodeName(), "Tiling4PrRoIPooling run success.");
  OP_LOGD(context->GetNodeName(), "Tiling4PrRoIPooling tiling_data:%s", GetTilingDataString<int64_t>(context).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4PrRoIPooling(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "begin to do TilingPrepare4PrRoIPooling.");
  auto compile_info = GetCompileInfoPtr<TilingPrepare4PrRoIPoolingCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetCompileInfoJson(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);

  const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(vars.empty(), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get vars failed."),
                  return ge::GRAPH_FAILED);

  OP_TILING_CHECK(!ReadCompileItem(vars, "core_num", compile_info->block_dim),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get core_num from compile info faided."),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(PrRoIPooling)
    .Tiling(Tiling4PrRoIPooling)
    .TilingParse<TilingPrepare4PrRoIPoolingCompileInfo>(TilingPrepare4PrRoIPooling);
}  // namespace optiling
