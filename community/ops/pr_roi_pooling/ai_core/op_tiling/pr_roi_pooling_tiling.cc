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
#include "utils/tiling_util.h"
//#include "op_util.h"
// #include "register/op_compile_info_base.h"


namespace optiling {
using namespace ge;
using namespace std;

// A. block tiling: indices tiling
// params is not cache
const int64_t TILING_MODE_1 = 1;
const int64_t HEIGHT_INDEX = 2;
const int64_t WIDTH_INDEX = 3;
const int64_t ROI_DIMS = 2;
const int64_t ROI_ROW_LEN = 5;

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

void InitPrRoIPoolingParams(PrRoIPoolingTilingParams& params) {
  params.tiling_mode = 1;
  params.rois_n = 1;
  params.c1_num = 1;
  params.in_height = 1;
  params.in_width = 1;
  params.num_per_core = 1;
  params.used_core_num = 1;
  params.num_tail_core = 1;
}

void SetPrRoIPoolingParams(const PrRoIPoolingTilingParams& Params, utils::OpRunInfo& runInfo) {
  // set tiling data
  runInfo.AddTilingData(Params.tiling_mode);
  runInfo.AddTilingData(Params.rois_n);
  runInfo.AddTilingData(Params.c1_num);
  runInfo.AddTilingData(Params.in_height);
  runInfo.AddTilingData(Params.in_width);
  runInfo.AddTilingData(Params.num_per_core);
  runInfo.AddTilingData(Params.used_core_num);
  runInfo.AddTilingData(Params.num_tail_core);
}

/*
void PrintPrRoIPoolingParams(const PrRoIPoolingTilingParams& params) {
  OP_LOGD("[PrRoIPoolingTiling]", "tiling_mode=%ld.", params.tiling_mode);
  OP_LOGD("[PrRoIPoolingTiling]", "rois_n=%ld.", params.rois_n);
  OP_LOGD("[PrRoIPoolingTiling]", "c1_num=%ld.", params.c1_num);
  OP_LOGD("[PrRoIPoolingTiling]", "in_height=%ld.", params.in_height);
  OP_LOGD("[PrRoIPoolingTiling]", "in_width=%ld.", params.in_width);
  OP_LOGD("[PrRoIPoolingTiling]", "num_per_core=%ld.", params.num_per_core);
  OP_LOGD("[PrRoIPoolingTiling]", "used_core_num=%ld.", params.used_core_num);
  OP_LOGD("[PrRoIPoolingTiling]", "num_tail_core=%ld.", params.num_tail_core);
}
*/

static bool CheckTensorShape(const std::string& opType, GeShape& x_diff_shape, GeShape& rois_shape) {
  int64_t x_diff_shape_dims = x_diff_shape.GetDimNum();
  int64_t rois_shape_dims = rois_shape.GetDimNum();
  if (x_diff_shape_dims != ROI_ROW_LEN) {
    //VECTOR_INNER_ERR_REPORT_TILIING(opType,
    //                                "op [PrRoIPoolingTiling] : CheckTensorShape, shape of features check failed.");
    return false;
  }
  if (rois_shape_dims != ROI_DIMS) {
    //VECTOR_INNER_ERR_REPORT_TILIING(opType, "op [PrRoIPoolingTiling] : CheckTensorShape, dims of rois must be 2.");
    return false;
  }

  return true;
}

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num"};

static void CalcBlockNum(const int64_t& core_num, const int64_t& rois_n, int64_t& num_per_core, int64_t& used_core_num,
                         int64_t& num_tail_core) {
  num_per_core = (rois_n + core_num - 1) / core_num;
  used_core_num = (rois_n + num_per_core - 1) / num_per_core;
  num_tail_core = rois_n - (used_core_num - 1) * num_per_core;
}

bool PrRoIPoolingTiling(const std::string& opType, const ge::Operator& opParas, const std::vector<int64_t>& op_info,
                        utils::OpRunInfo& runInfo) {
  //OP_LOGD("op[%s] PrRoIPoolingTiling running.", opType.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(opParas);
  //OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_info failed."),
  //                return false);

  auto features_desc = operator_info->MutableInputDesc(0);
  //OP_TILING_CHECK(features_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get features_desc failed."),
  //                return false);
  GeShape& feature_map_shape = features_desc->MutableShape();

  auto rois_desc = operator_info->MutableInputDesc(1);
  //OP_TILING_CHECK(rois_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get rois_desc failed."),
  //                return false);
  GeShape& rois_shape = rois_desc->MutableShape();

  // get compile info
  //OP_TILING_CHECK(COMPILE_INFO_KEY.size() != op_info.size(),
  //                VECTOR_INNER_ERR_REPORT_TILIING(opType, "parse op_info failed."), return false);
  if (COMPILE_INFO_KEY.size() != op_info.size()) {
    return false;
  }

  int64_t core_num = op_info[0];
  //OP_TILING_CHECK(core_num <= 0,
  //                VECTOR_INNER_ERR_REPORT_TILIING(opType, "get invalid core_num."), return false);
  if (core_num <= 0) {
    return false;
  }

  bool flag = true;
  flag = CheckTensorShape(opType, feature_map_shape, rois_shape);
  if (!flag) {
    //VECTOR_INNER_ERR_REPORT_TILIING(opType, "PrRoIPoolingTiling: params check failed.");
    return false;
  }

  PrRoIPoolingTilingParams runParams;
  InitPrRoIPoolingParams(runParams);

  int64_t rois_n = rois_shape.GetDim(0);
  //OP_TILING_CHECK(rois_n <= 0, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get invalid rois_n."), return false);
  if (rois_n <= 0) {
    return false;
  }
  int64_t c1_num = feature_map_shape.GetDim(1);
  int64_t x_width = feature_map_shape.GetDim(WIDTH_INDEX);
  runParams.rois_n = rois_n;
  runParams.c1_num = c1_num;
  runParams.in_width = x_width;
  runParams.in_height = feature_map_shape.GetDim(HEIGHT_INDEX);
  runParams.tiling_mode = 1;

  int64_t num_per_core = 1;
  int64_t real_core_num = 1;
  int64_t num_tail_core = 1;
  CalcBlockNum(core_num, rois_n, num_per_core, real_core_num, num_tail_core);
  runParams.num_per_core = num_per_core;
  runParams.used_core_num = real_core_num;
  runParams.num_tail_core = num_tail_core;


  SetPrRoIPoolingParams(runParams, runInfo);
  //PrintPrRoIPoolingParams(runParams);

  // block_dim, core num used in tik op
  runInfo.SetBlockDim(runParams.used_core_num);
  //OP_LOGI("op[%s] tiling run success.", opType.c_str());

  return true;
}
// register tiling interface of the PrRoIPooling op.
REGISTER_OP_TILING_V3_WITH_VECTOR(PrRoIPooling, PrRoIPoolingTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling

