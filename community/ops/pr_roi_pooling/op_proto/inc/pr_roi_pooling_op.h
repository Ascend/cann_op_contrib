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

#ifndef COMMUNITY_OPS_PRROIPOOLING_OP_PROTO_INC_PRROIPOOLING_OP_H
#define COMMUNITY_OPS_PRROIPOOLING_OP_PROTO_INC_PRROIPOOLING_OP_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief In IoUNet, PrRoI Pooling is an integration-based (bilinear interpolation) average 
pooling method for RoI Pooling. 
* It avoids any quantization and has a continuous gradient on bbox coordinates. \n

* @par Inputs:
* Two inputs, including:
* @li features: A 5HD Tensor of type float32 or float16.
* @li rois: ROI position. A 2D Tensor of float32 or float16 with shape (N, 5). "N" indicates the number of ROIs,
the value "5" indicates the indexes of images where the ROIs are located,
* "x0", "y0", "x1", and "y1". \n

* @par Attributes:
* @li pooled_height: A required attribute of type int32, specifying the H dimension.
* @li pooled_width: A required attribute of type int32, specifying the W dimension.
* @li spatial_scale: A required attribute of type float32, specifying the scaling ratio of "
features" to the original image. \n

* @par Outputs:
* y: Outputs the feature sample of each ROI position. The format is 5HD Tensor of type float32 or float16.
The axis N is the number of input ROIs. Axes H, W, and C are consistent
* with the values of "pooled_height",
* "pooled_width", and "features", respectively.
*/
REG_OP(PrRoIPooling)
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(pooled_height, Int)
    .REQUIRED_ATTR(pooled_width, Int)
    .REQUIRED_ATTR(spatial_scale, Float)
    .OP_END_FACTORY_REG(PrRoIPooling)
} // namespace ge

#endif // COMMUNITY_OPS_PRROIPOOLING_OP_PROTO_INC_PRROIPOOLING_OP_H