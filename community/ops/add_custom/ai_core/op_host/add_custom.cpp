/**
 * Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
struct TilingCompileInfo {
    int64_t ub_size;
};

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddCustomTilingData tiling;
    context->SetBlockDim(8);
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    tiling.set_blockDim(8);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static int32_t CheckOpSupport(const ge::Operator &op, ge::AscendString &result)
{
    std::string res_json_str = "{\"ret_code\": \"0\",\"reason\": \"check_supported_stub\"}";
    result = ge::AscendString(res_json_str.c_str());
    return 1;
}
} // namespace optiling

namespace ge {
ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeRange(gert::InferShapeRangeContext* context)
{
    const gert::Range<gert::Shape>* x1_shape_range = context->GetInputShapeRange(0);
    gert::Range<gert::Shape>* y_shape_range = context->GetOutputShapeRange(0);
    *y_shape_range = *x1_shape_range;
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const ge::DataType x1_datatype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x1_datatype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferShapeRange(ge::InferShapeRange)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .SetTilingParse(optiling::TilingPrepare)
            .SetCheckSupport(optiling::CheckOpSupport);

        OpAICoreConfig aicConfig;
        aicConfig.AsyncFlag(true)
            .DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(true)
            .PrecisionReduceFlag(true)
            .RangeLimitValue("limited");
        this->AICore().AddConfig("ascend910", aicConfig);
    }
};

OP_ADD(AddCustom, optiling::TilingCompileInfo);
} // namespace ops
