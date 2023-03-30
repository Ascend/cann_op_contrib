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

#include "op_proto_test_util.h"
#include "graph/utils/node_utils_ex.h"

ge::TensorDesc create_desc(std::initializer_list<int64_t> shape_dims,
                           ge::DataType dt) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);
  return tensorDesc;
}

ge::TensorDesc create_desc_with_ori(std::initializer_list<int64_t> shape_dims,
                                    ge::DataType dt,
                                    ge::Format format,
                                    std::initializer_list<int64_t> ori_shape_dims,
                                    ge::Format ori_format) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ori_format);
  return tensorDesc;
}

ge::TensorDesc create_desc_with_original_shape(std::initializer_list<int64_t> shape_dims,
                                               ge::DataType dt,
                                               ge::Format format,
                                               std::initializer_list<int64_t> ori_shape_dims,
                                               ge::Format ori_format) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(ori_shape);
  tensorDesc.SetOriginFormat(ori_format);
  return tensorDesc;
}

ge::TensorDesc create_desc_shape_range(
    std::initializer_list<int64_t> shape_dims,
    ge::DataType dt,
    ge::Format format,
    std::initializer_list<int64_t> ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ori_format);
  tensorDesc.SetShapeRange(shape_range);
  return tensorDesc;
}

ge::TensorDesc create_desc_shape_range(
    const std::vector<int64_t>& shape_dims,
    ge::DataType dt,
    ge::Format format,
    const std::vector<int64_t>& ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ori_format);
  tensorDesc.SetShapeRange(shape_range);
  return tensorDesc;
}

ge::TensorDesc create_desc_shape_and_origin_shape_range(
    const std::vector<int64_t>& shape_dims,
    ge::DataType dt,
    ge::Format format,
    const std::vector<int64_t>& ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(ori_shape);
  tensorDesc.SetOriginFormat(ori_format);
  tensorDesc.SetShapeRange(shape_range);
  return tensorDesc;
}

ge::TensorDesc create_desc_shape_and_origin_shape_range(
    std::initializer_list<int64_t> shape_dims,
    ge::DataType dt,
    ge::Format format,
    std::initializer_list<int64_t> ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range) {
  ge::TensorDesc tensorDesc;
  ge::Shape shape(shape_dims);
  ge::Shape ori_shape(ori_shape_dims);
  tensorDesc.SetDataType(dt);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(format);
  tensorDesc.SetOriginShape(ori_shape);
  tensorDesc.SetOriginFormat(ori_format);
  tensorDesc.SetShapeRange(shape_range);
  return tensorDesc;
}

ge::graphStatus InferShapeAndType4GraphInProtoUT(ge::ComputeGraphPtr computeGraphPtr) {
  computeGraphPtr->TopologicalSorting();
  for (auto nodePtr: computeGraphPtr->GetAllNodes()) {
    if (nodePtr->GetType() != "Data") {
      auto verifyStatus = ge::NodeUtilsEx::Verify(nodePtr);
      if (verifyStatus != ge::GRAPH_SUCCESS) {
        std::cout << "Graph Infer failed, " << nodePtr->GetName()
                  << "'s Verify() failed" << std::endl;
        return verifyStatus;
      }
      auto inferFormatStatus = ge::NodeUtilsEx::InferOriginFormat(nodePtr);
      if (inferFormatStatus != ge::GRAPH_SUCCESS) {
        std::cout << "Graph Infer failed, " << nodePtr->GetName()
                  << "'s InferOriginFormat() failed" << std::endl;
        return inferFormatStatus;
      }
      if (nodePtr->GetType() != "NetOutput") {
        auto inferShapeStatus = ge::NodeUtilsEx::InferShapeAndType(nodePtr);
        if (inferShapeStatus != ge::GRAPH_SUCCESS) {
          std::cout << "Graph Infer failed, " << nodePtr->GetName()
                    << "'s inferShapeAndType() failed" << std::endl;
          return inferShapeStatus;
        }
      }
    }
    for (auto outDataAnchor : nodePtr->GetAllOutAnchors()) {
      int outIdx = outDataAnchor->GetIdx();
      auto output_desc = nodePtr->GetOpDesc()->MutableOutputDesc(outIdx);
      if (output_desc->GetOriginShape().GetShapeSize() == 0 and output_desc->GetShape().GetShapeSize() != 0) {
        output_desc->SetOriginFormat(output_desc->GetFormat());
        output_desc->SetOriginShape(output_desc->GetShape());
      }
      for (auto anchor : outDataAnchor->GetPeerAnchors()) {
        int idx = anchor->GetIdx();
        auto update_status = anchor->GetOwnerNode()->GetOpDesc()
            ->UpdateInputDesc(idx, nodePtr->GetOpDesc()->GetOutputDesc(outIdx));
        if (update_status != ge::GRAPH_SUCCESS) {
          std::cout << "Graph Infer failed, update " << anchor->GetOwnerNode()->GetName()
                    << "'s input failed" << std::endl;
          return update_status;
        }
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::InferAxisTypeInfoFunc GetInferAxisTypeFunc(const std::string& op_type) {
  return ge::OperatorFactoryImpl::GetInferAxisTypeInfoFunc(op_type);
}

ge::InferAxisSliceFunc GetInferAxisSliceFunc(const std::string& op_type) {
  return ge::OperatorFactoryImpl::GetInferAxisSliceFunc(op_type);
}

gert::Shape CreateShape(const std::vector<int64_t> &shape) {
  gert::Shape gert_shape;
  gert_shape.SetDimNum(shape.size());
  for (size_t i = 0; i < gert_shape.GetDimNum(); ++i) {
    gert_shape.SetDim(i, shape[i]);
  }

  return gert_shape;
}

gert::StorageShape CreateStorageShape(const std::vector<int64_t> &ori_shape, const std::vector<int64_t> &shape) {
  gert::StorageShape storage_shape;
  auto &gert_ori_shape = storage_shape.MutableOriginShape();
  auto &gert_storage_shape = storage_shape.MutableStorageShape();

  gert_ori_shape = CreateShape(ori_shape);
  if (shape.empty()) {
    gert_storage_shape = CreateShape(ori_shape);
  } else {
    gert_storage_shape = CreateShape(shape);
  }

  return storage_shape;
}
