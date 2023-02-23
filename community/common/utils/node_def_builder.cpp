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

#include "node_def_builder.h"
#include "cpu_kernel_utils.h"

using namespace std;

namespace aicpu {
std::shared_ptr<NodeDef> NodeDefBuilder::CreateNodeDef() {
	return CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
}

NodeDefBuilder::NodeDefBuilder(NodeDef *nodeDef, std::string name, std::string opName) {
	nodeDef_ = nodeDef;
	name_ = name;
	nodeDef_->SetOpType(opName);
}

void NodeDefBuilder::BuildNodeFromInputOutputNode(const InputOutputNode& node, bool isInput) {
	std::shared_ptr<Tensor> tensor;
	if (isInput) {
		tensor = nodeDef_->AddInputs();
	} else {
		tensor = nodeDef_->AddOutputs();
	}
	aicpu::CpuKernelUtils::SetTensorName(node.node, tensor);
	tensor->SetDataType(node.dType);
	auto shape = tensor->GetTensorShape();
	shape->SetDimSizes(node.dims);
	shape->SetFormat(node.format);
	int64_t dataSize = 1;
	for (size_t i = 0; i < node.dims.size(); i++) {
		dataSize = dataSize * node.dims[i];
	}
	dataSize = dataSize * GetSizeByDataType(node.dType);
	if (node.dims.empty()) {
		dataSize = GetSizeByDataType(node.dType);
	}
	if (node.data == nullptr) {
		dataSize = 0;
	}
	tensor->SetDataSize(static_cast<uint64_t>(dataSize));
	tensor->SetData(node.data);
}

NodeDefBuilder& NodeDefBuilder::Input(const InputOutputNode& input) {
	BuildNodeFromInputOutputNode(input, true);
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Output(const InputOutputNode& output) {
	BuildNodeFromInputOutputNode(output, false);
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, int32_t value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetInt(value);
	(void)nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, int64_t value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetInt(value);
	nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, float value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetFloat(value);
	nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, double value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetFloat(static_cast<float>(value));
	nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, bool value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetBool(value);
	nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, aicpu::DataType value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetDataType(value);
	nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<bool> &value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetListBool(value);
	nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::string &value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetString(value);
	nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<std::string> &value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetListString(value);
	nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<int64_t> &value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetListInt(value);
	nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<std::vector<int64_t>> &value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetListListInt(value);
	nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<float> &value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetListFloat(value);
	nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<aicpu::DataType> &value) {
	auto attr = CpuKernelUtils::CreateAttrValue();
	attr->SetListDataType(value);
	nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<int64_t> &dims, std::string type) {
	if (type == "shape") {
		auto shape = CpuKernelUtils::CreateAttrValue();
		auto value = CpuKernelUtils::CreateTensorShape();
		value->SetDimSizes(dims);
		(void)nodeDef_->AddAttrs(name, shape.get());
		(void)shape->SetTensorShape(value.get());
	}
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<std::vector<int64_t>> &shapeLists,
                                     std::string type) {
	if (type == "shape_list") {
		auto shapeItems = CpuKernelUtils::CreateAttrValue();
		for (size_t i = 0; i < shapeLists.size(); i++) {
			auto value = shapeItems->AddListTensorShape();
			value->SetDimSizes(shapeLists[i]);
		}
		(void)nodeDef_->AddAttrs(name, shapeItems.get());
	}
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, aicpu::Tensor *tensor) {
	auto attr = CpuKernelUtils::CreateAttrValue();
    (void)attr->SetTensor(tensor);
    (void)nodeDef_->AddAttrs(name, attr.get());
	return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, std::vector<aicpu::Tensor *> &tensors) {
	auto attr = CpuKernelUtils::CreateAttrValue();
    (void)attr->SetListTensor(tensors);
    (void)nodeDef_->AddAttrs(name, attr.get());
	return *this;
}
}