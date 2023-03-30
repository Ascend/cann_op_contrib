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

#ifndef OP_IR_TEST__OP_TEST_UTIL_H_
#define OP_IR_TEST__OP_TEST_UTIL_H_
#include <iostream>

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "ge/ge_api_types.h"
#include "graph/operator_factory_impl.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "graph/utils/graph_utils.h"

ge::TensorDesc create_desc(std::initializer_list<int64_t> shape_dims,
                           ge::DataType dt=ge::DT_FLOAT);

ge::TensorDesc create_desc_with_ori(std::initializer_list<int64_t> shape_dims,
                                    ge::DataType dt=ge::DT_FLOAT,
                                    ge::Format format=ge::FORMAT_ND,
                                    std::initializer_list<int64_t> ori_shape_dims={},
                                    ge::Format ori_format=ge::FORMAT_ND);
ge::TensorDesc create_desc_shape_range(
    std::initializer_list<int64_t> shape_dims,
    ge::DataType dt,
    ge::Format format,
    std::initializer_list<int64_t> ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range);

ge::TensorDesc create_desc_shape_range(
    const std::vector<int64_t>& shape_dims,
    ge::DataType dt,
    ge::Format format,
    const std::vector<int64_t>& ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range);
ge::TensorDesc create_desc_with_original_shape(std::initializer_list<int64_t> shape_dims,
                                               ge::DataType dt,
                                               ge::Format format,
                                               std::initializer_list<int64_t> ori_shape_dims,
                                               ge::Format ori_format);
ge::TensorDesc create_desc_shape_and_origin_shape_range(
    std::initializer_list<int64_t> shape_dims,
    ge::DataType dt,
    ge::Format format,
    std::initializer_list<int64_t> ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range);

ge::TensorDesc create_desc_shape_and_origin_shape_range(
    const std::vector<int64_t>& shape_dims,
    ge::DataType dt,
    ge::Format format,
    const std::vector<int64_t>& ori_shape_dims,
    ge::Format ori_format,
    std::vector<std::pair<int64_t,int64_t>> shape_range);

/**
 * @brief: Infer shape for all graph nodes in proto unit test cases.
 * @param computeGraphPtr: the graph
 * @return graphStatus: execute result status
 */
ge::graphStatus InferShapeAndType4GraphInProtoUT(ge::ComputeGraphPtr computeGraphPtr);

/**
 * @brief: Get the register function for infer axis type info
 * @param op_type: operator type
 * @return InferAxisTypeInfoFunc: the register function for infer axis type info
 */
ge::InferAxisTypeInfoFunc GetInferAxisTypeFunc(const std::string& op_type);

/**
 * @brief: Get the register function for infer axis slice
 * @param op_type: operator type
 * @return InferAxisSliceFunc: the register function for infer axis slice
 */
ge::InferAxisSliceFunc GetInferAxisSliceFunc(const std::string& op_type);

/*
 * @brief: Create gert::Shape according to std::vector
 * @param shape: std::vector of shape
 * @return gert::Shape: the register function for infer axis type info
 */
gert::Shape CreateShape(const std::vector<int64_t>& shape);

/*
 * @brief: Create gert::StorageShape according to std::vector of ori_shape and shape
 * @param ori_shape: std::vector of original shape
 *            shape: std::vector of storage shape, default is the same as original shape
 * @return gert::Shape: the register function for infer axis type info
 */
gert::StorageShape CreateStorageShape(const std::vector<int64_t>& ori_shape, const std::vector<int64_t>& shape = {});
#endif // OP_IR_TEST__OP_TEST_UTIL_H_
