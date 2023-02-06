'''
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
'''
import math
import numpy as np


class PrRoIPoolingNp:
    def __init__(self, features, rois, pooled_height, pooled_width, spatial_scale):
        self.features = features
        self.rois = rois
        self.rois_n = rois.shape[0]
        self.channels = features.shape[1]
        self.in_height = features.shape[2]
        self.in_width = features.shape[3]
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.spatial_scale = spatial_scale

    def pr_roi_pooling_compute(self):
        roi_start_w_arr = self.rois[:, 1] * self.spatial_scale
        roi_start_h_arr = self.rois[:, 2] * self.spatial_scale
        roi_end_w_arr = self.rois[:, 3] * self.spatial_scale
        roi_end_h_arr = self.rois[:, 4] * self.spatial_scale
        roi_width_arr = np.maximum(roi_end_w_arr - roi_start_w_arr, 0.0)
        roi_height_arr = np.maximum(roi_end_h_arr - roi_start_h_arr, 0.0)
        bin_size_h_arr = roi_height_arr / self.pooled_height
        bin_size_w_arr = roi_width_arr / self.pooled_width
        win_size_arr = np.maximum(bin_size_w_arr * bin_size_h_arr, 0.0)

        output_tensor = np.zeros(
            [self.rois_n, self.channels, self.pooled_height, self.pooled_width], "float32")

        for roi_idx in range(self.rois_n):
            roi_start_w = roi_start_w_arr[roi_idx]
            roi_start_h = roi_start_h_arr[roi_idx]
            bin_size_w = bin_size_w_arr[roi_idx]
            bin_size_h = bin_size_h_arr[roi_idx]
            win_size = win_size_arr[roi_idx]
            batch_idx = int(self.rois[roi_idx, 0])
            if (win_size == 0):
                continue
            for i in range(self.pooled_height):
                for j in range(self.pooled_width):
                    # 表示池化后像素点需要在原roi中做积分的区域
                    win_start_h = roi_start_h + bin_size_h * i
                    win_start_w = roi_start_w + bin_size_w * j
                    win_end_h = win_start_h + bin_size_h
                    win_end_w = win_start_w + bin_size_w
                    # 取整后取浮点数
                    start_w = math.floor(win_start_w)
                    end_w = math.ceil(win_end_w)
                    start_h = math.floor(win_start_h)
                    end_h = math.ceil(win_end_h)
                    # 声明输出
                    sum_out = np.zeros([self.channels], "float32")
                    for h_iter in range(start_h, end_h):
                        for w_iter in range(start_w, end_w):
                            sum_out += self.mat_calculation(batch_idx, h_iter, w_iter, h_iter + 1, w_iter + 1,
                                                            max(win_start_h, float(h_iter)),
                                                            max(win_start_w, float(w_iter)),
                                                            min(win_end_h, float(h_iter) + 1.0),
                                                            min(win_end_w, float(w_iter + 1.0)))
                    output_tensor[roi_idx, :, i, j] = sum_out / win_size

        return output_tensor

    def mat_calculation(self, batch_idx, s_h, s_w, e_h, e_w, y0, x0, y1, x1):
        alpha = x0 - float(s_w)
        beta = y0 - float(s_h)
        lim_alpha = x1 - float(s_w)
        lim_beta = y1 - float(s_h)
        tmp = (lim_alpha - 0.5 * lim_alpha * lim_alpha - alpha + 0.5 * alpha *
               alpha) * (lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        sum_out = self.get_feature_data(batch_idx, s_h, s_w) * tmp

        alpha = float(e_w) - x1
        lim_alpha = float(e_w) - x0
        tmp = (lim_alpha - 0.5 * lim_alpha * lim_alpha - alpha + 0.5 * alpha *
               alpha) * (lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        sum_out += self.get_feature_data(batch_idx, s_h, e_w) * tmp

        alpha = x0 - float(s_w)
        beta = float(e_h) - y1
        lim_alpha = x1 - float(s_w)
        lim_beta = float(e_h) - y0
        tmp = (lim_alpha - 0.5 * lim_alpha * lim_alpha - alpha + 0.5 * alpha *
               alpha) * (lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        sum_out += self.get_feature_data(batch_idx, e_h, s_w) * tmp

        alpha = float(e_w) - x1
        lim_alpha = float(e_w) - x0
        tmp = (lim_alpha - 0.5 * lim_alpha * lim_alpha - alpha + 0.5 * alpha *
               alpha) * (lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        sum_out += self.get_feature_data(batch_idx, e_h, e_w) * tmp

        return sum_out

    def get_feature_data(self, batch_idx, h, w):
        overflow = ((h < 0) or (w < 0) or (h >= self.in_height) or (w >= self.in_width))
        retVal = 0.0 if overflow else self.features[batch_idx, :, h, w]
        return retVal


def calc_expect_func(features,
                     rois,
                     y,
                     pooled_height,
                     pooled_width,
                     spatial_scale):
    obj = PrRoIPoolingNp(
        features["value"], rois["value"], pooled_height, pooled_width, spatial_scale)
    res = obj.pr_roi_pooling_compute()
    return [res]
