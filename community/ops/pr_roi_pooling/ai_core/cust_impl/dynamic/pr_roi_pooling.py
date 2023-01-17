# Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
pr_roi_pooling
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


class Constant:
    """
    The class for constant
    """
    # max int64
    MAX_INT64 = 2**64 - 1
    TILING_ARG_NUM = 16
    # C0 size
    C0_SIZE = 16
    # batch size
    BATCH_SIZE = 64
    # one block size takes up 32b
    BLOCK_SIZE = 32
    # 1 block has 32// 4 fp32 elements
    FP32_BLOCK_ELES = BLOCK_SIZE // 4
    INT64_BLOCK_ELES = BLOCK_SIZE // 8
    FP32_VEC_REPEAT_MASK = FP32_BLOCK_ELES * 8
    TILING_MODE_1 = 1
    # roi elements nums
    ROI_ELE_NUMS = 5
    ZERO = 0.0
    # reserved ub size
    RESERVED_UB_SIZE = 10240
    # data_move stride limit [0,65535]
    STRIDE_MAX = 65535


class PrRoIPooling():
    """
    PrRoIPooling init
    """
    def __init__(self, pooled_height, pooled_width, spatial_scale, kernel_name):
        self.tik_instance = tik.Tik()
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.tiling_dtype = "int64"
        self.dtype = "float32"
        self.pooled_width = pooled_width
        self.pooled_height = pooled_height
        self.spatial_scale = spatial_scale
        self.kernel_name = kernel_name
        self.unknown_max_shape = [Constant.MAX_INT64]
        self.opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True, "dynamic_tik": True}
        self.features_gm = self.tik_instance.Tensor(self.dtype,
                                                    self.unknown_max_shape,
                                                    name="features_gm",
                                                    scope=tbe_platform.scope_gm)
        self.rois_gm = self.tik_instance.Tensor(self.dtype,
                                                self.unknown_max_shape,
                                                name="rois_gm",
                                                scope=tbe_platform.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype,
                                                  self.unknown_max_shape,
                                                  name="output_gm",
                                                  scope=tbe_platform.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.num_tail_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="num_tail_core")
        self.used_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="used_core_num")
        self.num_per_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="num_per_core")
        self.in_width = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="in_width")
        self.in_height = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="in_height")
        self.c1_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="c1_num")
        self.rois_n = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="rois_n")
        self.tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tiling_mode")
        self.rois_valid = self.tik_instance.Scalar(dtype="int32", name="rois_valid")
        self.rois_valid_in_block = self.tik_instance.Scalar(dtype="int32", init_value=Constant.BATCH_SIZE)
        self.move_out_idx = self.tik_instance.Scalar(dtype="int32", name="move_out_idx")
        self.fetures_index = self.tik_instance.Scalar(dtype="int32", name="fetures_index")
        self.vec_cal_size = self.tik_instance.Scalar(dtype="int32", name="vec_cal_size")
        self.roi_start_w = self.tik_instance.Scalar(dtype="float32")
        self.roi_start_h = self.tik_instance.Scalar(dtype="float32")
        self.bin_size_w = self.tik_instance.Scalar(dtype="float32")
        self.bin_size_h = self.tik_instance.Scalar(dtype="float32")
        self.win_size = self.tik_instance.Scalar(dtype="float32")
        self.win_start_w = self.tik_instance.Scalar(dtype="float32")
        self.win_start_h = self.tik_instance.Scalar(dtype="float32")
        self.win_end_w = self.tik_instance.Scalar(dtype="float32")
        self.win_end_h = self.tik_instance.Scalar(dtype="float32")
        self.h_iter_fp32 = self.tik_instance.Scalar(dtype="float32")
        self.w_iter_fp32 = self.tik_instance.Scalar(dtype="float32")
        self.max_h = self.tik_instance.Scalar(dtype="float32")
        self.max_w = self.tik_instance.Scalar(dtype="float32")
        self.min_h = self.tik_instance.Scalar(dtype="float32")
        self.min_w = self.tik_instance.Scalar(dtype="float32")
        self.start_w = self.tik_instance.Scalar(dtype="int32")
        self.end_w = self.tik_instance.Scalar(dtype="int32")
        self.start_h = self.tik_instance.Scalar(dtype="int32")
        self.end_h = self.tik_instance.Scalar(dtype="int32")
        self.h_int = self.tik_instance.Scalar(dtype="int32")
        self.w_int = self.tik_instance.Scalar(dtype="int32")
        self.rois_ub = None
        self.x0_ub = None
        self.y0_ub = None
        self.x1_ub = None
        self.y1_ub = None
        self.feture_index_fp32 = None
        self.feture_index_int32 = None
        self.roi_h_fp32 = None
        self.roi_w_fp32 = None
        self.vmax_0_fp32 = None
        self.roi_bin_h_fp32 = None
        self.roi_bin_w_fp32 = None
        self.win_size_fp32 = None
        self.feature_ele_ub = None
        self.sum_out_ub = None

    def pr_roi_pooling_compute_tiling(self):
        """
        pr_roi_pooling_compute_tiling
        """
        tik_instance = self.tik_instance
        with tik_instance.for_range(0, self.core_nums, block_num=self.core_nums) as core_idx:
            tiling_ub = tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                            name="tiling_ub",
                                            scope=tik.scope_ubuf)
            tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                   Constant.TILING_ARG_NUM // Constant.INT64_BLOCK_ELES, 0, 0)
            self._get_tiling_args(tiling_ub)
            self._calc_buffer_scalar_verify()
            with self.tik_instance.if_scope(core_idx < self.used_core_num):
                start_idx = self.tik_instance.Scalar(init_value=core_idx * self.num_per_core,
                                                     dtype="int32",
                                                     name="start_idx")
                with self.tik_instance.if_scope(core_idx == self.used_core_num - 1):
                    self.rois_valid.set_as(self.num_tail_core)
                with self.tik_instance.else_scope():
                    self.rois_valid.set_as(self.num_per_core)
                with tik_instance.if_scope(self.tiling_mode == Constant.TILING_MODE_1):
                    with tik_instance.new_stmt_scope():
                        self._compute_mode_1(start_idx)

    def pr_roi_pooling_compute(self):
        """
        pr_roi_pooling_compute
        """
        tik_instance = self.tik_instance
        self.pr_roi_pooling_compute_tiling()
        # add compile info
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_nums})

        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.features_gm, self.rois_gm],
                              outputs=[self.output_gm],
                              flowtable=(self.tiling_gm,),
                              config=self.opt_config)

        return tik_instance

    def _get_tiling_args(self, tiling_ub):
        self.tiling_mode.set_as(tiling_ub[0])
        self.rois_n.set_as(tiling_ub[1])
        self.c1_num.set_as(tiling_ub[2])
        self.in_height.set_as(tiling_ub[3])
        self.in_width.set_as(tiling_ub[4])
        self.num_per_core.set_as(tiling_ub[5])
        self.used_core_num.set_as(tiling_ub[6])
        self.num_tail_core.set_as(tiling_ub[7])

    def _calc_buffer_scalar_verify(self):
        tik_instance = self.tik_instance
        # ub init
        self.rois_ub = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE, Constant.ROI_ELE_NUMS],
                                           name="rois_ub",
                                           scope=tbe_platform.scope_ubuf)
        self.x0_ub = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE], name="x0_ub", scope=tbe_platform.scope_ubuf)
        self.y0_ub = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE], name="y0_ub", scope=tbe_platform.scope_ubuf)
        self.x1_ub = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE], name="x1_ub", scope=tbe_platform.scope_ubuf)
        self.y1_ub = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE], name="y1_ub", scope=tbe_platform.scope_ubuf)
        self.feture_index_fp32 = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                                     name="feture_index_fp32",
                                                     scope=tbe_platform.scope_ubuf)
        self.feture_index_int32 = tik_instance.Tensor("int32", [Constant.BATCH_SIZE],
                                                      name="feture_index_int32",
                                                      scope=tbe_platform.scope_ubuf)
        self.roi_h_fp32 = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                              name="roi_h_fp32",
                                              scope=tbe_platform.scope_ubuf)
        self.roi_w_fp32 = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                              name="roi_w_fp32",
                                              scope=tbe_platform.scope_ubuf)
        self.vmax_0_fp32 = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                               name="vmax_0_fp32",
                                               scope=tbe_platform.scope_ubuf)
        self.roi_bin_h_fp32 = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                                  name="roi_bin_h_fp32",
                                                  scope=tbe_platform.scope_ubuf)
        self.roi_bin_w_fp32 = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                                  name="roi_bin_w_fp32",
                                                  scope=tbe_platform.scope_ubuf)
        self.win_size_fp32 = tik_instance.Tensor(self.dtype, [Constant.BATCH_SIZE],
                                                 name="win_size_fp32",
                                                 scope=tbe_platform.scope_ubuf)
        self.feature_ele_ub = tik_instance.Tensor(self.dtype, [self.c1_num, Constant.C0_SIZE],
                                                  name="feature_ele_ub",
                                                  scope=tbe_platform.scope_ubuf)
        self.sum_out_ub = tik_instance.Tensor(self.dtype, [self.c1_num, Constant.C0_SIZE],
                                              name="sum_out_ub",
                                              scope=tbe_platform.scope_ubuf)

        # scalar init
        self.vec_cal_size.set_as(self.c1_num * Constant.C0_SIZE)


    def _get_feature_data(self, fetures_index, h, w):
        tik_instance = self.tik_instance
        self.h_int.set_as(h)
        self.w_int.set_as(w)
        gm_offset = (fetures_index * self.c1_num * self.in_height * self.in_width + self.h_int * self.in_width +
                     self.w_int) * Constant.C0_SIZE
        nburst = self.c1_num
        burst = Constant.C0_SIZE // Constant.FP32_BLOCK_ELES
        src_stride = (self.in_height * self.in_width - 1) * burst
        with tik_instance.if_scope(src_stride <= Constant.STRIDE_MAX):
            tik_instance.data_move(self.feature_ele_ub, self.features_gm[gm_offset], 0, nburst, burst, src_stride, 0)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, self.c1_num) as data_move_i:
                ub_offset = data_move_i * Constant.C0_SIZE
                gm_offset = (fetures_index * self.c1_num * self.in_height * self.in_width + self.h_int * self.in_width +
                             self.w_int + data_move_i * self.in_height * self.in_width) * Constant.C0_SIZE
                tik_instance.data_move(self.feature_ele_ub[ub_offset], self.features_gm[gm_offset], 0, 1, burst, 0, 0)
        return self.feature_ele_ub

    def _move_sum_out(self, out_index, i, j):
        tik_instance = self.tik_instance
        gm_offset = (out_index * self.c1_num * self.pooled_height * self.pooled_width + i * self.pooled_width +
                     j) * Constant.C0_SIZE
        nburst = self.c1_num
        burst = Constant.C0_SIZE // Constant.FP32_BLOCK_ELES
        dst_stride = (self.pooled_height * self.pooled_width - 1) * burst
        with tik_instance.if_scope(dst_stride <= Constant.STRIDE_MAX):
            tik_instance.data_move(self.output_gm[gm_offset], self.sum_out_ub, 0, nburst, burst, 0, dst_stride)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, self.c1_num) as data_move_i:
                ub_offset = data_move_i * Constant.C0_SIZE
                gm_offset = (out_index * self.c1_num * self.pooled_height * self.pooled_width + i * self.pooled_width +
                             j + data_move_i * self.pooled_height * self.pooled_width) * Constant.C0_SIZE
                tik_instance.data_move(self.output_gm[gm_offset], self.sum_out_ub[ub_offset], 0, 1, burst, 0, 0)

    def _vec_cal(self, cal_type, cal_ele_nums, dst, src_0, src_1=None, scalar_value=None):
        tik_instance = self.tik_instance
        cal_repeats = cal_ele_nums // Constant.FP32_VEC_REPEAT_MASK
        cal_remain = cal_ele_nums % Constant.FP32_VEC_REPEAT_MASK
        with tik_instance.if_scope(cal_repeats > 0):
            if cal_type == "vec_muls":
                tik_instance.vec_muls(Constant.FP32_VEC_REPEAT_MASK, dst, src_0, scalar_value, cal_repeats, 8, 8)
            if cal_type == "vec_add":
                tik_instance.vec_add(Constant.FP32_VEC_REPEAT_MASK, dst, src_0, src_1, cal_repeats, 8, 8, 8)
            if cal_type == "vector_dup":
                tik_instance.vector_dup(Constant.FP32_VEC_REPEAT_MASK, dst, scalar_value, cal_repeats, 1, 8)
        offset = Constant.FP32_VEC_REPEAT_MASK * cal_repeats
        with tik_instance.if_scope(cal_remain > 0):
            if cal_type == "vec_muls":
                tik_instance.vec_muls(cal_remain, dst[offset], src_0[offset], scalar_value, 1, 8, 8)
            if cal_type == "vec_add":
                tik_instance.vec_add(cal_remain, dst[offset], src_0[offset], src_1[offset], 1, 8, 8, 8)
            if cal_type == "vector_dup":
                tik_instance.vector_dup(cal_remain, dst, scalar_value, 1, 1, 8)

    def _mat_calculation(self, fetures_index, s_h, s_w, e_h, e_w, y0, x0, y1, x1):
        alpha = x0 - s_w
        beta = y0 - s_h
        lim_alpha = x1 - s_w
        lim_beta = y1 - s_h
        tmp = (lim_alpha - 0.5 * lim_alpha * lim_alpha - alpha +
               0.5 * alpha * alpha) * (lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        this_feature_ub = self._get_feature_data(fetures_index, s_h, s_w)
        self._vec_cal("vec_muls", self.vec_cal_size, this_feature_ub, this_feature_ub, scalar_value=tmp)
        self._vec_cal("vec_add", self.vec_cal_size, self.sum_out_ub, self.sum_out_ub, this_feature_ub)

        alpha = e_w - x1
        lim_alpha = e_w - x0
        tmp = (lim_alpha - 0.5 * lim_alpha * lim_alpha - alpha +
               0.5 * alpha * alpha) * (lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        this_feature_ub = self._get_feature_data(fetures_index, s_h, e_w)
        self._vec_cal("vec_muls", self.vec_cal_size, this_feature_ub, this_feature_ub, scalar_value=tmp)
        self._vec_cal("vec_add", self.vec_cal_size, self.sum_out_ub, self.sum_out_ub, this_feature_ub)

        alpha = x0 - s_w
        beta = e_h - y1
        lim_alpha = x1 - s_w
        lim_beta = e_h - y0
        tmp = (lim_alpha - 0.5 * lim_alpha * lim_alpha - alpha +
               0.5 * alpha * alpha) * (lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        this_feature_ub = self._get_feature_data(fetures_index, e_h, s_w)
        self._vec_cal("vec_muls", self.vec_cal_size, this_feature_ub, this_feature_ub, scalar_value=tmp)
        self._vec_cal("vec_add", self.vec_cal_size, self.sum_out_ub, self.sum_out_ub, this_feature_ub)

        alpha = e_w - x1
        lim_alpha = e_w - x0
        tmp = (lim_alpha - 0.5 * lim_alpha * lim_alpha - alpha +
               0.5 * alpha * alpha) * (lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        this_feature_ub = self._get_feature_data(fetures_index, e_h, e_w)
        self._vec_cal("vec_muls", self.vec_cal_size, this_feature_ub, this_feature_ub, scalar_value=tmp)
        self._vec_cal("vec_add", self.vec_cal_size, self.sum_out_ub, self.sum_out_ub, this_feature_ub)

    def _common_compute(self, move_out_idx, fetures_index, roi_start_w, roi_start_h, bin_size_w, bin_size_h, win_size):
        """
        common_compute
        """
        tik_instance = self.tik_instance
        with tik_instance.for_range(0, self.pooled_height) as i:
            with tik_instance.for_range(0, self.pooled_width) as j:
                self.win_start_h.set_as(roi_start_h + bin_size_h * i)
                self.win_start_w.set_as(roi_start_w + bin_size_w * j)
                self.win_end_h.set_as(self.win_start_h + bin_size_h)
                self.win_end_w.set_as(self.win_start_w + bin_size_w)
                tik_instance.scalar_conv("floor", self.start_h, self.win_start_h)
                tik_instance.scalar_conv("floor", self.start_w, self.win_start_w)
                tik_instance.scalar_conv("ceil", self.end_h, self.win_end_h)
                tik_instance.scalar_conv("ceil", self.end_w, self.win_end_w)
                self._vec_cal("vector_dup", self.vec_cal_size, self.sum_out_ub, self.sum_out_ub, scalar_value=0.0)
                with tik_instance.for_range(self.start_h, self.end_h) as h_iter:
                    with tik_instance.for_range(self.start_w, self.end_w) as w_iter:
                        self.h_iter_fp32.set_as(h_iter)
                        self.w_iter_fp32.set_as(w_iter)
                        with tik_instance.if_scope(self.win_start_h >= self.h_iter_fp32):
                            self.max_h.set_as(self.win_start_h)
                        with tik_instance.else_scope():
                            self.max_h.set_as(self.h_iter_fp32)
                        with tik_instance.if_scope(self.win_start_w >= self.w_iter_fp32):
                            self.max_w.set_as(self.win_start_w)
                        with tik_instance.else_scope():
                            self.max_w.set_as(self.w_iter_fp32)

                        with tik_instance.if_scope(self.win_end_h <= self.h_iter_fp32 + 1.0):
                            self.min_h.set_as(self.win_end_h)
                        with tik_instance.else_scope():
                            self.min_h.set_as(self.h_iter_fp32 + 1.0)
                        with tik_instance.if_scope(self.win_end_w <= self.w_iter_fp32 + 1.0):
                            self.min_w.set_as(self.win_end_w)
                        with tik_instance.else_scope():
                            self.min_w.set_as(self.w_iter_fp32 + 1.0)
                        self._mat_calculation(fetures_index, self.h_iter_fp32, self.w_iter_fp32, self.h_iter_fp32 + 1.0,
                                              self.w_iter_fp32 + 1.0, self.max_h, self.max_w, self.min_h, self.min_w)
                self._vec_cal("vec_muls",
                              self.vec_cal_size,
                              self.sum_out_ub,
                              self.sum_out_ub,
                              scalar_value=1 / win_size)
                self._move_sum_out(move_out_idx, i, j)

    def _get_size_paras(self, x0, y0, x1, y1):
        """
        get satart point, bin_size
        """
        tik_instance = self.tik_instance
        tik_instance.vector_dup(64, self.vmax_0_fp32, 0.0, 1, 1, 8)
        tik_instance.vec_muls(64, x0[0], x0[0], self.spatial_scale, 1, 8, 8)
        tik_instance.vec_muls(64, y0[0], y0[0], self.spatial_scale, 1, 8, 8)
        tik_instance.vec_muls(64, x1[0], x1[0], self.spatial_scale, 1, 8, 8)
        tik_instance.vec_muls(64, y1[0], y1[0], self.spatial_scale, 1, 8, 8)
        tik_instance.vec_sub(64, self.roi_h_fp32, y1[0], y0[0], 1, 8, 8, 8)
        tik_instance.vec_sub(64, self.roi_w_fp32, x1[0], x0[0], 1, 8, 8, 8)
        tik_instance.vec_max(64, self.roi_h_fp32, self.roi_h_fp32, self.vmax_0_fp32, 1, 8, 8, 8)
        tik_instance.vec_max(64, self.roi_w_fp32, self.roi_w_fp32, self.vmax_0_fp32, 1, 8, 8, 8)
        tik_instance.vec_muls(64, self.roi_bin_h_fp32[:], self.roi_h_fp32[:], 1.0 / self.pooled_height, 1, 8, 8)
        tik_instance.vec_muls(64, self.roi_bin_w_fp32[:], self.roi_w_fp32[:], 1.0 / self.pooled_width, 1, 8, 8)
        tik_instance.vec_mul(64, self.win_size_fp32, self.roi_bin_h_fp32, self.roi_bin_w_fp32, 1, 8, 8, 8)
        tik_instance.vec_max(64, self.win_size_fp32, self.win_size_fp32, self.vmax_0_fp32, 1, 8, 8, 8)

        return [x0, y0, self.roi_bin_w_fp32, self.roi_bin_h_fp32, self.win_size_fp32]

    def _compute_mode_1(self, start_idx):
        """
        tiling mode 1 case
        """
        tik_instance = self.tik_instance
        rois_batch_num = (self.rois_valid + Constant.BATCH_SIZE - 1) // Constant.BATCH_SIZE
        with tik_instance.if_scope(self.rois_valid != 0):
            with tik_instance.for_range(0, rois_batch_num) as roi_64_number:
                with tik_instance.if_scope(roi_64_number == (rois_batch_num - 1)):
                    self.rois_valid_in_block.set_as(self.rois_valid - roi_64_number * Constant.BATCH_SIZE)
                tik_instance.vector_dup(Constant.BATCH_SIZE, self.x0_ub, 0.0, 1, 1, 8)
                tik_instance.vector_dup(Constant.BATCH_SIZE, self.y0_ub, 0.0, 1, 1, 8)
                tik_instance.vector_dup(Constant.BATCH_SIZE, self.x1_ub, 0.0, 1, 1, 8)
                tik_instance.vector_dup(Constant.BATCH_SIZE, self.y1_ub, 0.0, 1, 1, 8)
                tik_instance.vector_dup(Constant.BATCH_SIZE, self.feture_index_fp32, 0.0, 1, 1, 8)
                tik_instance.data_move(
                    self.rois_ub, self.rois_gm[(start_idx + roi_64_number * Constant.BATCH_SIZE) * 5], 0, 1,
                    (self.rois_valid_in_block * 5 + Constant.FP32_BLOCK_ELES - 1) // Constant.FP32_BLOCK_ELES, 0, 0)
                with tik_instance.for_range(0, self.rois_valid_in_block) as j:
                    self.feture_index_fp32[j].set_as(self.rois_ub[j, 0])
                    self.x0_ub[j].set_as(self.rois_ub[j, 1])
                    self.y0_ub[j].set_as(self.rois_ub[j, 2])
                    self.x1_ub[j].set_as(self.rois_ub[j, 3])
                    self.y1_ub[j].set_as(self.rois_ub[j, 4])
                tik_instance.vec_conv(64, "floor", self.feture_index_int32[0], self.feture_index_fp32[0], 1, 8, 8)
                roi_start_w_ub, roi_start_h_ub, bin_size_w_ub, bin_size_h_ub, win_size_ub = \
                    self._get_size_paras(self.x0_ub, self.y0_ub, self.x1_ub, self.y1_ub)

                with tik_instance.for_range(0, self.rois_valid_in_block) as roi_idx:
                    self.move_out_idx.set_as(start_idx + roi_64_number * Constant.BATCH_SIZE + roi_idx)
                    self.fetures_index.set_as(self.feture_index_int32[roi_idx])
                    self.roi_start_w.set_as(roi_start_w_ub[roi_idx])
                    self.roi_start_h.set_as(roi_start_h_ub[roi_idx])
                    self.bin_size_w.set_as(bin_size_w_ub[roi_idx])
                    self.bin_size_h.set_as(bin_size_h_ub[roi_idx])
                    self.win_size.set_as(win_size_ub[roi_idx])
                    self._common_compute(self.move_out_idx, self.fetures_index, self.roi_start_w, self.roi_start_h,
                                         self.bin_size_w, self.bin_size_h, self.win_size)


@register_operator("PrRoIPooling")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def pr_roi_pooling(features, rois, y, pooled_height, pooled_width, spatial_scale, kernel_name="pr_roi_pooling"):
    """
    PrRoIPooling operator
    Parameters
    ----------
    features : dict
        shape and dtype of features input
    rois: dict
        shape and dtype of rois input
    y: dict
        shape and dtype of output
    pooled_height: int
    pooled_width: int
    spatial_scale: float
    kernel_name : str
        cce kernel name, default value is "pr_roi_pooling"

    Returns
    -------
    None.
    """
    features_dtype = features.get("dtype").lower()
    rois_dtype = rois.get("dtype").lower()
    supported_dtype = ("float32",)
    para_check.check_dtype(features_dtype, supported_dtype, param_name="features")
    para_check.check_dtype(rois_dtype, supported_dtype, param_name="rois")
    obj = PrRoIPooling(pooled_height, pooled_width, spatial_scale, kernel_name)

    return obj.pr_roi_pooling_compute()
