// tvm target: c -keys=cpu 
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_add_rsqrt_multiply(float* p0, float* T_multiply, uint8_t* global_const_workspace_26_var, uint8_t* global_workspace_27_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_mean(float* p0, float* T_divide, uint8_t* global_const_workspace_22_var, uint8_t* global_workspace_23_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_multiply_layout_transform(float* p0, float* T_layout_trans, uint8_t* global_const_workspace_30_var, uint8_t* global_workspace_31_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_negative_multiply_add_divide_add(float* p0, float* p1, float* p2, float* p3, float* T_add, uint8_t* global_const_workspace_28_var, uint8_t* global_workspace_29_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu(float* p0, float* T_relu, uint8_t* global_const_workspace_4_var, uint8_t* global_workspace_5_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_1(float* p0, float* T_relu, uint8_t* global_const_workspace_6_var, uint8_t* global_workspace_7_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2(float* p0, float* T_relu, uint8_t* global_const_workspace_10_var, uint8_t* global_workspace_11_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_3(float* p0, float* T_relu, uint8_t* global_const_workspace_12_var, uint8_t* global_workspace_13_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_4(float* p0, float* T_relu, uint8_t* global_const_workspace_16_var, uint8_t* global_workspace_17_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_5(float* p0, float* T_relu, uint8_t* global_const_workspace_18_var, uint8_t* global_workspace_19_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add(float* p0, float* T_add, uint8_t* global_const_workspace_44_var, uint8_t* global_workspace_45_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_1(float* p0, float* T_add, uint8_t* global_const_workspace_50_var, uint8_t* global_workspace_51_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu(float* p0, float* p1, float* T_relu, uint8_t* global_const_workspace_32_var, uint8_t* global_workspace_33_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_1(float* p0, float* T_relu, uint8_t* global_const_workspace_34_var, uint8_t* global_workspace_35_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_2(float* p0, float* T_relu, uint8_t* global_const_workspace_38_var, uint8_t* global_workspace_39_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_3(float* p0, float* T_relu, uint8_t* global_const_workspace_40_var, uint8_t* global_workspace_41_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_4(float* p0, float* T_relu, uint8_t* global_const_workspace_42_var, uint8_t* global_workspace_43_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_5(float* p0, float* T_relu, uint8_t* global_const_workspace_48_var, uint8_t* global_workspace_49_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_6(float* p0, float* T_relu, uint8_t* global_const_workspace_54_var, uint8_t* global_workspace_55_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_7(float* p0, float* T_relu, uint8_t* global_const_workspace_58_var, uint8_t* global_workspace_59_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_8(float* p0, float* T_relu, uint8_t* global_const_workspace_60_var, uint8_t* global_workspace_61_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_sigmoid_multiply(float* p0, float* T_multiply, uint8_t* global_const_workspace_56_var, uint8_t* global_workspace_57_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_global_avg_pool2d(float* p0, float* adaptive_pool_avg, uint8_t* global_const_workspace_20_var, uint8_t* global_workspace_21_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_max_pool2d(float* p0, float* pool_max, uint8_t* global_const_workspace_8_var, uint8_t* global_workspace_9_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_max_pool2d_1(float* p0, float* pool_max, uint8_t* global_const_workspace_14_var, uint8_t* global_workspace_15_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_softmax(float* p0, float* T_softmax_norm, uint8_t* global_const_workspace_46_var, uint8_t* global_workspace_47_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_softmax_1(float* p0, float* T_softmax_norm, uint8_t* global_const_workspace_52_var, uint8_t* global_workspace_53_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_squeeze_layout_transform_concatenate(float* p0, float* p1, float* concatenate_ext, uint8_t* global_const_workspace_36_var, uint8_t* global_workspace_37_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_subtract_multiply_mean(float* p0, float* p1, float* T_divide, uint8_t* global_const_workspace_24_var, uint8_t* global_workspace_25_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_layout_transform(float* p0, float* T_layout_trans, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* aux_buffer_var, float* spec_buffer_var, float* output_buffer_var, float* output2_buffer_var, float* output3_buffer_var, float* output4_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL float sqrtf(float);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL float expf(float);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL float expf(float);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL float expf(float);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_add_rsqrt_multiply(float* p0, float* T_multiply, uint8_t* global_const_workspace_26_var, uint8_t* global_workspace_27_var) {
  void* fused_add_rsqrt_constant_let = (&(global_const_workspace_26_var[1832480]));
  for (int32_t ax1_outer = 0; ax1_outer < 7; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
      if (((ax1_outer * 8) + (ax1_inner >> 1)) < 51) {
        int32_t cse_var_1 = ((ax1_outer * 16) + ax1_inner);
        T_multiply[cse_var_1] = ((1.000000e+00f / sqrtf((p0[0] + 1.000000e-03f))) * ((float*)fused_add_rsqrt_constant_let)[cse_var_1]);
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_mean(float* p0, float* T_divide, uint8_t* global_const_workspace_22_var, uint8_t* global_workspace_23_var) {
  void* p0_red_let = (&(global_workspace_23_var[54096]));
  ((float*)p0_red_let)[0] = 0.000000e+00f;
  for (int32_t k1 = 0; k1 < 102; ++k1) {
    ((float*)p0_red_let)[0] = (((float*)p0_red_let)[0] + p0[k1]);
  }
  T_divide[0] = (((float*)p0_red_let)[0] * 9.803922e-03f);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_multiply_layout_transform(float* p0, float* T_layout_trans, uint8_t* global_const_workspace_30_var, uint8_t* global_workspace_31_var) {
  void* fused_constant_6_let = (&(global_const_workspace_30_var[1630208]));
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 1632; ++ax0_ax1_fused) {
    for (int32_t ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
      int32_t cse_var_1 = (ax0_ax1_fused % 102);
      T_layout_trans[((ax0_ax1_fused * 8) + ax2_inner)] = (((float*)fused_constant_6_let)[((((ax0_ax1_fused / 102) * 816) + (ax2_inner * 102)) + cse_var_1)] * p0[cse_var_1]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_negative_multiply_add_divide_add(float* p0, float* p1, float* p2, float* p3, float* T_add, uint8_t* global_const_workspace_28_var, uint8_t* global_workspace_29_var) {
  void* fused_negative_multiply_constant_let = (&(global_const_workspace_28_var[1832064]));
  for (int32_t ax1_outer = 0; ax1_outer < 7; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
      if (((ax1_outer * 8) + (ax1_inner >> 1)) < 51) {
        int32_t cse_var_1 = ((ax1_outer * 16) + ax1_inner);
        T_add[cse_var_1] = (p3[cse_var_1] + ((((0.000000e+00f - p0[0]) * p1[cse_var_1]) + ((float*)fused_negative_multiply_constant_let)[cse_var_1]) / p2[cse_var_1]));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu(float* p0, float* T_relu, uint8_t* global_const_workspace_4_var, uint8_t* global_workspace_5_var) {
  void* fused_nn_contrib_conv2d_NCHWc_constant_let = (&(global_const_workspace_4_var[1834816]));
  void* fused_constant_let = (&(global_const_workspace_4_var[1825792]));
  void* data_pad_let = (&(global_workspace_5_var[4260352]));
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 130; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 130; ++i3) {
      ((float*)data_pad_let)[((i0_i1_fused_i2_fused * 130) + i3)] = (((((1 <= i0_i1_fused_i2_fused) && (i0_i1_fused_i2_fused < 129)) && (1 <= i3)) && (i3 < 129)) ? p0[(((i0_i1_fused_i2_fused * 128) + i3) - 129)] : 0.000000e+00f);
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1024; ++ax0_ax1_fused_ax2_fused) {
    void* conv2d_NCHWc_let = (&(global_workspace_5_var[4327952]));
    void* conv2d_NCHWc_global_let = (&(global_workspace_5_var[4330000]));
    for (int32_t ow_outer = 0; ow_outer < 8; ++ow_outer) {
      for (int32_t oc_block_c_init = 0; oc_block_c_init < 4; ++oc_block_c_init) {
        ((float*)conv2d_NCHWc_global_let)[oc_block_c_init] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_1 = 0; oc_block_c_init_1 < 4; ++oc_block_c_init_1) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_1 + 4)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_2 = 0; oc_block_c_init_2 < 4; ++oc_block_c_init_2) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_2 + 8)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_3 = 0; oc_block_c_init_3 < 4; ++oc_block_c_init_3) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_3 + 12)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_4 = 0; oc_block_c_init_4 < 4; ++oc_block_c_init_4) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_4 + 16)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_5 = 0; oc_block_c_init_5 < 4; ++oc_block_c_init_5) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_5 + 20)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_6 = 0; oc_block_c_init_6 < 4; ++oc_block_c_init_6) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_6 + 24)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_7 = 0; oc_block_c_init_7 < 4; ++oc_block_c_init_7) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_7 + 28)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_8 = 0; oc_block_c_init_8 < 4; ++oc_block_c_init_8) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_8 + 32)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_9 = 0; oc_block_c_init_9 < 4; ++oc_block_c_init_9) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_9 + 36)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_10 = 0; oc_block_c_init_10 < 4; ++oc_block_c_init_10) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_10 + 40)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_11 = 0; oc_block_c_init_11 < 4; ++oc_block_c_init_11) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_11 + 44)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_12 = 0; oc_block_c_init_12 < 4; ++oc_block_c_init_12) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_12 + 48)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_13 = 0; oc_block_c_init_13 < 4; ++oc_block_c_init_13) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_13 + 52)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_14 = 0; oc_block_c_init_14 < 4; ++oc_block_c_init_14) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_14 + 56)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_15 = 0; oc_block_c_init_15 < 4; ++oc_block_c_init_15) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_15 + 60)] = 0.000000e+00f;
      }
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t oc_block_c = 0; oc_block_c < 4; ++oc_block_c) {
            ((float*)conv2d_NCHWc_global_let)[oc_block_c] = (((float*)conv2d_NCHWc_global_let)[oc_block_c] + (((float*)data_pad_let)[((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c)]));
          }
          for (int32_t oc_block_c_1 = 0; oc_block_c_1 < 4; ++oc_block_c_1) {
            int32_t cse_var_1 = (oc_block_c_1 + 4);
            ((float*)conv2d_NCHWc_global_let)[cse_var_1] = (((float*)conv2d_NCHWc_global_let)[cse_var_1] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 1)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_1)]));
          }
          for (int32_t oc_block_c_2 = 0; oc_block_c_2 < 4; ++oc_block_c_2) {
            int32_t cse_var_2 = (oc_block_c_2 + 8);
            ((float*)conv2d_NCHWc_global_let)[cse_var_2] = (((float*)conv2d_NCHWc_global_let)[cse_var_2] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 2)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_2)]));
          }
          for (int32_t oc_block_c_3 = 0; oc_block_c_3 < 4; ++oc_block_c_3) {
            int32_t cse_var_3 = (oc_block_c_3 + 12);
            ((float*)conv2d_NCHWc_global_let)[cse_var_3] = (((float*)conv2d_NCHWc_global_let)[cse_var_3] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 3)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_3)]));
          }
          for (int32_t oc_block_c_4 = 0; oc_block_c_4 < 4; ++oc_block_c_4) {
            int32_t cse_var_4 = (oc_block_c_4 + 16);
            ((float*)conv2d_NCHWc_global_let)[cse_var_4] = (((float*)conv2d_NCHWc_global_let)[cse_var_4] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 4)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_4)]));
          }
          for (int32_t oc_block_c_5 = 0; oc_block_c_5 < 4; ++oc_block_c_5) {
            int32_t cse_var_5 = (oc_block_c_5 + 20);
            ((float*)conv2d_NCHWc_global_let)[cse_var_5] = (((float*)conv2d_NCHWc_global_let)[cse_var_5] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 5)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_5)]));
          }
          for (int32_t oc_block_c_6 = 0; oc_block_c_6 < 4; ++oc_block_c_6) {
            int32_t cse_var_6 = (oc_block_c_6 + 24);
            ((float*)conv2d_NCHWc_global_let)[cse_var_6] = (((float*)conv2d_NCHWc_global_let)[cse_var_6] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 6)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_6)]));
          }
          for (int32_t oc_block_c_7 = 0; oc_block_c_7 < 4; ++oc_block_c_7) {
            int32_t cse_var_7 = (oc_block_c_7 + 28);
            ((float*)conv2d_NCHWc_global_let)[cse_var_7] = (((float*)conv2d_NCHWc_global_let)[cse_var_7] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 7)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_7)]));
          }
          for (int32_t oc_block_c_8 = 0; oc_block_c_8 < 4; ++oc_block_c_8) {
            int32_t cse_var_8 = (oc_block_c_8 + 32);
            ((float*)conv2d_NCHWc_global_let)[cse_var_8] = (((float*)conv2d_NCHWc_global_let)[cse_var_8] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 8)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_8)]));
          }
          for (int32_t oc_block_c_9 = 0; oc_block_c_9 < 4; ++oc_block_c_9) {
            int32_t cse_var_9 = (oc_block_c_9 + 36);
            ((float*)conv2d_NCHWc_global_let)[cse_var_9] = (((float*)conv2d_NCHWc_global_let)[cse_var_9] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 9)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_9)]));
          }
          for (int32_t oc_block_c_10 = 0; oc_block_c_10 < 4; ++oc_block_c_10) {
            int32_t cse_var_10 = (oc_block_c_10 + 40);
            ((float*)conv2d_NCHWc_global_let)[cse_var_10] = (((float*)conv2d_NCHWc_global_let)[cse_var_10] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 10)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_10)]));
          }
          for (int32_t oc_block_c_11 = 0; oc_block_c_11 < 4; ++oc_block_c_11) {
            int32_t cse_var_11 = (oc_block_c_11 + 44);
            ((float*)conv2d_NCHWc_global_let)[cse_var_11] = (((float*)conv2d_NCHWc_global_let)[cse_var_11] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 11)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_11)]));
          }
          for (int32_t oc_block_c_12 = 0; oc_block_c_12 < 4; ++oc_block_c_12) {
            int32_t cse_var_12 = (oc_block_c_12 + 48);
            ((float*)conv2d_NCHWc_global_let)[cse_var_12] = (((float*)conv2d_NCHWc_global_let)[cse_var_12] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 12)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_12)]));
          }
          for (int32_t oc_block_c_13 = 0; oc_block_c_13 < 4; ++oc_block_c_13) {
            int32_t cse_var_13 = (oc_block_c_13 + 52);
            ((float*)conv2d_NCHWc_global_let)[cse_var_13] = (((float*)conv2d_NCHWc_global_let)[cse_var_13] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 13)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_13)]));
          }
          for (int32_t oc_block_c_14 = 0; oc_block_c_14 < 4; ++oc_block_c_14) {
            int32_t cse_var_14 = (oc_block_c_14 + 56);
            ((float*)conv2d_NCHWc_global_let)[cse_var_14] = (((float*)conv2d_NCHWc_global_let)[cse_var_14] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 14)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_14)]));
          }
          for (int32_t oc_block_c_15 = 0; oc_block_c_15 < 4; ++oc_block_c_15) {
            int32_t cse_var_15 = (oc_block_c_15 + 60);
            ((float*)conv2d_NCHWc_global_let)[cse_var_15] = (((float*)conv2d_NCHWc_global_let)[cse_var_15] + (((float*)data_pad_let)[(((((kh * 130) + ((ax0_ax1_fused_ax2_fused & 127) * 130)) + (ow_outer * 16)) + kw) + 15)] * ((float*)fused_constant_let)[(((((ax0_ax1_fused_ax2_fused >> 7) * 36) + (kh * 12)) + (kw * 4)) + oc_block_c_15)]));
          }
        }
      }
      for (int32_t ow_inner = 0; ow_inner < 16; ++ow_inner) {
        for (int32_t oc_block = 0; oc_block < 4; ++oc_block) {
          int32_t cse_var_16 = (ow_inner * 4);
          ((float*)conv2d_NCHWc_let)[(((ow_outer * 64) + cse_var_16) + oc_block)] = ((float*)conv2d_NCHWc_global_let)[(cse_var_16 + oc_block)];
        }
      }
    }
    for (int32_t ax3_outer = 0; ax3_outer < 8; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
          int32_t cse_var_18 = (ax3_outer * 64);
          int32_t cse_var_17 = (ax3_inner * 4);
          float v_ = ((float*)conv2d_NCHWc_let)[((cse_var_18 + cse_var_17) + ax4)] + ((float*)fused_nn_contrib_conv2d_NCHWc_constant_let)[(((ax0_ax1_fused_ax2_fused >> 7) * 4) + ax4)];
          T_relu[((((ax0_ax1_fused_ax2_fused * 512) + cse_var_18) + cse_var_17) + ax4)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_1(float* p0, float* T_relu, uint8_t* global_const_workspace_6_var, uint8_t* global_workspace_7_var) {
  void* fused_nn_contrib_conv2d_NCHWc_constant_1_let = (&(global_const_workspace_6_var[1834688]));
  void* fused_constant_1_let = (&(global_const_workspace_6_var[1682432]));
  void* data_pad_let = (&(global_workspace_7_var[0]));
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1040; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 130; ++i3) {
      for (int32_t i4 = 0; i4 < 4; ++i4) {
        int32_t cse_var_2 = (i0_i1_fused_i2_fused % 130);
        int32_t cse_var_1 = (i3 * 4);
        ((float*)data_pad_let)[(((i0_i1_fused_i2_fused * 520) + cse_var_1) + i4)] = (((((1 <= cse_var_2) && (cse_var_2 < 129)) && (1 <= i3)) && (i3 < 129)) ? p0[((((((i0_i1_fused_i2_fused / 130) * 65536) + (cse_var_2 * 512)) + cse_var_1) + i4) - 516)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1024; ++ax0_ax1_fused_ax2_fused) {
    void* conv2d_NCHWc_let = (&(global_workspace_7_var[4260352]));
    void* conv2d_NCHWc_global_let = (&(global_workspace_7_var[4262400]));
    for (int32_t ow_outer = 0; ow_outer < 8; ++ow_outer) {
      for (int32_t oc_block_c_init = 0; oc_block_c_init < 4; ++oc_block_c_init) {
        ((float*)conv2d_NCHWc_global_let)[oc_block_c_init] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_1 = 0; oc_block_c_init_1 < 4; ++oc_block_c_init_1) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_1 + 4)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_2 = 0; oc_block_c_init_2 < 4; ++oc_block_c_init_2) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_2 + 8)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_3 = 0; oc_block_c_init_3 < 4; ++oc_block_c_init_3) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_3 + 12)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_4 = 0; oc_block_c_init_4 < 4; ++oc_block_c_init_4) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_4 + 16)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_5 = 0; oc_block_c_init_5 < 4; ++oc_block_c_init_5) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_5 + 20)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_6 = 0; oc_block_c_init_6 < 4; ++oc_block_c_init_6) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_6 + 24)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_7 = 0; oc_block_c_init_7 < 4; ++oc_block_c_init_7) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_7 + 28)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_8 = 0; oc_block_c_init_8 < 4; ++oc_block_c_init_8) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_8 + 32)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_9 = 0; oc_block_c_init_9 < 4; ++oc_block_c_init_9) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_9 + 36)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_10 = 0; oc_block_c_init_10 < 4; ++oc_block_c_init_10) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_10 + 40)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_11 = 0; oc_block_c_init_11 < 4; ++oc_block_c_init_11) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_11 + 44)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_12 = 0; oc_block_c_init_12 < 4; ++oc_block_c_init_12) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_12 + 48)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_13 = 0; oc_block_c_init_13 < 4; ++oc_block_c_init_13) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_13 + 52)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_14 = 0; oc_block_c_init_14 < 4; ++oc_block_c_init_14) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_14 + 56)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_15 = 0; oc_block_c_init_15 < 4; ++oc_block_c_init_15) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_15 + 60)] = 0.000000e+00f;
      }
      for (int32_t ic_outer = 0; ic_outer < 8; ++ic_outer) {
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            for (int32_t ic_inner = 0; ic_inner < 4; ++ic_inner) {
              for (int32_t oc_block_c = 0; oc_block_c < 4; ++oc_block_c) {
                ((float*)conv2d_NCHWc_global_let)[oc_block_c] = (((float*)conv2d_NCHWc_global_let)[oc_block_c] + (((float*)data_pad_let)[((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c)]));
              }
              for (int32_t oc_block_c_1 = 0; oc_block_c_1 < 4; ++oc_block_c_1) {
                int32_t cse_var_3 = (oc_block_c_1 + 4);
                ((float*)conv2d_NCHWc_global_let)[cse_var_3] = (((float*)conv2d_NCHWc_global_let)[cse_var_3] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 4)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_1)]));
              }
              for (int32_t oc_block_c_2 = 0; oc_block_c_2 < 4; ++oc_block_c_2) {
                int32_t cse_var_4 = (oc_block_c_2 + 8);
                ((float*)conv2d_NCHWc_global_let)[cse_var_4] = (((float*)conv2d_NCHWc_global_let)[cse_var_4] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 8)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_2)]));
              }
              for (int32_t oc_block_c_3 = 0; oc_block_c_3 < 4; ++oc_block_c_3) {
                int32_t cse_var_5 = (oc_block_c_3 + 12);
                ((float*)conv2d_NCHWc_global_let)[cse_var_5] = (((float*)conv2d_NCHWc_global_let)[cse_var_5] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 12)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_3)]));
              }
              for (int32_t oc_block_c_4 = 0; oc_block_c_4 < 4; ++oc_block_c_4) {
                int32_t cse_var_6 = (oc_block_c_4 + 16);
                ((float*)conv2d_NCHWc_global_let)[cse_var_6] = (((float*)conv2d_NCHWc_global_let)[cse_var_6] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 16)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_4)]));
              }
              for (int32_t oc_block_c_5 = 0; oc_block_c_5 < 4; ++oc_block_c_5) {
                int32_t cse_var_7 = (oc_block_c_5 + 20);
                ((float*)conv2d_NCHWc_global_let)[cse_var_7] = (((float*)conv2d_NCHWc_global_let)[cse_var_7] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 20)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_5)]));
              }
              for (int32_t oc_block_c_6 = 0; oc_block_c_6 < 4; ++oc_block_c_6) {
                int32_t cse_var_8 = (oc_block_c_6 + 24);
                ((float*)conv2d_NCHWc_global_let)[cse_var_8] = (((float*)conv2d_NCHWc_global_let)[cse_var_8] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 24)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_6)]));
              }
              for (int32_t oc_block_c_7 = 0; oc_block_c_7 < 4; ++oc_block_c_7) {
                int32_t cse_var_9 = (oc_block_c_7 + 28);
                ((float*)conv2d_NCHWc_global_let)[cse_var_9] = (((float*)conv2d_NCHWc_global_let)[cse_var_9] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 28)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_7)]));
              }
              for (int32_t oc_block_c_8 = 0; oc_block_c_8 < 4; ++oc_block_c_8) {
                int32_t cse_var_10 = (oc_block_c_8 + 32);
                ((float*)conv2d_NCHWc_global_let)[cse_var_10] = (((float*)conv2d_NCHWc_global_let)[cse_var_10] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 32)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_8)]));
              }
              for (int32_t oc_block_c_9 = 0; oc_block_c_9 < 4; ++oc_block_c_9) {
                int32_t cse_var_11 = (oc_block_c_9 + 36);
                ((float*)conv2d_NCHWc_global_let)[cse_var_11] = (((float*)conv2d_NCHWc_global_let)[cse_var_11] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 36)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_9)]));
              }
              for (int32_t oc_block_c_10 = 0; oc_block_c_10 < 4; ++oc_block_c_10) {
                int32_t cse_var_12 = (oc_block_c_10 + 40);
                ((float*)conv2d_NCHWc_global_let)[cse_var_12] = (((float*)conv2d_NCHWc_global_let)[cse_var_12] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 40)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_10)]));
              }
              for (int32_t oc_block_c_11 = 0; oc_block_c_11 < 4; ++oc_block_c_11) {
                int32_t cse_var_13 = (oc_block_c_11 + 44);
                ((float*)conv2d_NCHWc_global_let)[cse_var_13] = (((float*)conv2d_NCHWc_global_let)[cse_var_13] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 44)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_11)]));
              }
              for (int32_t oc_block_c_12 = 0; oc_block_c_12 < 4; ++oc_block_c_12) {
                int32_t cse_var_14 = (oc_block_c_12 + 48);
                ((float*)conv2d_NCHWc_global_let)[cse_var_14] = (((float*)conv2d_NCHWc_global_let)[cse_var_14] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 48)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_12)]));
              }
              for (int32_t oc_block_c_13 = 0; oc_block_c_13 < 4; ++oc_block_c_13) {
                int32_t cse_var_15 = (oc_block_c_13 + 52);
                ((float*)conv2d_NCHWc_global_let)[cse_var_15] = (((float*)conv2d_NCHWc_global_let)[cse_var_15] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 52)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_13)]));
              }
              for (int32_t oc_block_c_14 = 0; oc_block_c_14 < 4; ++oc_block_c_14) {
                int32_t cse_var_16 = (oc_block_c_14 + 56);
                ((float*)conv2d_NCHWc_global_let)[cse_var_16] = (((float*)conv2d_NCHWc_global_let)[cse_var_16] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 56)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_14)]));
              }
              for (int32_t oc_block_c_15 = 0; oc_block_c_15 < 4; ++oc_block_c_15) {
                int32_t cse_var_17 = (oc_block_c_15 + 60);
                ((float*)conv2d_NCHWc_global_let)[cse_var_17] = (((float*)conv2d_NCHWc_global_let)[cse_var_17] + (((float*)data_pad_let)[(((((((ic_outer * 67600) + (kh * 520)) + ((ax0_ax1_fused_ax2_fused & 127) * 520)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 60)] * ((float*)fused_constant_1_let)[(((((((ax0_ax1_fused_ax2_fused >> 7) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_15)]));
              }
            }
          }
        }
      }
      for (int32_t ow_inner = 0; ow_inner < 16; ++ow_inner) {
        for (int32_t oc_block = 0; oc_block < 4; ++oc_block) {
          int32_t cse_var_18 = (ow_inner * 4);
          ((float*)conv2d_NCHWc_let)[(((ow_outer * 64) + cse_var_18) + oc_block)] = ((float*)conv2d_NCHWc_global_let)[(cse_var_18 + oc_block)];
        }
      }
    }
    for (int32_t ax3_outer = 0; ax3_outer < 8; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
          int32_t cse_var_20 = (ax3_outer * 64);
          int32_t cse_var_19 = (ax3_inner * 4);
          float v_ = ((float*)conv2d_NCHWc_let)[((cse_var_20 + cse_var_19) + ax4)] + ((float*)fused_nn_contrib_conv2d_NCHWc_constant_1_let)[(((ax0_ax1_fused_ax2_fused >> 7) * 4) + ax4)];
          T_relu[((((ax0_ax1_fused_ax2_fused * 512) + cse_var_20) + cse_var_19) + ax4)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2(float* p0, float* T_relu, uint8_t* global_const_workspace_10_var, uint8_t* global_workspace_11_var) {
  void* fused_nn_contrib_conv2d_NCHWc_constant_2_let = (&(global_const_workspace_10_var[1833920]));
  void* fused_constant_2_let = (&(global_const_workspace_10_var[1425408]));
  void* data_pad_let = (&(global_workspace_11_var[2163712]));
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 528; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 66; ++i3) {
      for (int32_t i4 = 0; i4 < 4; ++i4) {
        int32_t cse_var_2 = (i0_i1_fused_i2_fused % 66);
        int32_t cse_var_1 = (i3 * 4);
        ((float*)data_pad_let)[(((i0_i1_fused_i2_fused * 264) + cse_var_1) + i4)] = (((((1 <= cse_var_2) && (cse_var_2 < 65)) && (1 <= i3)) && (i3 < 65)) ? p0[((((((i0_i1_fused_i2_fused / 66) * 16384) + (cse_var_2 * 256)) + cse_var_1) + i4) - 260)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1024; ++ax0_ax1_fused_ax2_fused) {
    void* conv2d_NCHWc_let = (&(global_workspace_11_var[2721280]));
    void* conv2d_NCHWc_global_let = (&(global_workspace_11_var[2722304]));
    for (int32_t ow_outer = 0; ow_outer < 4; ++ow_outer) {
      for (int32_t oc_block_c_init = 0; oc_block_c_init < 4; ++oc_block_c_init) {
        ((float*)conv2d_NCHWc_global_let)[oc_block_c_init] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_1 = 0; oc_block_c_init_1 < 4; ++oc_block_c_init_1) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_1 + 4)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_2 = 0; oc_block_c_init_2 < 4; ++oc_block_c_init_2) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_2 + 8)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_3 = 0; oc_block_c_init_3 < 4; ++oc_block_c_init_3) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_3 + 12)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_4 = 0; oc_block_c_init_4 < 4; ++oc_block_c_init_4) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_4 + 16)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_5 = 0; oc_block_c_init_5 < 4; ++oc_block_c_init_5) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_5 + 20)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_6 = 0; oc_block_c_init_6 < 4; ++oc_block_c_init_6) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_6 + 24)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_7 = 0; oc_block_c_init_7 < 4; ++oc_block_c_init_7) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_7 + 28)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_8 = 0; oc_block_c_init_8 < 4; ++oc_block_c_init_8) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_8 + 32)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_9 = 0; oc_block_c_init_9 < 4; ++oc_block_c_init_9) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_9 + 36)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_10 = 0; oc_block_c_init_10 < 4; ++oc_block_c_init_10) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_10 + 40)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_11 = 0; oc_block_c_init_11 < 4; ++oc_block_c_init_11) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_11 + 44)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_12 = 0; oc_block_c_init_12 < 4; ++oc_block_c_init_12) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_12 + 48)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_13 = 0; oc_block_c_init_13 < 4; ++oc_block_c_init_13) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_13 + 52)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_14 = 0; oc_block_c_init_14 < 4; ++oc_block_c_init_14) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_14 + 56)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_15 = 0; oc_block_c_init_15 < 4; ++oc_block_c_init_15) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_15 + 60)] = 0.000000e+00f;
      }
      for (int32_t ic_outer = 0; ic_outer < 8; ++ic_outer) {
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            for (int32_t ic_inner = 0; ic_inner < 4; ++ic_inner) {
              for (int32_t oc_block_c = 0; oc_block_c < 4; ++oc_block_c) {
                ((float*)conv2d_NCHWc_global_let)[oc_block_c] = (((float*)conv2d_NCHWc_global_let)[oc_block_c] + (((float*)data_pad_let)[((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c)]));
              }
              for (int32_t oc_block_c_1 = 0; oc_block_c_1 < 4; ++oc_block_c_1) {
                int32_t cse_var_3 = (oc_block_c_1 + 4);
                ((float*)conv2d_NCHWc_global_let)[cse_var_3] = (((float*)conv2d_NCHWc_global_let)[cse_var_3] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 4)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_1)]));
              }
              for (int32_t oc_block_c_2 = 0; oc_block_c_2 < 4; ++oc_block_c_2) {
                int32_t cse_var_4 = (oc_block_c_2 + 8);
                ((float*)conv2d_NCHWc_global_let)[cse_var_4] = (((float*)conv2d_NCHWc_global_let)[cse_var_4] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 8)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_2)]));
              }
              for (int32_t oc_block_c_3 = 0; oc_block_c_3 < 4; ++oc_block_c_3) {
                int32_t cse_var_5 = (oc_block_c_3 + 12);
                ((float*)conv2d_NCHWc_global_let)[cse_var_5] = (((float*)conv2d_NCHWc_global_let)[cse_var_5] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 12)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_3)]));
              }
              for (int32_t oc_block_c_4 = 0; oc_block_c_4 < 4; ++oc_block_c_4) {
                int32_t cse_var_6 = (oc_block_c_4 + 16);
                ((float*)conv2d_NCHWc_global_let)[cse_var_6] = (((float*)conv2d_NCHWc_global_let)[cse_var_6] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 16)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_4)]));
              }
              for (int32_t oc_block_c_5 = 0; oc_block_c_5 < 4; ++oc_block_c_5) {
                int32_t cse_var_7 = (oc_block_c_5 + 20);
                ((float*)conv2d_NCHWc_global_let)[cse_var_7] = (((float*)conv2d_NCHWc_global_let)[cse_var_7] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 20)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_5)]));
              }
              for (int32_t oc_block_c_6 = 0; oc_block_c_6 < 4; ++oc_block_c_6) {
                int32_t cse_var_8 = (oc_block_c_6 + 24);
                ((float*)conv2d_NCHWc_global_let)[cse_var_8] = (((float*)conv2d_NCHWc_global_let)[cse_var_8] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 24)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_6)]));
              }
              for (int32_t oc_block_c_7 = 0; oc_block_c_7 < 4; ++oc_block_c_7) {
                int32_t cse_var_9 = (oc_block_c_7 + 28);
                ((float*)conv2d_NCHWc_global_let)[cse_var_9] = (((float*)conv2d_NCHWc_global_let)[cse_var_9] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 28)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_7)]));
              }
              for (int32_t oc_block_c_8 = 0; oc_block_c_8 < 4; ++oc_block_c_8) {
                int32_t cse_var_10 = (oc_block_c_8 + 32);
                ((float*)conv2d_NCHWc_global_let)[cse_var_10] = (((float*)conv2d_NCHWc_global_let)[cse_var_10] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 32)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_8)]));
              }
              for (int32_t oc_block_c_9 = 0; oc_block_c_9 < 4; ++oc_block_c_9) {
                int32_t cse_var_11 = (oc_block_c_9 + 36);
                ((float*)conv2d_NCHWc_global_let)[cse_var_11] = (((float*)conv2d_NCHWc_global_let)[cse_var_11] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 36)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_9)]));
              }
              for (int32_t oc_block_c_10 = 0; oc_block_c_10 < 4; ++oc_block_c_10) {
                int32_t cse_var_12 = (oc_block_c_10 + 40);
                ((float*)conv2d_NCHWc_global_let)[cse_var_12] = (((float*)conv2d_NCHWc_global_let)[cse_var_12] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 40)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_10)]));
              }
              for (int32_t oc_block_c_11 = 0; oc_block_c_11 < 4; ++oc_block_c_11) {
                int32_t cse_var_13 = (oc_block_c_11 + 44);
                ((float*)conv2d_NCHWc_global_let)[cse_var_13] = (((float*)conv2d_NCHWc_global_let)[cse_var_13] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 44)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_11)]));
              }
              for (int32_t oc_block_c_12 = 0; oc_block_c_12 < 4; ++oc_block_c_12) {
                int32_t cse_var_14 = (oc_block_c_12 + 48);
                ((float*)conv2d_NCHWc_global_let)[cse_var_14] = (((float*)conv2d_NCHWc_global_let)[cse_var_14] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 48)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_12)]));
              }
              for (int32_t oc_block_c_13 = 0; oc_block_c_13 < 4; ++oc_block_c_13) {
                int32_t cse_var_15 = (oc_block_c_13 + 52);
                ((float*)conv2d_NCHWc_global_let)[cse_var_15] = (((float*)conv2d_NCHWc_global_let)[cse_var_15] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 52)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_13)]));
              }
              for (int32_t oc_block_c_14 = 0; oc_block_c_14 < 4; ++oc_block_c_14) {
                int32_t cse_var_16 = (oc_block_c_14 + 56);
                ((float*)conv2d_NCHWc_global_let)[cse_var_16] = (((float*)conv2d_NCHWc_global_let)[cse_var_16] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 56)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_14)]));
              }
              for (int32_t oc_block_c_15 = 0; oc_block_c_15 < 4; ++oc_block_c_15) {
                int32_t cse_var_17 = (oc_block_c_15 + 60);
                ((float*)conv2d_NCHWc_global_let)[cse_var_17] = (((float*)conv2d_NCHWc_global_let)[cse_var_17] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 60)] * ((float*)fused_constant_2_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 1152) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_15)]));
              }
            }
          }
        }
      }
      for (int32_t ow_inner = 0; ow_inner < 16; ++ow_inner) {
        for (int32_t oc_block = 0; oc_block < 4; ++oc_block) {
          int32_t cse_var_18 = (ow_inner * 4);
          ((float*)conv2d_NCHWc_let)[(((ow_outer * 64) + cse_var_18) + oc_block)] = ((float*)conv2d_NCHWc_global_let)[(cse_var_18 + oc_block)];
        }
      }
    }
    for (int32_t ax3_outer = 0; ax3_outer < 4; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
          int32_t cse_var_20 = (ax3_outer * 64);
          int32_t cse_var_19 = (ax3_inner * 4);
          float v_ = ((float*)conv2d_NCHWc_let)[((cse_var_20 + cse_var_19) + ax4)] + ((float*)fused_nn_contrib_conv2d_NCHWc_constant_2_let)[(((ax0_ax1_fused_ax2_fused >> 6) * 4) + ax4)];
          T_relu[((((ax0_ax1_fused_ax2_fused * 256) + cse_var_20) + cse_var_19) + ax4)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_3(float* p0, float* T_relu, uint8_t* global_const_workspace_12_var, uint8_t* global_workspace_13_var) {
  void* fused_nn_contrib_conv2d_NCHWc_constant_3_let = (&(global_const_workspace_12_var[1833664]));
  void* fused_constant_3_let = (&(global_const_workspace_12_var[1146880]));
  void* data_pad_let = (&(global_workspace_13_var[0]));
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1056; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 66; ++i3) {
      for (int32_t i4 = 0; i4 < 4; ++i4) {
        int32_t cse_var_2 = (i0_i1_fused_i2_fused % 66);
        int32_t cse_var_1 = (i3 * 4);
        ((float*)data_pad_let)[(((i0_i1_fused_i2_fused * 264) + cse_var_1) + i4)] = (((((1 <= cse_var_2) && (cse_var_2 < 65)) && (1 <= i3)) && (i3 < 65)) ? p0[((((((i0_i1_fused_i2_fused / 66) * 16384) + (cse_var_2 * 256)) + cse_var_1) + i4) - 260)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1024; ++ax0_ax1_fused_ax2_fused) {
    void* conv2d_NCHWc_let = (&(global_workspace_13_var[2163712]));
    void* conv2d_NCHWc_global_let = (&(global_workspace_13_var[2164736]));
    for (int32_t ow_outer = 0; ow_outer < 4; ++ow_outer) {
      for (int32_t oc_block_c_init = 0; oc_block_c_init < 4; ++oc_block_c_init) {
        ((float*)conv2d_NCHWc_global_let)[oc_block_c_init] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_1 = 0; oc_block_c_init_1 < 4; ++oc_block_c_init_1) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_1 + 4)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_2 = 0; oc_block_c_init_2 < 4; ++oc_block_c_init_2) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_2 + 8)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_3 = 0; oc_block_c_init_3 < 4; ++oc_block_c_init_3) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_3 + 12)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_4 = 0; oc_block_c_init_4 < 4; ++oc_block_c_init_4) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_4 + 16)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_5 = 0; oc_block_c_init_5 < 4; ++oc_block_c_init_5) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_5 + 20)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_6 = 0; oc_block_c_init_6 < 4; ++oc_block_c_init_6) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_6 + 24)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_7 = 0; oc_block_c_init_7 < 4; ++oc_block_c_init_7) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_7 + 28)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_8 = 0; oc_block_c_init_8 < 4; ++oc_block_c_init_8) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_8 + 32)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_9 = 0; oc_block_c_init_9 < 4; ++oc_block_c_init_9) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_9 + 36)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_10 = 0; oc_block_c_init_10 < 4; ++oc_block_c_init_10) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_10 + 40)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_11 = 0; oc_block_c_init_11 < 4; ++oc_block_c_init_11) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_11 + 44)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_12 = 0; oc_block_c_init_12 < 4; ++oc_block_c_init_12) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_12 + 48)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_13 = 0; oc_block_c_init_13 < 4; ++oc_block_c_init_13) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_13 + 52)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_14 = 0; oc_block_c_init_14 < 4; ++oc_block_c_init_14) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_14 + 56)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_15 = 0; oc_block_c_init_15 < 4; ++oc_block_c_init_15) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_15 + 60)] = 0.000000e+00f;
      }
      for (int32_t ic_outer = 0; ic_outer < 16; ++ic_outer) {
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            for (int32_t ic_inner = 0; ic_inner < 4; ++ic_inner) {
              for (int32_t oc_block_c = 0; oc_block_c < 4; ++oc_block_c) {
                ((float*)conv2d_NCHWc_global_let)[oc_block_c] = (((float*)conv2d_NCHWc_global_let)[oc_block_c] + (((float*)data_pad_let)[((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c)]));
              }
              for (int32_t oc_block_c_1 = 0; oc_block_c_1 < 4; ++oc_block_c_1) {
                int32_t cse_var_3 = (oc_block_c_1 + 4);
                ((float*)conv2d_NCHWc_global_let)[cse_var_3] = (((float*)conv2d_NCHWc_global_let)[cse_var_3] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 4)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_1)]));
              }
              for (int32_t oc_block_c_2 = 0; oc_block_c_2 < 4; ++oc_block_c_2) {
                int32_t cse_var_4 = (oc_block_c_2 + 8);
                ((float*)conv2d_NCHWc_global_let)[cse_var_4] = (((float*)conv2d_NCHWc_global_let)[cse_var_4] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 8)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_2)]));
              }
              for (int32_t oc_block_c_3 = 0; oc_block_c_3 < 4; ++oc_block_c_3) {
                int32_t cse_var_5 = (oc_block_c_3 + 12);
                ((float*)conv2d_NCHWc_global_let)[cse_var_5] = (((float*)conv2d_NCHWc_global_let)[cse_var_5] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 12)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_3)]));
              }
              for (int32_t oc_block_c_4 = 0; oc_block_c_4 < 4; ++oc_block_c_4) {
                int32_t cse_var_6 = (oc_block_c_4 + 16);
                ((float*)conv2d_NCHWc_global_let)[cse_var_6] = (((float*)conv2d_NCHWc_global_let)[cse_var_6] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 16)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_4)]));
              }
              for (int32_t oc_block_c_5 = 0; oc_block_c_5 < 4; ++oc_block_c_5) {
                int32_t cse_var_7 = (oc_block_c_5 + 20);
                ((float*)conv2d_NCHWc_global_let)[cse_var_7] = (((float*)conv2d_NCHWc_global_let)[cse_var_7] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 20)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_5)]));
              }
              for (int32_t oc_block_c_6 = 0; oc_block_c_6 < 4; ++oc_block_c_6) {
                int32_t cse_var_8 = (oc_block_c_6 + 24);
                ((float*)conv2d_NCHWc_global_let)[cse_var_8] = (((float*)conv2d_NCHWc_global_let)[cse_var_8] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 24)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_6)]));
              }
              for (int32_t oc_block_c_7 = 0; oc_block_c_7 < 4; ++oc_block_c_7) {
                int32_t cse_var_9 = (oc_block_c_7 + 28);
                ((float*)conv2d_NCHWc_global_let)[cse_var_9] = (((float*)conv2d_NCHWc_global_let)[cse_var_9] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 28)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_7)]));
              }
              for (int32_t oc_block_c_8 = 0; oc_block_c_8 < 4; ++oc_block_c_8) {
                int32_t cse_var_10 = (oc_block_c_8 + 32);
                ((float*)conv2d_NCHWc_global_let)[cse_var_10] = (((float*)conv2d_NCHWc_global_let)[cse_var_10] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 32)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_8)]));
              }
              for (int32_t oc_block_c_9 = 0; oc_block_c_9 < 4; ++oc_block_c_9) {
                int32_t cse_var_11 = (oc_block_c_9 + 36);
                ((float*)conv2d_NCHWc_global_let)[cse_var_11] = (((float*)conv2d_NCHWc_global_let)[cse_var_11] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 36)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_9)]));
              }
              for (int32_t oc_block_c_10 = 0; oc_block_c_10 < 4; ++oc_block_c_10) {
                int32_t cse_var_12 = (oc_block_c_10 + 40);
                ((float*)conv2d_NCHWc_global_let)[cse_var_12] = (((float*)conv2d_NCHWc_global_let)[cse_var_12] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 40)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_10)]));
              }
              for (int32_t oc_block_c_11 = 0; oc_block_c_11 < 4; ++oc_block_c_11) {
                int32_t cse_var_13 = (oc_block_c_11 + 44);
                ((float*)conv2d_NCHWc_global_let)[cse_var_13] = (((float*)conv2d_NCHWc_global_let)[cse_var_13] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 44)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_11)]));
              }
              for (int32_t oc_block_c_12 = 0; oc_block_c_12 < 4; ++oc_block_c_12) {
                int32_t cse_var_14 = (oc_block_c_12 + 48);
                ((float*)conv2d_NCHWc_global_let)[cse_var_14] = (((float*)conv2d_NCHWc_global_let)[cse_var_14] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 48)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_12)]));
              }
              for (int32_t oc_block_c_13 = 0; oc_block_c_13 < 4; ++oc_block_c_13) {
                int32_t cse_var_15 = (oc_block_c_13 + 52);
                ((float*)conv2d_NCHWc_global_let)[cse_var_15] = (((float*)conv2d_NCHWc_global_let)[cse_var_15] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 52)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_13)]));
              }
              for (int32_t oc_block_c_14 = 0; oc_block_c_14 < 4; ++oc_block_c_14) {
                int32_t cse_var_16 = (oc_block_c_14 + 56);
                ((float*)conv2d_NCHWc_global_let)[cse_var_16] = (((float*)conv2d_NCHWc_global_let)[cse_var_16] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 56)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_14)]));
              }
              for (int32_t oc_block_c_15 = 0; oc_block_c_15 < 4; ++oc_block_c_15) {
                int32_t cse_var_17 = (oc_block_c_15 + 60);
                ((float*)conv2d_NCHWc_global_let)[cse_var_17] = (((float*)conv2d_NCHWc_global_let)[cse_var_17] + (((float*)data_pad_let)[(((((((ic_outer * 17424) + (kh * 264)) + ((ax0_ax1_fused_ax2_fused & 63) * 264)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 60)] * ((float*)fused_constant_3_let)[(((((((ax0_ax1_fused_ax2_fused >> 6) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_15)]));
              }
            }
          }
        }
      }
      for (int32_t ow_inner = 0; ow_inner < 16; ++ow_inner) {
        for (int32_t oc_block = 0; oc_block < 4; ++oc_block) {
          int32_t cse_var_18 = (ow_inner * 4);
          ((float*)conv2d_NCHWc_let)[(((ow_outer * 64) + cse_var_18) + oc_block)] = ((float*)conv2d_NCHWc_global_let)[(cse_var_18 + oc_block)];
        }
      }
    }
    for (int32_t ax3_outer = 0; ax3_outer < 4; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
          int32_t cse_var_20 = (ax3_outer * 64);
          int32_t cse_var_19 = (ax3_inner * 4);
          float v_ = ((float*)conv2d_NCHWc_let)[((cse_var_20 + cse_var_19) + ax4)] + ((float*)fused_nn_contrib_conv2d_NCHWc_constant_3_let)[(((ax0_ax1_fused_ax2_fused >> 6) * 4) + ax4)];
          T_relu[((((ax0_ax1_fused_ax2_fused * 256) + cse_var_20) + cse_var_19) + ax4)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_4(float* p0, float* T_relu, uint8_t* global_const_workspace_16_var, uint8_t* global_workspace_17_var) {
  void* fused_nn_contrib_conv2d_NCHWc_constant_4_let = (&(global_const_workspace_16_var[1831552]));
  void* fused_constant_4_let = (&(global_const_workspace_16_var[589824]));
  void* data_pad_let = (&(global_workspace_17_var[1116160]));
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 544; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 34; ++i3) {
      for (int32_t i4 = 0; i4 < 4; ++i4) {
        int32_t cse_var_2 = (i0_i1_fused_i2_fused % 34);
        int32_t cse_var_1 = (i3 * 4);
        ((float*)data_pad_let)[(((i0_i1_fused_i2_fused * 136) + cse_var_1) + i4)] = (((((1 <= cse_var_2) && (cse_var_2 < 33)) && (1 <= i3)) && (i3 < 33)) ? p0[((((((i0_i1_fused_i2_fused / 34) * 4096) + (cse_var_2 * 128)) + cse_var_1) + i4) - 132)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1024; ++ax0_ax1_fused_ax2_fused) {
    void* conv2d_NCHWc_let = (&(global_workspace_17_var[1412096]));
    void* conv2d_NCHWc_global_let = (&(global_workspace_17_var[1412608]));
    for (int32_t ow_outer = 0; ow_outer < 2; ++ow_outer) {
      for (int32_t oc_block_c_init = 0; oc_block_c_init < 4; ++oc_block_c_init) {
        ((float*)conv2d_NCHWc_global_let)[oc_block_c_init] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_1 = 0; oc_block_c_init_1 < 4; ++oc_block_c_init_1) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_1 + 4)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_2 = 0; oc_block_c_init_2 < 4; ++oc_block_c_init_2) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_2 + 8)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_3 = 0; oc_block_c_init_3 < 4; ++oc_block_c_init_3) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_3 + 12)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_4 = 0; oc_block_c_init_4 < 4; ++oc_block_c_init_4) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_4 + 16)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_5 = 0; oc_block_c_init_5 < 4; ++oc_block_c_init_5) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_5 + 20)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_6 = 0; oc_block_c_init_6 < 4; ++oc_block_c_init_6) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_6 + 24)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_7 = 0; oc_block_c_init_7 < 4; ++oc_block_c_init_7) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_7 + 28)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_8 = 0; oc_block_c_init_8 < 4; ++oc_block_c_init_8) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_8 + 32)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_9 = 0; oc_block_c_init_9 < 4; ++oc_block_c_init_9) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_9 + 36)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_10 = 0; oc_block_c_init_10 < 4; ++oc_block_c_init_10) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_10 + 40)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_11 = 0; oc_block_c_init_11 < 4; ++oc_block_c_init_11) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_11 + 44)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_12 = 0; oc_block_c_init_12 < 4; ++oc_block_c_init_12) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_12 + 48)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_13 = 0; oc_block_c_init_13 < 4; ++oc_block_c_init_13) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_13 + 52)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_14 = 0; oc_block_c_init_14 < 4; ++oc_block_c_init_14) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_14 + 56)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_15 = 0; oc_block_c_init_15 < 4; ++oc_block_c_init_15) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_15 + 60)] = 0.000000e+00f;
      }
      for (int32_t ic_outer = 0; ic_outer < 16; ++ic_outer) {
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            for (int32_t ic_inner = 0; ic_inner < 4; ++ic_inner) {
              for (int32_t oc_block_c = 0; oc_block_c < 4; ++oc_block_c) {
                ((float*)conv2d_NCHWc_global_let)[oc_block_c] = (((float*)conv2d_NCHWc_global_let)[oc_block_c] + (((float*)data_pad_let)[((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c)]));
              }
              for (int32_t oc_block_c_1 = 0; oc_block_c_1 < 4; ++oc_block_c_1) {
                int32_t cse_var_3 = (oc_block_c_1 + 4);
                ((float*)conv2d_NCHWc_global_let)[cse_var_3] = (((float*)conv2d_NCHWc_global_let)[cse_var_3] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 4)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_1)]));
              }
              for (int32_t oc_block_c_2 = 0; oc_block_c_2 < 4; ++oc_block_c_2) {
                int32_t cse_var_4 = (oc_block_c_2 + 8);
                ((float*)conv2d_NCHWc_global_let)[cse_var_4] = (((float*)conv2d_NCHWc_global_let)[cse_var_4] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 8)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_2)]));
              }
              for (int32_t oc_block_c_3 = 0; oc_block_c_3 < 4; ++oc_block_c_3) {
                int32_t cse_var_5 = (oc_block_c_3 + 12);
                ((float*)conv2d_NCHWc_global_let)[cse_var_5] = (((float*)conv2d_NCHWc_global_let)[cse_var_5] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 12)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_3)]));
              }
              for (int32_t oc_block_c_4 = 0; oc_block_c_4 < 4; ++oc_block_c_4) {
                int32_t cse_var_6 = (oc_block_c_4 + 16);
                ((float*)conv2d_NCHWc_global_let)[cse_var_6] = (((float*)conv2d_NCHWc_global_let)[cse_var_6] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 16)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_4)]));
              }
              for (int32_t oc_block_c_5 = 0; oc_block_c_5 < 4; ++oc_block_c_5) {
                int32_t cse_var_7 = (oc_block_c_5 + 20);
                ((float*)conv2d_NCHWc_global_let)[cse_var_7] = (((float*)conv2d_NCHWc_global_let)[cse_var_7] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 20)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_5)]));
              }
              for (int32_t oc_block_c_6 = 0; oc_block_c_6 < 4; ++oc_block_c_6) {
                int32_t cse_var_8 = (oc_block_c_6 + 24);
                ((float*)conv2d_NCHWc_global_let)[cse_var_8] = (((float*)conv2d_NCHWc_global_let)[cse_var_8] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 24)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_6)]));
              }
              for (int32_t oc_block_c_7 = 0; oc_block_c_7 < 4; ++oc_block_c_7) {
                int32_t cse_var_9 = (oc_block_c_7 + 28);
                ((float*)conv2d_NCHWc_global_let)[cse_var_9] = (((float*)conv2d_NCHWc_global_let)[cse_var_9] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 28)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_7)]));
              }
              for (int32_t oc_block_c_8 = 0; oc_block_c_8 < 4; ++oc_block_c_8) {
                int32_t cse_var_10 = (oc_block_c_8 + 32);
                ((float*)conv2d_NCHWc_global_let)[cse_var_10] = (((float*)conv2d_NCHWc_global_let)[cse_var_10] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 32)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_8)]));
              }
              for (int32_t oc_block_c_9 = 0; oc_block_c_9 < 4; ++oc_block_c_9) {
                int32_t cse_var_11 = (oc_block_c_9 + 36);
                ((float*)conv2d_NCHWc_global_let)[cse_var_11] = (((float*)conv2d_NCHWc_global_let)[cse_var_11] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 36)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_9)]));
              }
              for (int32_t oc_block_c_10 = 0; oc_block_c_10 < 4; ++oc_block_c_10) {
                int32_t cse_var_12 = (oc_block_c_10 + 40);
                ((float*)conv2d_NCHWc_global_let)[cse_var_12] = (((float*)conv2d_NCHWc_global_let)[cse_var_12] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 40)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_10)]));
              }
              for (int32_t oc_block_c_11 = 0; oc_block_c_11 < 4; ++oc_block_c_11) {
                int32_t cse_var_13 = (oc_block_c_11 + 44);
                ((float*)conv2d_NCHWc_global_let)[cse_var_13] = (((float*)conv2d_NCHWc_global_let)[cse_var_13] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 44)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_11)]));
              }
              for (int32_t oc_block_c_12 = 0; oc_block_c_12 < 4; ++oc_block_c_12) {
                int32_t cse_var_14 = (oc_block_c_12 + 48);
                ((float*)conv2d_NCHWc_global_let)[cse_var_14] = (((float*)conv2d_NCHWc_global_let)[cse_var_14] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 48)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_12)]));
              }
              for (int32_t oc_block_c_13 = 0; oc_block_c_13 < 4; ++oc_block_c_13) {
                int32_t cse_var_15 = (oc_block_c_13 + 52);
                ((float*)conv2d_NCHWc_global_let)[cse_var_15] = (((float*)conv2d_NCHWc_global_let)[cse_var_15] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 52)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_13)]));
              }
              for (int32_t oc_block_c_14 = 0; oc_block_c_14 < 4; ++oc_block_c_14) {
                int32_t cse_var_16 = (oc_block_c_14 + 56);
                ((float*)conv2d_NCHWc_global_let)[cse_var_16] = (((float*)conv2d_NCHWc_global_let)[cse_var_16] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 56)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_14)]));
              }
              for (int32_t oc_block_c_15 = 0; oc_block_c_15 < 4; ++oc_block_c_15) {
                int32_t cse_var_17 = (oc_block_c_15 + 60);
                ((float*)conv2d_NCHWc_global_let)[cse_var_17] = (((float*)conv2d_NCHWc_global_let)[cse_var_17] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 60)] * ((float*)fused_constant_4_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 2304) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_15)]));
              }
            }
          }
        }
      }
      for (int32_t ow_inner = 0; ow_inner < 16; ++ow_inner) {
        for (int32_t oc_block = 0; oc_block < 4; ++oc_block) {
          int32_t cse_var_18 = (ow_inner * 4);
          ((float*)conv2d_NCHWc_let)[(((ow_outer * 64) + cse_var_18) + oc_block)] = ((float*)conv2d_NCHWc_global_let)[(cse_var_18 + oc_block)];
        }
      }
    }
    for (int32_t ax3_outer = 0; ax3_outer < 2; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
          int32_t cse_var_20 = (ax3_outer * 64);
          int32_t cse_var_19 = (ax3_inner * 4);
          float v_ = ((float*)conv2d_NCHWc_let)[((cse_var_20 + cse_var_19) + ax4)] + ((float*)fused_nn_contrib_conv2d_NCHWc_constant_4_let)[(((ax0_ax1_fused_ax2_fused >> 5) * 4) + ax4)];
          T_relu[((((ax0_ax1_fused_ax2_fused * 128) + cse_var_20) + cse_var_19) + ax4)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_5(float* p0, float* T_relu, uint8_t* global_const_workspace_18_var, uint8_t* global_workspace_19_var) {
  void* fused_nn_contrib_conv2d_NCHWc_constant_5_let = (&(global_const_workspace_18_var[1831040]));
  void* fused_constant_5_let = (&(global_const_workspace_18_var[0]));
  void* data_pad_let = (&(global_workspace_19_var[0]));
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1088; ++i0_i1_fused_i2_fused) {
    for (int32_t i3 = 0; i3 < 34; ++i3) {
      for (int32_t i4 = 0; i4 < 4; ++i4) {
        int32_t cse_var_2 = (i0_i1_fused_i2_fused % 34);
        int32_t cse_var_1 = (i3 * 4);
        ((float*)data_pad_let)[(((i0_i1_fused_i2_fused * 136) + cse_var_1) + i4)] = (((((1 <= cse_var_2) && (cse_var_2 < 33)) && (1 <= i3)) && (i3 < 33)) ? p0[((((((i0_i1_fused_i2_fused / 34) * 4096) + (cse_var_2 * 128)) + cse_var_1) + i4) - 132)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1024; ++ax0_ax1_fused_ax2_fused) {
    void* conv2d_NCHWc_let = (&(global_workspace_19_var[1116160]));
    void* conv2d_NCHWc_global_let = (&(global_workspace_19_var[1116672]));
    for (int32_t ow_outer = 0; ow_outer < 2; ++ow_outer) {
      for (int32_t oc_block_c_init = 0; oc_block_c_init < 4; ++oc_block_c_init) {
        ((float*)conv2d_NCHWc_global_let)[oc_block_c_init] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_1 = 0; oc_block_c_init_1 < 4; ++oc_block_c_init_1) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_1 + 4)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_2 = 0; oc_block_c_init_2 < 4; ++oc_block_c_init_2) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_2 + 8)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_3 = 0; oc_block_c_init_3 < 4; ++oc_block_c_init_3) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_3 + 12)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_4 = 0; oc_block_c_init_4 < 4; ++oc_block_c_init_4) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_4 + 16)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_5 = 0; oc_block_c_init_5 < 4; ++oc_block_c_init_5) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_5 + 20)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_6 = 0; oc_block_c_init_6 < 4; ++oc_block_c_init_6) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_6 + 24)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_7 = 0; oc_block_c_init_7 < 4; ++oc_block_c_init_7) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_7 + 28)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_8 = 0; oc_block_c_init_8 < 4; ++oc_block_c_init_8) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_8 + 32)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_9 = 0; oc_block_c_init_9 < 4; ++oc_block_c_init_9) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_9 + 36)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_10 = 0; oc_block_c_init_10 < 4; ++oc_block_c_init_10) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_10 + 40)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_11 = 0; oc_block_c_init_11 < 4; ++oc_block_c_init_11) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_11 + 44)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_12 = 0; oc_block_c_init_12 < 4; ++oc_block_c_init_12) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_12 + 48)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_13 = 0; oc_block_c_init_13 < 4; ++oc_block_c_init_13) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_13 + 52)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_14 = 0; oc_block_c_init_14 < 4; ++oc_block_c_init_14) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_14 + 56)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_15 = 0; oc_block_c_init_15 < 4; ++oc_block_c_init_15) {
        ((float*)conv2d_NCHWc_global_let)[(oc_block_c_init_15 + 60)] = 0.000000e+00f;
      }
      for (int32_t ic_outer = 0; ic_outer < 32; ++ic_outer) {
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            for (int32_t ic_inner = 0; ic_inner < 4; ++ic_inner) {
              for (int32_t oc_block_c = 0; oc_block_c < 4; ++oc_block_c) {
                ((float*)conv2d_NCHWc_global_let)[oc_block_c] = (((float*)conv2d_NCHWc_global_let)[oc_block_c] + (((float*)data_pad_let)[((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c)]));
              }
              for (int32_t oc_block_c_1 = 0; oc_block_c_1 < 4; ++oc_block_c_1) {
                int32_t cse_var_3 = (oc_block_c_1 + 4);
                ((float*)conv2d_NCHWc_global_let)[cse_var_3] = (((float*)conv2d_NCHWc_global_let)[cse_var_3] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 4)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_1)]));
              }
              for (int32_t oc_block_c_2 = 0; oc_block_c_2 < 4; ++oc_block_c_2) {
                int32_t cse_var_4 = (oc_block_c_2 + 8);
                ((float*)conv2d_NCHWc_global_let)[cse_var_4] = (((float*)conv2d_NCHWc_global_let)[cse_var_4] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 8)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_2)]));
              }
              for (int32_t oc_block_c_3 = 0; oc_block_c_3 < 4; ++oc_block_c_3) {
                int32_t cse_var_5 = (oc_block_c_3 + 12);
                ((float*)conv2d_NCHWc_global_let)[cse_var_5] = (((float*)conv2d_NCHWc_global_let)[cse_var_5] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 12)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_3)]));
              }
              for (int32_t oc_block_c_4 = 0; oc_block_c_4 < 4; ++oc_block_c_4) {
                int32_t cse_var_6 = (oc_block_c_4 + 16);
                ((float*)conv2d_NCHWc_global_let)[cse_var_6] = (((float*)conv2d_NCHWc_global_let)[cse_var_6] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 16)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_4)]));
              }
              for (int32_t oc_block_c_5 = 0; oc_block_c_5 < 4; ++oc_block_c_5) {
                int32_t cse_var_7 = (oc_block_c_5 + 20);
                ((float*)conv2d_NCHWc_global_let)[cse_var_7] = (((float*)conv2d_NCHWc_global_let)[cse_var_7] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 20)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_5)]));
              }
              for (int32_t oc_block_c_6 = 0; oc_block_c_6 < 4; ++oc_block_c_6) {
                int32_t cse_var_8 = (oc_block_c_6 + 24);
                ((float*)conv2d_NCHWc_global_let)[cse_var_8] = (((float*)conv2d_NCHWc_global_let)[cse_var_8] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 24)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_6)]));
              }
              for (int32_t oc_block_c_7 = 0; oc_block_c_7 < 4; ++oc_block_c_7) {
                int32_t cse_var_9 = (oc_block_c_7 + 28);
                ((float*)conv2d_NCHWc_global_let)[cse_var_9] = (((float*)conv2d_NCHWc_global_let)[cse_var_9] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 28)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_7)]));
              }
              for (int32_t oc_block_c_8 = 0; oc_block_c_8 < 4; ++oc_block_c_8) {
                int32_t cse_var_10 = (oc_block_c_8 + 32);
                ((float*)conv2d_NCHWc_global_let)[cse_var_10] = (((float*)conv2d_NCHWc_global_let)[cse_var_10] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 32)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_8)]));
              }
              for (int32_t oc_block_c_9 = 0; oc_block_c_9 < 4; ++oc_block_c_9) {
                int32_t cse_var_11 = (oc_block_c_9 + 36);
                ((float*)conv2d_NCHWc_global_let)[cse_var_11] = (((float*)conv2d_NCHWc_global_let)[cse_var_11] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 36)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_9)]));
              }
              for (int32_t oc_block_c_10 = 0; oc_block_c_10 < 4; ++oc_block_c_10) {
                int32_t cse_var_12 = (oc_block_c_10 + 40);
                ((float*)conv2d_NCHWc_global_let)[cse_var_12] = (((float*)conv2d_NCHWc_global_let)[cse_var_12] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 40)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_10)]));
              }
              for (int32_t oc_block_c_11 = 0; oc_block_c_11 < 4; ++oc_block_c_11) {
                int32_t cse_var_13 = (oc_block_c_11 + 44);
                ((float*)conv2d_NCHWc_global_let)[cse_var_13] = (((float*)conv2d_NCHWc_global_let)[cse_var_13] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 44)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_11)]));
              }
              for (int32_t oc_block_c_12 = 0; oc_block_c_12 < 4; ++oc_block_c_12) {
                int32_t cse_var_14 = (oc_block_c_12 + 48);
                ((float*)conv2d_NCHWc_global_let)[cse_var_14] = (((float*)conv2d_NCHWc_global_let)[cse_var_14] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 48)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_12)]));
              }
              for (int32_t oc_block_c_13 = 0; oc_block_c_13 < 4; ++oc_block_c_13) {
                int32_t cse_var_15 = (oc_block_c_13 + 52);
                ((float*)conv2d_NCHWc_global_let)[cse_var_15] = (((float*)conv2d_NCHWc_global_let)[cse_var_15] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 52)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_13)]));
              }
              for (int32_t oc_block_c_14 = 0; oc_block_c_14 < 4; ++oc_block_c_14) {
                int32_t cse_var_16 = (oc_block_c_14 + 56);
                ((float*)conv2d_NCHWc_global_let)[cse_var_16] = (((float*)conv2d_NCHWc_global_let)[cse_var_16] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 56)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_14)]));
              }
              for (int32_t oc_block_c_15 = 0; oc_block_c_15 < 4; ++oc_block_c_15) {
                int32_t cse_var_17 = (oc_block_c_15 + 60);
                ((float*)conv2d_NCHWc_global_let)[cse_var_17] = (((float*)conv2d_NCHWc_global_let)[cse_var_17] + (((float*)data_pad_let)[(((((((ic_outer * 4624) + (kh * 136)) + ((ax0_ax1_fused_ax2_fused & 31) * 136)) + (ow_outer * 64)) + (kw * 4)) + ic_inner) + 60)] * ((float*)fused_constant_5_let)[(((((((ax0_ax1_fused_ax2_fused >> 5) * 4608) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_15)]));
              }
            }
          }
        }
      }
      for (int32_t ow_inner = 0; ow_inner < 16; ++ow_inner) {
        for (int32_t oc_block = 0; oc_block < 4; ++oc_block) {
          int32_t cse_var_18 = (ow_inner * 4);
          ((float*)conv2d_NCHWc_let)[(((ow_outer * 64) + cse_var_18) + oc_block)] = ((float*)conv2d_NCHWc_global_let)[(cse_var_18 + oc_block)];
        }
      }
    }
    for (int32_t ax3_outer = 0; ax3_outer < 2; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
          int32_t cse_var_20 = (ax3_outer * 64);
          int32_t cse_var_19 = (ax3_inner * 4);
          float v_ = ((float*)conv2d_NCHWc_let)[((cse_var_20 + cse_var_19) + ax4)] + ((float*)fused_nn_contrib_conv2d_NCHWc_constant_5_let)[(((ax0_ax1_fused_ax2_fused >> 5) * 4) + ax4)];
          T_relu[((((ax0_ax1_fused_ax2_fused * 128) + cse_var_20) + cse_var_19) + ax4)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add(float* p0, float* T_add, uint8_t* global_const_workspace_44_var, uint8_t* global_workspace_45_var) {
  void* fused_nn_contrib_dense_pack_constant_5_let = (&(global_const_workspace_44_var[1834944]));
  void* fused_constant_11_let = (&(global_const_workspace_44_var[1817600]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 2; ++ax1_outer_ax0_outer_fused) {
    void* compute_global_let = (&(global_workspace_45_var[2112]));
    for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
      ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
    }
    for (int32_t k_outer = 0; k_outer < 128; ++k_outer) {
      for (int32_t x_c = 0; x_c < 8; ++x_c) {
        ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_11_let)[(((ax1_outer_ax0_outer_fused * 1024) + (k_outer * 8)) + x_c)]));
      }
    }
    for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
      int32_t cse_var_1 = ((ax1_outer_ax0_outer_fused * 8) + ax1_inner_inner);
      T_add[cse_var_1] = (((float*)compute_global_let)[ax1_inner_inner] + ((float*)fused_nn_contrib_dense_pack_constant_5_let)[cse_var_1]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_1(float* p0, float* T_add, uint8_t* global_const_workspace_50_var, uint8_t* global_workspace_51_var) {
  void* fused_nn_contrib_dense_pack_constant_7_let = (&(global_const_workspace_50_var[1835008]));
  void* fused_constant_13_let = (&(global_const_workspace_50_var[1827968]));
  void* compute_global_let = (&(global_workspace_51_var[1792]));
  for (int32_t x_c_init = 0; x_c_init < 4; ++x_c_init) {
    ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
  }
  for (int32_t k_outer = 0; k_outer < 64; ++k_outer) {
    for (int32_t x_c = 0; x_c < 4; ++x_c) {
      ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_13_let)[((k_outer * 4) + x_c)]));
    }
  }
  for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 4; ++ax1_inner_inner) {
    T_add[ax1_inner_inner] = (((float*)compute_global_let)[ax1_inner_inner] + ((float*)fused_nn_contrib_dense_pack_constant_7_let)[ax1_inner_inner]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu(float* p0, float* p1, float* T_relu, uint8_t* global_const_workspace_32_var, uint8_t* global_workspace_33_var) {
  void* fused_nn_contrib_dense_pack_constant_let = (&(global_const_workspace_32_var[1830528]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 4; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_33_var[53664]));
    void* compute_global_let = (&(global_workspace_33_var[53792]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 4; ++y_inner_outer_x_inner_outer_fused) {
      for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
        ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
      }
      for (int32_t k_outer = 0; k_outer < 102; ++k_outer) {
        for (int32_t x_c = 0; x_c < 8; ++x_c) {
          ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * p1[((((ax1_outer_ax0_outer_fused * 3264) + (y_inner_outer_x_inner_outer_fused * 816)) + (k_outer * 8)) + x_c)]));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 8; ++x_inner_inner) {
        ((float*)compute_let)[((y_inner_outer_x_inner_outer_fused * 8) + x_inner_inner)] = ((float*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 4; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
        int32_t cse_var_2 = (ax1_inner_outer * 8);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 32) + cse_var_2) + ax1_inner_inner);
        float v_ = ((float*)compute_let)[(cse_var_2 + ax1_inner_inner)] + ((float*)fused_nn_contrib_dense_pack_constant_let)[cse_var_1];
        T_relu[cse_var_1] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_1(float* p0, float* T_relu, uint8_t* global_const_workspace_34_var, uint8_t* global_workspace_35_var) {
  void* fused_nn_contrib_dense_pack_constant_1_let = (&(global_const_workspace_34_var[1830016]));
  void* fused_constant_7_let = (&(global_const_workspace_34_var[1499136]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 4; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_35_var[53760]));
    void* compute_global_let = (&(global_workspace_35_var[53888]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 4; ++y_inner_outer_x_inner_outer_fused) {
      for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
        ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
      }
      for (int32_t k_outer = 0; k_outer < 128; ++k_outer) {
        for (int32_t x_c = 0; x_c < 8; ++x_c) {
          ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_7_let)[((((ax1_outer_ax0_outer_fused * 4096) + (y_inner_outer_x_inner_outer_fused * 1024)) + (k_outer * 8)) + x_c)]));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 8; ++x_inner_inner) {
        ((float*)compute_let)[((y_inner_outer_x_inner_outer_fused * 8) + x_inner_inner)] = ((float*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 4; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
        int32_t cse_var_2 = (ax1_inner_outer * 8);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 32) + cse_var_2) + ax1_inner_inner);
        float v_ = ((float*)compute_let)[(cse_var_2 + ax1_inner_inner)] + ((float*)fused_nn_contrib_dense_pack_constant_1_let)[cse_var_1];
        T_relu[cse_var_1] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_2(float* p0, float* T_relu, uint8_t* global_const_workspace_38_var, uint8_t* global_workspace_39_var) {
  void* fused_nn_contrib_dense_pack_constant_2_let = (&(global_const_workspace_38_var[1826944]));
  void* fused_constant_8_let = (&(global_const_workspace_38_var[884736]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 4; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_39_var[2048]));
    void* compute_global_let = (&(global_workspace_39_var[2304]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 8; ++y_inner_outer_x_inner_outer_fused) {
      for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
        ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
      }
      for (int32_t k_outer = 0; k_outer < 256; ++k_outer) {
        for (int32_t x_c = 0; x_c < 8; ++x_c) {
          ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_8_let)[((((ax1_outer_ax0_outer_fused * 16384) + (y_inner_outer_x_inner_outer_fused * 2048)) + (k_outer * 8)) + x_c)]));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 8; ++x_inner_inner) {
        ((float*)compute_let)[((y_inner_outer_x_inner_outer_fused * 8) + x_inner_inner)] = ((float*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 8; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
        int32_t cse_var_2 = (ax1_inner_outer * 8);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 64) + cse_var_2) + ax1_inner_inner);
        float v_ = ((float*)compute_let)[(cse_var_2 + ax1_inner_inner)] + ((float*)fused_nn_contrib_dense_pack_constant_2_let)[cse_var_1];
        T_relu[cse_var_1] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_3(float* p0, float* T_relu, uint8_t* global_const_workspace_40_var, uint8_t* global_workspace_41_var) {
  void* fused_nn_contrib_dense_pack_constant_3_let = (&(global_const_workspace_40_var[1829504]));
  void* fused_constant_9_let = (&(global_const_workspace_40_var[1294336]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 4; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_41_var[1536]));
    void* compute_global_let = (&(global_workspace_41_var[1664]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 4; ++y_inner_outer_x_inner_outer_fused) {
      for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
        ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
      }
      for (int32_t k_outer = 0; k_outer < 256; ++k_outer) {
        for (int32_t x_c = 0; x_c < 8; ++x_c) {
          ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_9_let)[((((ax1_outer_ax0_outer_fused * 8192) + (y_inner_outer_x_inner_outer_fused * 2048)) + (k_outer * 8)) + x_c)]));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 8; ++x_inner_inner) {
        ((float*)compute_let)[((y_inner_outer_x_inner_outer_fused * 8) + x_inner_inner)] = ((float*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 4; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
        int32_t cse_var_2 = (ax1_inner_outer * 8);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 32) + cse_var_2) + ax1_inner_inner);
        float v_ = ((float*)compute_let)[(cse_var_2 + ax1_inner_inner)] + ((float*)fused_nn_contrib_dense_pack_constant_3_let)[cse_var_1];
        T_relu[cse_var_1] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_4(float* p0, float* T_relu, uint8_t* global_const_workspace_42_var, uint8_t* global_workspace_43_var) {
  void* fused_nn_contrib_dense_pack_constant_4_let = (&(global_const_workspace_42_var[1828992]));
  void* fused_constant_10_let = (&(global_const_workspace_42_var[1564672]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 4; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_43_var[2048]));
    void* compute_global_let = (&(global_workspace_43_var[2176]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 4; ++y_inner_outer_x_inner_outer_fused) {
      for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
        ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
      }
      for (int32_t k_outer = 0; k_outer < 128; ++k_outer) {
        for (int32_t x_c = 0; x_c < 8; ++x_c) {
          ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_10_let)[((((ax1_outer_ax0_outer_fused * 4096) + (y_inner_outer_x_inner_outer_fused * 1024)) + (k_outer * 8)) + x_c)]));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 8; ++x_inner_inner) {
        ((float*)compute_let)[((y_inner_outer_x_inner_outer_fused * 8) + x_inner_inner)] = ((float*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 4; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
        int32_t cse_var_2 = (ax1_inner_outer * 8);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 32) + cse_var_2) + ax1_inner_inner);
        float v_ = ((float*)compute_let)[(cse_var_2 + ax1_inner_inner)] + ((float*)fused_nn_contrib_dense_pack_constant_4_let)[cse_var_1];
        T_relu[cse_var_1] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_5(float* p0, float* T_relu, uint8_t* global_const_workspace_48_var, uint8_t* global_workspace_49_var) {
  void* fused_nn_contrib_dense_pack_constant_6_let = (&(global_const_workspace_48_var[1833152]));
  void* fused_constant_12_let = (&(global_const_workspace_48_var[1784832]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 4; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_49_var[1792]));
    void* compute_global_let = (&(global_workspace_49_var[1856]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 2; ++y_inner_outer_x_inner_outer_fused) {
      for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
        ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
      }
      for (int32_t k_outer = 0; k_outer < 128; ++k_outer) {
        for (int32_t x_c = 0; x_c < 8; ++x_c) {
          ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_12_let)[((((ax1_outer_ax0_outer_fused * 2048) + (y_inner_outer_x_inner_outer_fused * 1024)) + (k_outer * 8)) + x_c)]));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 8; ++x_inner_inner) {
        ((float*)compute_let)[((y_inner_outer_x_inner_outer_fused * 8) + x_inner_inner)] = ((float*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 2; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
        int32_t cse_var_2 = (ax1_inner_outer * 8);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 16) + cse_var_2) + ax1_inner_inner);
        float v_ = ((float*)compute_let)[(cse_var_2 + ax1_inner_inner)] + ((float*)fused_nn_contrib_dense_pack_constant_6_let)[cse_var_1];
        T_relu[cse_var_1] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_6(float* p0, float* T_relu, uint8_t* global_const_workspace_54_var, uint8_t* global_workspace_55_var) {
  void* fused_nn_contrib_dense_pack_constant_8_let = (&(global_const_workspace_54_var[1832896]));
  void* fused_constant_14_let = (&(global_const_workspace_54_var[1752064]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 4; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_55_var[1792]));
    void* compute_global_let = (&(global_workspace_55_var[1856]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 2; ++y_inner_outer_x_inner_outer_fused) {
      for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
        ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
      }
      for (int32_t k_outer = 0; k_outer < 128; ++k_outer) {
        for (int32_t x_c = 0; x_c < 8; ++x_c) {
          ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_14_let)[((((ax1_outer_ax0_outer_fused * 2048) + (y_inner_outer_x_inner_outer_fused * 1024)) + (k_outer * 8)) + x_c)]));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 8; ++x_inner_inner) {
        ((float*)compute_let)[((y_inner_outer_x_inner_outer_fused * 8) + x_inner_inner)] = ((float*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 2; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
        int32_t cse_var_2 = (ax1_inner_outer * 8);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 16) + cse_var_2) + ax1_inner_inner);
        float v_ = ((float*)compute_let)[(cse_var_2 + ax1_inner_inner)] + ((float*)fused_nn_contrib_dense_pack_constant_8_let)[cse_var_1];
        T_relu[cse_var_1] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_7(float* p0, float* T_relu, uint8_t* global_const_workspace_58_var, uint8_t* global_workspace_59_var) {
  void* fused_nn_contrib_dense_pack_constant_10_let = (&(global_const_workspace_58_var[1833408]));
  void* fused_constant_16_let = (&(global_const_workspace_58_var[1719296]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 4; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_59_var[1792]));
    void* compute_global_let = (&(global_workspace_59_var[1856]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 2; ++y_inner_outer_x_inner_outer_fused) {
      for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
        ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
      }
      for (int32_t k_outer = 0; k_outer < 128; ++k_outer) {
        for (int32_t x_c = 0; x_c < 8; ++x_c) {
          ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_16_let)[((((ax1_outer_ax0_outer_fused * 2048) + (y_inner_outer_x_inner_outer_fused * 1024)) + (k_outer * 8)) + x_c)]));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 8; ++x_inner_inner) {
        ((float*)compute_let)[((y_inner_outer_x_inner_outer_fused * 8) + x_inner_inner)] = ((float*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 2; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
        int32_t cse_var_2 = (ax1_inner_outer * 8);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 16) + cse_var_2) + ax1_inner_inner);
        float v_ = ((float*)compute_let)[(cse_var_2 + ax1_inner_inner)] + ((float*)fused_nn_contrib_dense_pack_constant_10_let)[cse_var_1];
        T_relu[cse_var_1] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_8(float* p0, float* T_relu, uint8_t* global_const_workspace_60_var, uint8_t* global_workspace_61_var) {
  void* fused_nn_contrib_dense_pack_constant_11_let = (&(global_const_workspace_60_var[1835040]));
  void* fused_constant_17_let = (&(global_const_workspace_60_var[1834176]));
  void* compute_global_let = (&(global_workspace_61_var[1792]));
  ((float*)compute_global_let)[0] = 0.000000e+00f;
  for (int32_t k_outer = 0; k_outer < 64; ++k_outer) {
    ((float*)compute_global_let)[0] = (((float*)compute_global_let)[0] + (p0[k_outer] * ((float*)fused_constant_17_let)[k_outer]));
  }
  float v_ = ((float*)compute_global_let)[0] + ((float*)fused_nn_contrib_dense_pack_constant_11_let)[0];
  T_relu[0] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_sigmoid_multiply(float* p0, float* T_multiply, uint8_t* global_const_workspace_56_var, uint8_t* global_workspace_57_var) {
  void* fused_nn_contrib_dense_pack_constant_9_let = (&(global_const_workspace_56_var[1835024]));
  void* fused_constant_15_let = (&(global_const_workspace_56_var[1834432]));
  void* compute_global_let = (&(global_workspace_57_var[1792]));
  ((float*)compute_global_let)[0] = 0.000000e+00f;
  for (int32_t k_outer = 0; k_outer < 64; ++k_outer) {
    ((float*)compute_global_let)[0] = (((float*)compute_global_let)[0] + (p0[k_outer] * ((float*)fused_constant_15_let)[k_outer]));
  }
  T_multiply[0] = ((1.000000e+00f / (1.000000e+00f + expf((0.000000e+00f - (((float*)compute_global_let)[0] + ((float*)fused_nn_contrib_dense_pack_constant_9_let)[0]))))) * 1.000000e+02f);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_global_avg_pool2d(float* p0, float* adaptive_pool_avg, uint8_t* global_const_workspace_20_var, uint8_t* global_workspace_21_var) {
  void* adaptive_pool_sum_let = (&(global_workspace_21_var[1116160]));
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 32; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
      ((float*)adaptive_pool_sum_let)[((ax0_ax1_fused_ax2_fused * 4) + ax4)] = 0.000000e+00f;
      for (int32_t rv0 = 0; rv0 < 32; ++rv0) {
        for (int32_t rv1 = 0; rv1 < 32; ++rv1) {
          int32_t cse_var_1 = ((ax0_ax1_fused_ax2_fused * 4) + ax4);
          ((float*)adaptive_pool_sum_let)[cse_var_1] = (((float*)adaptive_pool_sum_let)[cse_var_1] + p0[((((ax0_ax1_fused_ax2_fused * 4096) + (rv0 * 128)) + (rv1 * 4)) + ax4)]);
        }
      }
    }
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 32; ++ax0_ax1_fused) {
    for (int32_t ax4_1 = 0; ax4_1 < 4; ++ax4_1) {
      int32_t cse_var_2 = ((ax0_ax1_fused * 4) + ax4_1);
      adaptive_pool_avg[cse_var_2] = (((float*)adaptive_pool_sum_let)[cse_var_2] * 9.765625e-04f);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_max_pool2d(float* p0, float* pool_max, uint8_t* global_const_workspace_8_var, uint8_t* global_workspace_9_var) {
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 512; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3 = 0; ax3 < 64; ++ax3) {
      for (int32_t ax4_init = 0; ax4_init < 4; ++ax4_init) {
        pool_max[(((ax0_ax1_fused_ax2_fused * 256) + (ax3 * 4)) + ax4_init)] = -3.402823e+38f;
      }
      for (int32_t rv0_rv1_fused = 0; rv0_rv1_fused < 4; ++rv0_rv1_fused) {
        for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
          int32_t cse_var_1 = (((ax0_ax1_fused_ax2_fused * 256) + (ax3 * 4)) + ax4);
          float v_ = pool_max[cse_var_1];
          float v__1 = p0[(((((ax0_ax1_fused_ax2_fused * 1024) + ((rv0_rv1_fused >> 1) * 512)) + (ax3 * 8)) + ((rv0_rv1_fused & 1) * 4)) + ax4)];
          pool_max[cse_var_1] = ((v_) > (v__1) ? (v_) : (v__1));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_max_pool2d_1(float* p0, float* pool_max, uint8_t* global_const_workspace_14_var, uint8_t* global_workspace_15_var) {
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 512; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3 = 0; ax3 < 32; ++ax3) {
      for (int32_t ax4_init = 0; ax4_init < 4; ++ax4_init) {
        pool_max[(((ax0_ax1_fused_ax2_fused * 128) + (ax3 * 4)) + ax4_init)] = -3.402823e+38f;
      }
      for (int32_t rv0_rv1_fused = 0; rv0_rv1_fused < 4; ++rv0_rv1_fused) {
        for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
          int32_t cse_var_1 = (((ax0_ax1_fused_ax2_fused * 128) + (ax3 * 4)) + ax4);
          float v_ = pool_max[cse_var_1];
          float v__1 = p0[(((((ax0_ax1_fused_ax2_fused * 512) + ((rv0_rv1_fused >> 1) * 256)) + (ax3 * 8)) + ((rv0_rv1_fused & 1) * 4)) + ax4)];
          pool_max[cse_var_1] = ((v_) > (v__1) ? (v_) : (v__1));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_softmax(float* p0, float* T_softmax_norm, uint8_t* global_const_workspace_46_var, uint8_t* global_workspace_47_var) {
  void* T_softmax_maxelem_let = (&(global_workspace_47_var[2176]));
  void* T_softmax_exp_let = (&(global_workspace_47_var[2112]));
  void* T_softmax_expsum_let = (&(global_workspace_47_var[2176]));
  ((float*)T_softmax_maxelem_let)[0] = -3.402823e+38f;
  for (int32_t k = 0; k < 16; ++k) {
    float v_ = ((float*)T_softmax_maxelem_let)[0];
    float v__1 = p0[k];
    ((float*)T_softmax_maxelem_let)[0] = ((v_) > (v__1) ? (v_) : (v__1));
  }
  for (int32_t i1 = 0; i1 < 16; ++i1) {
    ((float*)T_softmax_exp_let)[i1] = expf((p0[i1] - ((float*)T_softmax_maxelem_let)[0]));
  }
  ((float*)T_softmax_expsum_let)[0] = 0.000000e+00f;
  for (int32_t k_1 = 0; k_1 < 16; ++k_1) {
    ((float*)T_softmax_expsum_let)[0] = (((float*)T_softmax_expsum_let)[0] + ((float*)T_softmax_exp_let)[k_1]);
  }
  for (int32_t i1_1 = 0; i1_1 < 16; ++i1_1) {
    T_softmax_norm[i1_1] = (((float*)T_softmax_exp_let)[i1_1] / ((float*)T_softmax_expsum_let)[0]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_softmax_1(float* p0, float* T_softmax_norm, uint8_t* global_const_workspace_52_var, uint8_t* global_workspace_53_var) {
  void* T_softmax_maxelem_let = (&(global_workspace_53_var[1568]));
  void* T_softmax_exp_let = (&(global_workspace_53_var[1552]));
  void* T_softmax_expsum_let = (&(global_workspace_53_var[1568]));
  ((float*)T_softmax_maxelem_let)[0] = -3.402823e+38f;
  for (int32_t k = 0; k < 4; ++k) {
    float v_ = ((float*)T_softmax_maxelem_let)[0];
    float v__1 = p0[k];
    ((float*)T_softmax_maxelem_let)[0] = ((v_) > (v__1) ? (v_) : (v__1));
  }
  for (int32_t i1 = 0; i1 < 4; ++i1) {
    ((float*)T_softmax_exp_let)[i1] = expf((p0[i1] - ((float*)T_softmax_maxelem_let)[0]));
  }
  ((float*)T_softmax_expsum_let)[0] = 0.000000e+00f;
  for (int32_t k_1 = 0; k_1 < 4; ++k_1) {
    ((float*)T_softmax_expsum_let)[0] = (((float*)T_softmax_expsum_let)[0] + ((float*)T_softmax_exp_let)[k_1]);
  }
  for (int32_t i1_1 = 0; i1_1 < 4; ++i1_1) {
    T_softmax_norm[i1_1] = (((float*)T_softmax_exp_let)[i1_1] / ((float*)T_softmax_expsum_let)[0]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_squeeze_layout_transform_concatenate(float* p0, float* p1, float* concatenate_ext, uint8_t* global_const_workspace_36_var, uint8_t* global_workspace_37_var) {
  void* T_squeeze_let = (&(global_workspace_37_var[53760]));
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 128; ++ax0_ax1_fused_ax2_fused) {
    ((float*)T_squeeze_let)[ax0_ax1_fused_ax2_fused] = p0[ax0_ax1_fused_ax2_fused];
  }
  for (int32_t j = 0; j < 128; ++j) {
    concatenate_ext[j] = ((float*)T_squeeze_let)[j];
  }
  for (int32_t j_1 = 0; j_1 < 128; ++j_1) {
    concatenate_ext[(j_1 + 128)] = p1[j_1];
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_subtract_multiply_mean(float* p0, float* p1, float* T_divide, uint8_t* global_const_workspace_24_var, uint8_t* global_workspace_25_var) {
  void* T_multiply_red_let = (&(global_workspace_25_var[54112]));
  ((float*)T_multiply_red_let)[0] = 0.000000e+00f;
  for (int32_t k1 = 0; k1 < 102; ++k1) {
    ((float*)T_multiply_red_let)[0] = (((float*)T_multiply_red_let)[0] + ((p0[k1] - p1[0]) * (p0[k1] - p1[0])));
  }
  T_divide[0] = (((float*)T_multiply_red_let)[0] * 9.803922e-03f);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_layout_transform(float* p0, float* T_layout_trans, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var) {
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 128; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3 = 0; ax3 < 128; ++ax3) {
      int32_t cse_var_1 = ((ax0_ax1_fused_ax2_fused * 128) + ax3);
      T_layout_trans[cse_var_1] = p0[cse_var_1];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* aux_buffer_var, float* spec_buffer_var, float* output_buffer_var, float* output2_buffer_var, float* output3_buffer_var, float* output4_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var) {
  void* sid_28_let = (&(global_workspace_1_var[1536]));
  void* sid_17_let = (&(global_workspace_1_var[52736]));
  void* sid_23_let = (&(global_workspace_1_var[2048]));
  void* sid_20_let = (&(global_workspace_1_var[0]));
  void* sid_22_let = (&(global_workspace_1_var[1536]));
  void* sid_16_let = (&(global_workspace_1_var[0]));
  void* sid_2_let = (&(global_workspace_1_var[4327952]));
  void* sid_19_let = (&(global_workspace_1_var[1024]));
  void* sid_13_let = (&(global_workspace_1_var[54096]));
  void* sid_21_let = (&(global_workspace_1_var[1024]));
  void* sid_6_let = (&(global_workspace_1_var[1115136]));
  void* sid_15_let = (&(global_workspace_1_var[53248]));
  void* sid_7_let = (&(global_workspace_1_var[1115136]));
  void* sid_5_let = (&(global_workspace_1_var[4260352]));
  void* sid_8_let = (&(global_workspace_1_var[2163712]));
  void* sid_9_let = (&(global_workspace_1_var[591872]));
  void* sid_18_let = (&(global_workspace_1_var[53248]));
  void* sid_3_let = (&(global_workspace_1_var[2163200]));
  void* sid_12_let = (&(global_workspace_1_var[54080]));
  void* sid_26_let = (&(global_workspace_1_var[1536]));
  void* sid_10_let = (&(global_workspace_1_var[591872]));
  void* sid_11_let = (&(global_workspace_1_var[52224]));
  void* sid_4_let = (&(global_workspace_1_var[2163200]));
  void* sid_25_let = (&(global_workspace_1_var[1536]));
  void* sid_14_let = (&(global_workspace_1_var[53664]));
  void* sid_30_let = (&(global_workspace_1_var[1536]));
  if (tvmgen_default_fused_transpose_layout_transform(spec_buffer_var, sid_2_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu(sid_2_let, sid_3_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_1(sid_3_let, sid_4_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_max_pool2d(sid_4_let, sid_5_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2(sid_5_let, sid_6_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_3(sid_6_let, sid_7_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_max_pool2d_1(sid_7_let, sid_8_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_4(sid_8_let, sid_9_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_5(sid_9_let, sid_10_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_global_avg_pool2d(sid_10_let, sid_11_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_mean(aux_buffer_var, sid_12_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_subtract_multiply_mean(aux_buffer_var, sid_12_let, sid_13_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_add_rsqrt_multiply(sid_13_let, sid_14_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_negative_multiply_add_divide_add(sid_12_let, sid_14_let, sid_14_let, aux_buffer_var, sid_15_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_multiply_layout_transform(sid_14_let, sid_16_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu(sid_15_let, sid_16_let, sid_17_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_1(sid_17_let, sid_18_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_squeeze_layout_transform_concatenate(sid_11_let, sid_18_let, sid_19_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_2(sid_19_let, sid_20_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_3(sid_20_let, sid_21_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_4(sid_21_let, sid_22_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add(sid_22_let, sid_23_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_softmax(sid_23_let, output_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_5(sid_21_let, sid_25_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_1(sid_25_let, sid_26_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_softmax_1(sid_26_let, output2_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_6(sid_21_let, sid_28_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_sigmoid_multiply(sid_28_let, output3_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_7(sid_21_let, sid_30_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_8(sid_30_let, output4_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  return 0;
}

