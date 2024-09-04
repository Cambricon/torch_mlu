#pragma once

#include "aten/operators/bang/internal/common_util.h"

namespace torch_mlu {
namespace ops {

void dump(
    void* input,
    int32_t size,
    cnrtDim3_t dim,
    cnrtFunctionType_t ktype,
    cnrtQueue_t queue,
    cnrtDataType_t cnrt_type);

void amp_update_scale_internal(
    void* new_scale,
    void* growth_tracker_output,
    void* growth_tracker,
    void* current_scale,
    void* found_inf,
    const float growth_factor,
    const float backoff_factor,
    const int64_t growth_interval,
    cnrtDim3_t& dim,
    cnrtFunctionType_t ktype,
    cnrtQueue_t queue);

bool amp_unscale_internal(
    void* scaled_grad,
    void* found_inf,
    void* inv_scale,
    void* found_inf_out,
    int32_t elem_num,
    cnrtFunctionType_t k_type,
    cnrtDim3_t k_dim,
    cnrtQueue_t queue,
    cnrtDataType_t cnrt_type);

bool amp_unscale_internal(
    const void* const* scaled_grads,
    const uint64_t* const scaled_grads_numel,
    void* found_inf,
    void* inv_scale,
    void* found_inf_out,
    int32_t tensors_num,
    cnrtFunctionType_t k_type,
    cnrtDim3_t k_dim,
    cnrtQueue_t queue,
    cnrtDataType_t cnrt_type);

void bang_fused_l2_norm_internal(
    AddressList tensors,
    SizeList sizes,
    float* output_buffer_ptr,
    float* output_buffer_per_tensor_ptr,
    int tensor_num,
    bool per_tensor,
    int32_t* overflow,
    cnrtDim3_t k_dim,
    cnrtFunctionType_t k_type,
    cnrtQueue_t queue,
    cnrtDataType_V2_t cnrt_type,
    bool amp_opt);

void bang_fused_l2_norm_clean_internal(
    float* output_ptr,
    float* output_per_tensor_ptr,
    float* output_buffer_ptr,
    float* output_buffer_per_tensor_ptr,
    bool per_tensor,
    int tensor_num,
    int* overflow,
    cnrtDim3_t k_dim,
    cnrtFunctionType_t k_type,
    cnrtQueue_t queue,
    bool amp_opt);

void bang_fused_adam_internal(
    AddressList grad,
    AddressList exp_avg,
    AddressList exp_avg_sq,
    AddressList param,
    SizeList sizes,
    int tensor_num,
    float beta1,
    float beta2,
    float beta1_correction,
    float beta2_correction,
    float epsilon,
    float epsilon_correction,
    float learning_rate,
    float learning_rate_correction,
    int adam_mode,
    float decay,
    float decay_correction,
    cnrtDim3_t k_dim,
    cnrtFunctionType_t k_type,
    cnrtQueue_t queue,
    cnrtDataType_V2_t cnrt_type);

void bang_fused_sgd_internal(
    AddressList g,
    AddressList i,
    AddressList m,
    AddressList o,
    SizeList s,
    int tensor_num,
    int32_t* overflow,
    float weight_decay,
    float momentum,
    float dampening,
    float learning_rate,
    bool nesterov,
    bool first_run,
    bool wd_after_momentum,
    float scale,
    cnrtDim3_t k_dim,
    cnrtFunctionType_t k_type,
    cnrtQueue_t queue,
    cnrtDataType_t in_type,
    cnrtDataType_t out_type,
    int N);

void bang_fused_lamb_internal(
    AddressList grad,
    AddressList param,
    AddressList m,
    AddressList v,
    SizeList sizes,
    float* global_grad_norm,
    int tensor_num,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon_correction,
    float decay,
    float correction_rate,
    float max_grad_norm,
    int mode,
    int grad_averaging,
    void* global_param_norm,
    void* global_update_norm,
    int* overflow,
    bool use_adaptive_lr,
    cnrtDim3_t k_dim,
    cnrtFunctionType_t k_type,
    cnrtQueue_t queue,
    cnrtDataType_t cnrt_type);

void bang_fused_lamb_amp_internal(
    AddressList grad,
    AddressList param,
    AddressList m,
    AddressList v,
    AddressList half_param,
    SizeList sizes,
    int tensor_num,
    int* overflow,
    float beta1,
    float beta2,
    int grad_averaging,
    int* step,
    int bias_correction,
    float epsilon,
    int mode,
    float weight_decay,
    float* global_grad_norm,
    float* max_grad_norm,
    void* global_param_norm,
    void* global_update_norm,
    float* inv_scale,
    float* lr,
    bool use_adaptive_lr,
    cnrtDim3_t k_dim,
    cnrtFunctionType_t k_type,
    cnrtQueue_t queue,
    cnrtDataType_t grad_cnrt_type,
    cnrtDataType_t param_cnrt_type);

} // namespace ops
} // namespace torch_mlu
