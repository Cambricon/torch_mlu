#include "aten/operators/cpu/internal/nms3D_cpu_util.h"
#include "aten/operators/cpu/cpu_kernel.h"

namespace torch_mlu {
namespace ops {

void nms3D_cpu_kernel(
    float* output_data,
    int& output_box_num,
    float* input_data,
    int input_box_num,
    float thresh_iou) {
  // int output_box_num = 0;
  float* x1 = new float[input_box_num];
  float* y1 = new float[input_box_num];
  float* dx = new float[input_box_num];
  float* dy = new float[input_box_num];
  float* angle = new float[input_box_num];
  float* score = new float[input_box_num];
  float* box_a = new float[7];
  float* box_b = new float[7];
  memset(score, 1, input_box_num * sizeof(int));
  // x,y,z,dx,dy,dz,angle
  for (int i = 0; i < input_box_num; i++) {
    x1[i] = input_data[0 + i * 7];
    y1[i] = input_data[1 + i * 7];
    dx[i] = input_data[3 + i * 7];
    dy[i] = input_data[4 + i * 7];
    angle[i] = input_data[6 + i * 7];
  }

  for (int cur_box = 0; cur_box < input_box_num; cur_box++) {
    if (score[cur_box] == 0) {
      continue;
    }
    output_data[output_box_num] = cur_box;
    output_box_num++;
    // params box_a: [x, y, z, dx, dy, dz, heading]
    box_a[0] = x1[cur_box], box_a[1] = y1[cur_box];
    box_a[3] = dx[cur_box], box_a[4] = dy[cur_box];
    box_a[6] = angle[cur_box];

    for (int i = 0; i < input_box_num; i++) {
      box_b[0] = x1[i], box_b[1] = y1[i];
      box_b[3] = dx[i], box_b[4] = dy[i];
      box_b[6] = angle[i];
      // get IOU
      float iou = UtilsFunctions::iou_bev(box_a, box_b);
      if (iou > thresh_iou) {
        score[i] = 0;
      }
    }
  }
  // VLOG(4) << "ouput_boxes_num:" << output_box_num;
  delete[] score;
  delete[] x1;
  delete[] y1;
  delete[] dx;
  delete[] dy;
  delete[] angle;
  delete[] box_a;
  delete[] box_b;
}

at::Tensor nms3D_cpu(const at::Tensor& dets, double iou_threshold) {
  // auto dets_contiguous = cnnl_contiguous(dets);
  auto dets_contiguous = dets.contiguous();
  TORCH_CHECK(
      dets_contiguous.dim() == 2,
      "boxes should be a 2D tensor, got ",
      dets_contiguous.dim(),
      "D");
  TORCH_CHECK(
      dets_contiguous.size(1) == 7, "boxes sizes should be (batch_size, 7).");
  TORCH_CHECK(
      dets.scalar_type() == at::ScalarType::Float,
      "dets dtype should be float");
  int input_box_num = (int)dets.size(0);
  auto dtype = dets.scalar_type();
  auto output =
      at::empty({input_box_num}, dets_contiguous.options().dtype(dtype));
  // auto output_cpu = output.cpu();
  // auto dets_cpu = dets_contiguous.cpu();
  float* output_ptr = (float*)output.data_ptr();
  float* dets_ptr = (float*)dets_contiguous.data_ptr();
  int output_box_num = 0;
  nms3D_cpu_kernel(
      output_ptr, output_box_num, dets_ptr, input_box_num, iou_threshold);
  // output.copy_(output_cpu);
  return output.slice(0, 0, output_box_num);
}

} // namespace ops

TORCH_LIBRARY_FRAGMENT(torch_mlu, m) {
  m.def("nms3D_cpu(Tensor dets, float iou_threshold) -> Tensor", {});
}

TORCH_LIBRARY_IMPL(torch_mlu, CPU, m) {
  m.impl("nms3D_cpu", at::kCPU, TORCH_FN(ops::nms3D_cpu));
}

} // namespace torch_mlu
