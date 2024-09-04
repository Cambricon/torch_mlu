/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/pytorch/pytorch/graphs/contributors Redistribution and use in
source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <ATen/native/TypeProperties.h>
#include <algorithm>

namespace torch_mlu {
namespace ops {

const float EPS = 1e-8;

struct Point {
  float x, y;
  Point() {}
  Point(double _x, double _y) {
    x = _x, y = _y;
  }

  void set(float _x, float _y) {
    x = _x;
    y = _y;
  }

  Point operator+(const Point& b) const {
    return Point(x + b.x, y + b.y);
  }

  Point operator-(const Point& b) const {
    return Point(x - b.x, y - b.y);
  }
};

class UtilsFunctions {
 public:
  static float cross(const Point& a, const Point& b) {
    return a.x * b.y - a.y * b.x;
  }
  static float cross(const Point& p1, const Point& p2, const Point& p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
  }
  static int check_rect_cross(
      const Point& p1,
      const Point& p2,
      const Point& q1,
      const Point& q2) {
    int ret = std::min(p1.x, p2.x) <= std::max(q1.x, q2.x) &&
        std::min(q1.x, q2.x) <= std::max(p1.x, p2.x) &&
        std::min(p1.y, p2.y) <= std::max(q1.y, q2.y) &&
        std::min(q1.y, q2.y) <= std::max(p1.y, p2.y);
    return ret;
  }
  static int check_in_box2d(const float* box, const Point& p) {
    // params: (7) [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-7;

    float center_x = box[0], center_y = box[1];
    float angle_cos = cos(-box[6]),
          angle_sin =
              sin(-box[6]); // rotate the point in the opposite direction of box
    float rot_x =
        (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (
        fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN);
  }
  static int intersection(
      const Point& p1,
      const Point& p0,
      const Point& q1,
      const Point& q0,
      Point& ans) {
    // fast exclusion
    if (check_rect_cross(p0, p1, q0, q1) == 0)
      return 0;

    // check cross standing
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0))
      return 0;

    // calculate intersection of two lines
    float s5 = cross(q1, p1, p0);
    if (fabs(s5 - s1) > EPS) {
      ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
      ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    } else {
      float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
      float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
      float D = a0 * b1 - a1 * b0;

      ans.x = (b0 * c1 - b1 * c0) / D;
      ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
  }
  static void rotate_around_center(
      const Point& center,
      const float angle_cos,
      const float angle_sin,
      Point& p) {
    float new_x = (p.x - center.x) * angle_cos +
        (p.y - center.y) * (-angle_sin) + center.x;
    float new_y =
        (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
  }

  static int point_cmp(const Point& a, const Point& b, const Point& center) {
    return atan2(a.y - center.y, a.x - center.x) >
        atan2(b.y - center.y, b.x - center.x);
  }

  static float box_overlap(const float* box_a, const float* box_b) {
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]

    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2,
          a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

    Point center_a(box_a[0], box_a[1]);
    Point center_b(box_b[0], box_b[1]);

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1);
    box_a_corners[1].set(a_x2, a_y1);
    box_a_corners[2].set(a_x2, a_y2);
    box_a_corners[3].set(a_x1, a_y2);

    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x2, b_y2);
    box_b_corners[3].set(b_x1, b_y2);

    // get oriented corners
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
      rotate_around_center(
          center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
      rotate_around_center(
          center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        flag = intersection(
            box_a_corners[i + 1],
            box_a_corners[i],
            box_b_corners[j + 1],
            box_b_corners[j],
            cross_points[cnt]);
        if (flag) {
          poly_center = poly_center + cross_points[cnt];
          cnt++;
        }
      }
    }

    // check corners
    for (int k = 0; k < 4; k++) {
      if (check_in_box2d(box_a, box_b_corners[k])) {
        poly_center = poly_center + box_b_corners[k];
        cross_points[cnt] = box_b_corners[k];
        cnt++;
      }
      if (check_in_box2d(box_b, box_a_corners[k])) {
        poly_center = poly_center + box_a_corners[k];
        cross_points[cnt] = box_a_corners[k];
        cnt++;
      }
    }
    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    Point temp;
    for (int j = 0; j < cnt - 1; j++) {
      for (int i = 0; i < cnt - j - 1; i++) {
        if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
          temp = cross_points[i];
          cross_points[i] = cross_points[i + 1];
          cross_points[i + 1] = temp;
        }
      }
    }

    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
      area += cross(
          cross_points[k] - cross_points[0],
          cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
  }

  static float iou_bev(const float* box_a, const float* box_b) {
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
  }
};

} // namespace ops
} // namespace torch_mlu
