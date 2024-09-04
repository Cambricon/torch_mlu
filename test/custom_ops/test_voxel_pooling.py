from __future__ import print_function

import sys
import os
import numpy as np
import unittest
import torch
import torch_mlu  # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411


class TestMluVoxelPoolingOp(TestCase):
    def _voxel_pooling(self, geom_xyz, input_features, voxel_num):
        """Forward function for `voxel pooling.
        Args:
            geom_xyz (Tensor): xyz coord for each voxel with the shape
                of [B, N, 3].
            input_features (Tensor): feature for each voxel with the
                shape of [B, N, C].
            voxel_num (Tensor): Number of voxels for each dim with the
                shape of [3].
        Returns:
            Tensor: (B, C, H, W) bev feature map.
        """
        assert geom_xyz.is_contiguous()
        assert input_features.is_contiguous()

        geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1])
        input_features = input_features.reshape(
            (geom_xyz.shape[0], -1, input_features.shape[-1])
        )
        assert geom_xyz.shape[1] == input_features.shape[1]
        batch_size = input_features.shape[0]
        num_points = input_features.shape[1]
        num_channels = input_features.shape[2]
        output_features = input_features.new_zeros(
            batch_size, voxel_num[1], voxel_num[0], num_channels
        )
        # Save the position of bev_feature_map for each input point.
        pos_memo = geom_xyz.new_ones(batch_size, num_points, 3) * -1
        torch.ops.torch_mlu.voxel_pooling(
            batch_size,
            num_points,
            num_channels,
            voxel_num[0],
            voxel_num[1],
            voxel_num[2],
            geom_xyz,
            input_features,
            output_features,
            pos_memo,
        )
        return output_features.permute(0, 3, 1, 2)

    # @unittest.skip("not test")
    @testinfo()
    def test_voxel_pooling(self):
        np.random.seed(0)
        torch.manual_seed(0)
        geom_shape = [
            (2, 6, 10, 10, 10, 3),
        ]
        features_shape = [
            (2, 6, 10, 10, 10, 80),
        ]
        for i, shape in enumerate(geom_shape):
            geom_xyz = torch.rand(shape, dtype=torch.float) * 160 - 80
            geom_xyz[..., 2] /= 100
            geom_xyz = geom_xyz.reshape(2, -1, 3)
            features = torch.rand(features_shape[i], dtype=torch.float) - 0.5
            gt_features = features.reshape(2, -1, 80)
            gt_bev_featuremap = features.new_zeros(2, 128, 128, 80)
            for i in range(2):
                for j in range(geom_xyz.shape[1]):
                    x = geom_xyz[i, j, 0].int()
                    y = geom_xyz[i, j, 1].int()
                    z = geom_xyz[i, j, 2].int()
                    if x < 0 or x >= 128 or y < 0 or y >= 128 or z < 0 or z >= 1:
                        continue
                    gt_bev_featuremap[i, y, x, :] += gt_features[i, j, :]
            gt_bev_featuremap = gt_bev_featuremap.permute(0, 3, 1, 2)
            # geom_xyz must be int tensor, and features must be float tensor.
            bev_featuremap = self._voxel_pooling(
                geom_xyz.mlu().int(),
                features.mlu(),
                torch.tensor([128, 128, 1], dtype=torch.int, device="mlu"),
            )
            self.assertTrue(
                torch.allclose(gt_bev_featuremap, bev_featuremap.cpu(), atol=3e-3)
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_voxel_pooling_exceptions(self):
        geom_shape = (2, 16, 3)
        features_shape = (2, 16, 8)
        geom_xyz = torch.rand(geom_shape, dtype=torch.float)
        features = torch.rand(features_shape, dtype=torch.float)
        voxel_num = torch.tensor([128, 128, 1], dtype=torch.int, device="mlu")

        ref_msg = r"^geom_xyz and pos_memo must be int tensor."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            featuremap = self._voxel_pooling(geom_xyz.mlu(), features.mlu(), voxel_num)

        ref_msg = r"^input_features and output_features must be float tensor."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            featuremap = self._voxel_pooling(
                geom_xyz.mlu().int(), features.mlu().int(), voxel_num
            )


if __name__ == "__main__":
    unittest.main()
