# pylint: disable=W0223,R0201,C0413,C0411,C0301
from __future__ import print_function

import sys
import os
import logging
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)


class TestArgsortOp(TestCase):
    def assertIsOrdered(self, order, x, mxx, ixx, task):
        SIZE = x.size(1)
        if order == "descending":

            def check_order(a, b):
                # `a != a` because we put NaNs
                # at the end of ascending sorted lists,
                # and the beginning of descending ones.
                return ((a.isnan()) | (a >= b)).all().item()

        elif order == "ascending":

            def check_order(a, b):
                # see above
                return ((b.isnan()) | (a <= b)).all().item()

        else:
            raise Exception(
                'unknown order "{}", must be "ascending" or "descending"'.format(order)
            )

        are_ordered = True
        for k in range(1, SIZE):
            self.assertTrue(
                check_order(mxx[:, k - 1], mxx[:, k]),
                "torch.sort ({}) values unordered for {}".format(order, task),
            )
        seen = set()
        indicesCorrect = True
        size0 = x.size(0)
        size = x.size(x.dim() - 1)
        x = x.tolist()
        mxx = mxx.tolist()
        ixx = ixx.tolist()
        for k in range(size0):
            seen.clear()
            for j in range(size):
                self.assertEqual(
                    x[k][ixx[k][j]],
                    mxx[k][j],
                    msg="torch.sort ({}) indices wrong for {}".format(order, task),
                )
                seen.add(ixx[k][j])
            self.assertEqual(len(seen), size)

    # @unittest.skip("not test")
    @testinfo()
    def test_sort(self):
        device = "mlu"
        for SIZE in (4, 2049):
            x = torch.rand(4, SIZE, device=device)
            res1val, res1ind = torch.sort(x)

            # Test use of result tensor
            res2val = torch.tensor((), device=device)
            res2ind = torch.tensor((), device=device, dtype=torch.long)
            torch.sort(x, out=(res2val, res2ind))
            self.assertEqual(res1val, res2val, atol=0, rtol=0)
            self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
            self.assertEqual(torch.argsort(x), res1ind)
            self.assertEqual(x.argsort(), res1ind)

            # Test sorting of random numbers
            self.assertIsOrdered("ascending", x, res2val, res2ind, "random")

            # Test simple sort
            self.assertEqual(
                torch.sort(torch.tensor((50, 40, 30, 20, 10), device=device))[0],
                torch.tensor((10, 20, 30, 40, 50), device=device),
                atol=0,
                rtol=0,
            )

            # Test that we still have proper sorting with duplicate keys
            x = torch.floor(torch.rand(4, SIZE, device=device) * 10)
            torch.sort(x, out=(res2val, res2ind))
            self.assertIsOrdered(
                "ascending", x, res2val, res2ind, "random with duplicate keys"
            )

            # DESCENDING SORT
            x = torch.rand(4, SIZE, device=device)
            res1val, res1ind = torch.sort(x, x.dim() - 1, True)

            # Test use of result tensor
            res2val = torch.tensor((), device=device)
            res2ind = torch.tensor((), device=device, dtype=torch.long)
            torch.sort(x, x.dim() - 1, True, out=(res2val, res2ind))
            self.assertEqual(res1val, res2val, atol=0, rtol=0)
            self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
            self.assertEqual(torch.argsort(x, x.dim() - 1, True), res1ind)
            self.assertEqual(x.argsort(x.dim() - 1, True), res1ind)

            # Test sorting of random numbers
            self.assertIsOrdered("descending", x, res2val, res2ind, "random")

            # Test simple sort task
            self.assertEqual(
                torch.sort(torch.tensor((10, 20, 30, 40, 50), device=device), 0, True)[
                    0
                ],
                torch.tensor((50, 40, 30, 20, 10), device=device),
                atol=0,
                rtol=0,
            )

            # Test that we still have proper sorting with duplicate keys
            self.assertIsOrdered(
                "descending", x, res2val, res2ind, "random with duplicate keys"
            )

            # TODO(sifengyang): Currently MLU can't handle sorting with NaNs.
            # this problem will be solved in CNNLCORE-7394.
            # Test sorting with NaNs
            # x = torch.rand(4, SIZE, device=device)
            # x[1][2] = float('NaN')
            # x[3][0] = float('NaN')
            # torch.sort(x, out=(res2val, res2ind))
            # self.assertIsOrdered('ascending', x, res2val, res2ind,
            #                      'random with NaNs')
            # torch.sort(x, out=(res2val, res2ind), descending=True)
            # self.assertIsOrdered('descending', x, res2val, res2ind,
            #                      'random with NaNs')


if __name__ == "__main__":
    run_tests()
