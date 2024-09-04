from __future__ import print_function
import sys
import os
import unittest
import logging
import multiprocessing
from subprocess import check_output, CalledProcessError

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase
import torch
from torch.testing._internal.common_utils import TemporaryFileName
import torch_mlu  # pylint: disable=W0611

logging.basicConfig(level=logging.DEBUG)


class TestLazyInit(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_lazy_init(self):
        """Validate that no MLU calls are made during `import torch` call"""
        test_script = "import os; import torch; \
                        import torch_mlu; os.environ['MLU_VISIBLE_DEVICES']='32'; \
                        print(torch.mlu.device_count())"
        rc = check_output([sys.executable, "-c", test_script]).decode("ascii").strip()
        self.assertEqual(rc, "0")

    # @unittest.skip("not test")
    @testinfo()
    def test_mlu_init(self):
        torch.manual_seed(0)
        torch.seed()
        torch.mlu.seed_all()
        torch.mlu.manual_seed_all(0)
        initialized = torch.mlu.is_initialized()
        self.assertFalse(initialized)
        torch.mlu.init()
        initialized = torch.mlu.is_initialized()
        self.assertTrue(initialized)

    # @unittest.skip("not test")
    @testinfo()
    def test_lazy_init_exception(self):
        stderr = TestCase.runWithPytorchAPIUsageStderr(
            """\
import torch
import torch_mlu
from torch.multiprocessing import Process
def run(rank):
    torch.mlu.set_device(rank)
if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        # it would work fine without the line below
        x = torch.rand(20, 2).to("mlu")
        p = Process(target=run, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
"""
        )
        self.assertRegex(stderr, "Cannot re-initialize MLU in forked subprocess.")

    @testinfo()
    def test_no_mlus(self):
        """Validate that no MLU calls are made during `import torch` call"""
        test_script = "import os; import torch; \
                        import torch_mlu; os.environ['MLU_VISIBLE_DEVICES']=''; \
                        torch_mlu._MLUC._mlu_init()"
        try:
            check_output([sys.executable, "-c", test_script]).decode("ascii").strip()
        except CalledProcessError:
            return
        self.fail(msg="Did not raise when expected to")


if __name__ == "__main__":
    unittest.main()
