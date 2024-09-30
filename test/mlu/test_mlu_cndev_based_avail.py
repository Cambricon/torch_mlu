import multiprocessing
import os
import sys
import unittest
from unittest.mock import patch

import torch
import torch_mlu


# NOTE: Each of the tests in this module need to be run in a brand new process to ensure MLU is uninitialized
# prior to test initiation.
with patch.dict(os.environ, {"PYTORCH_CNDEV_BASED_MLU_CHECK": "1"}):
    # Before executing the desired tests, we need to disable MLU initialization and fork_handler additions that would
    # otherwise be triggered by the `torch.testing._internal.common_utils` module import
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(cur_dir + "/../")
    from common_utils import TestCase, run_tests  # pylint: disable=C0413
    from torch.testing._internal.common_utils import (
        instantiate_parametrized_tests,
        parametrize,
    )

    # NOTE: Because `remove_device_and_dtype_suffixes` initializes MLU context (triggered via the import of
    # `torch.testing._internal.common_device_type` which imports `torch.testing._internal.common_cuda`) we need
    # to bypass that method here which should be irrelevant to the parameterized tests in this module.
    torch.testing._internal.common_utils.remove_device_and_dtype_suffixes = lambda x: x


class TestExtendedMLUIsAvail(TestCase):
    SUBPROCESS_REMINDER_MSG = (
        "\n REMINDER: Tests defined in test_mlu_cndev_based_avail.py must be run in a process "
        "where there MLU Driver API has not been initialized. Before further debugging, ensure you are either using "
        "run_test.py or have added --subprocess to run each test in a different subprocess."
    )

    def setUp(self):
        super().setUp()
        torch.mlu._cached_device_count = (
            None  # clear the lru_cache on this method before our test
        )

    @staticmethod
    def in_bad_fork_test() -> bool:
        _ = torch.mlu.is_available()
        return torch.mlu._is_in_bad_fork()

    # These tests validate the behavior and activation of the weaker, CNDEV-based, user-requested
    # `torch.mlu.is_available()` assessment. The CNDEV-based assessment should be attempted when
    # `PYTORCH_CNDEV_BASED_MLU_CHECK` is set to 1, reverting to the default MLU Runtime API check otherwise.
    # If the CNDEV-based assessment is attempted but fails, the MLU Runtime API check should be executed
    @parametrize("cndev_avail", [True, False])
    @parametrize("avoid_init", ["1", "0", None])
    def test_mlu_is_available(self, avoid_init, cndev_avail):
        patch_env = {"PYTORCH_CNDEV_BASED_MLU_CHECK": avoid_init} if avoid_init else {}
        with patch.dict(os.environ, **patch_env):
            if cndev_avail:
                _ = torch.mlu.is_available()
            else:
                with patch.object(torch.mlu, "_device_count_cndev", return_value=-1):
                    _ = torch.mlu.is_available()
            with multiprocessing.get_context("fork").Pool(1) as pool:
                in_bad_fork = pool.apply(TestExtendedMLUIsAvail.in_bad_fork_test)
            if os.getenv("PYTORCH_CNDEV_BASED_MLU_CHECK") == "1" and cndev_avail:
                self.assertFalse(
                    in_bad_fork, TestExtendedMLUIsAvail.SUBPROCESS_REMINDER_MSG
                )
            else:
                assert in_bad_fork


class TestVisibleDeviceParses(TestCase):
    def test_env_var_parsing(self):
        def _parse_visible_devices(val):
            from torch.mlu import _parse_visible_devices as _pvd

            with patch.dict(os.environ, {"CN_VISIBLE_DEVICES": val}, clear=True):
                return _pvd()

        # rest of the string is ignored
        self.assertEqual(_parse_visible_devices("1gpu2,2ampere"), [])
        # Negatives abort parsing
        self.assertEqual(_parse_visible_devices("0, 1, 2, -1, 3"), [0, 1, 2])
        # Double mention of ordinal returns empty set
        self.assertEqual(_parse_visible_devices("0, 1, 2, 1"), [])
        # Unary pluses and minuses
        self.assertEqual(_parse_visible_devices("2, +3, -0, 5"), [2])
        # Random string is used as empty set
        self.assertEqual(_parse_visible_devices("one,two,3,4"), [])
        # Random string is used as separator
        self.assertEqual(_parse_visible_devices("4,3,two,one"), [4, 3])
        # MLU ids are parsed
        self.assertEqual(
            _parse_visible_devices("33033011-2157-0000-0000-000000000000"),
            ["33033011-2157-0000-0000-000000000000"],
        )
        # Ordinals are not included in GPUid set
        self.assertEqual(
            _parse_visible_devices("33033011-2157-0000-0000-000000000000, 2"),
            ["33033011-2157-0000-0000-000000000000"],
        )

    def test_partial_uuid_resolver(self):
        from torch.mlu import _transform_uuid_to_ordinals

        uuids = [
            "49003004-2154-0000-0000-000000000000",
            "49003004-2154-0000-0000-000000000001",
            "47003004-2154-0000-0000-000000000000",
            "47003004-2154-0000-0000-000000000001",
            "53003004-2154-0000-0000-000000000000",
            "53003004-2154-0000-0000-000000000001",
            "59003004-2154-0000-0000-000000000000",
            "59003004-2154-0000-0000-000000000001",
        ]
        self.assertEqual(
            _transform_uuid_to_ordinals(
                ["49003004-2154-0000-0000-000000000001"], uuids
            ),
            [1],
        )
        self.assertEqual(
            _transform_uuid_to_ordinals(
                [
                    "47003004-2154-0000-0000-000000000000",
                    "49003004-2154-0000-0000-000000000001",
                ],
                uuids,
            ),
            [2, 1],
        )
        self.assertEqual(
            _transform_uuid_to_ordinals(
                "49003004-2154-0000-0000-000000000001,59003004-2154-0000-0000-000000000001,53003004-2154-0000-0000-000000000001".split(
                    ","
                ),
                uuids,
            ),
            [1, 7, 5],
        )
        # First invalid UUID aborts parsing
        self.assertEqual(
            _transform_uuid_to_ordinals(
                ["4564879-231", "59003004-2154-0000-0000-000000000001"], uuids
            ),
            [],
        )
        self.assertEqual(
            _transform_uuid_to_ordinals(
                [
                    "49003004-2154-0000-0000-000000000001",
                    "2131-1",
                    "47003004-2154-0000-0000-000000000001",
                ],
                uuids,
            ),
            [1],
        )
        # Duplicate UUIDs result in empty set
        self.assertEqual(
            _transform_uuid_to_ordinals(
                [
                    "49003004-2154-0000-0000-000000000000",
                    "49003004-2154-0000-0000-000000000001",
                    "49003004-2154-0000-0000-000000000000",
                ],
                uuids,
            ),
            [],
        )

    def test_ordinal_parse_visible_devices(self):
        def _device_count_cndev(val):
            from torch.mlu import _device_count_cndev as _dc

            with patch.dict(os.environ, {"CN_VISIBLE_DEVICES": val}, clear=True):
                return _dc()

        with patch.object(torch.mlu, "_raw_device_count_cndev", return_value=2):
            self.assertEqual(_device_count_cndev("1, 0"), 2)
            # Ordinal out of bounds aborts parsing
            self.assertEqual(_device_count_cndev("1, 5, 0"), 1)


instantiate_parametrized_tests(TestExtendedMLUIsAvail)

if __name__ == "__main__":
    run_tests()
