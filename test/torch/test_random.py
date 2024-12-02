import sys
import logging
import os
import torch
import torch_mlu
import torch.nn.functional as F

import unittest  # pylint: disable=C0411

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    freeze_rng_state,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class RNG(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_SetState(self):
        state = torch.mlu.get_rng_state()
        stateCloned = state.clone()
        before = torch.rand(1000).to("mlu").uniform_()

        self.assertEqual(state.ne(stateCloned).long().sum(), 0, atol=0, rtol=0)

        torch.mlu.set_rng_state(state)
        after = torch.rand(1000).to("mlu").uniform_()
        self.assertTensorsEqual(before.cpu(), after.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_GetState(self):
        torch.manual_seed(123)
        state1 = torch.mlu.get_rng_state()
        torch.rand(1000).to("mlu").uniform_()
        state2 = torch.mlu.get_rng_state()
        self.assertNotEqual(state1.cpu().sum(), state2.cpu().sum())

        torch.manual_seed(123)
        state3 = torch.mlu.get_rng_state()
        torch.rand(1000).to("mlu").uniform_()
        state4 = torch.mlu.get_rng_state()
        self.assertTensorsEqual(state1.cpu(), state3.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(state2.cpu(), state4.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ManualSeed(self):
        torch.manual_seed(1000)
        before = torch.rand(1000).to("mlu").uniform_()

        torch.manual_seed(1000)
        after = torch.rand(1000).to("mlu").uniform_()

        self.assertTensorsEqual(before.cpu(), after.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_randint(self):
        state = torch.mlu.get_rng_state()
        stateCloned = state.clone()
        before = torch.randint(10, (2, 2), device="mlu", dtype=torch.long)

        self.assertEqual(state.ne(stateCloned).long().sum(), 0, atol=0, rtol=0)

        torch.mlu.set_rng_state(state)
        after = torch.randint(10, (2, 2), device="mlu", dtype=torch.long)
        self.assertTensorsEqual(before.cpu(), after.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_InitialSeed(self):
        torch.mlu.seed()
        before = torch.mlu.initial_seed()
        torch.mlu.seed()
        after = torch.mlu.initial_seed()
        self.assertNotEqual(before, after)

    # @unittest.skip("not test")
    @testinfo()
    def test_mlu_default_generator(self):
        torch.mlu.init()
        # test default generators are equal
        self.assertEqual(torch.mlu.default_generators, torch.mlu.default_generators)

        # tests Generator API
        # manual_seed, seed, initial_seed, get_state, set_state
        g1 = torch.mlu.default_generators[0]
        g2 = torch.mlu.default_generators[0]
        g1.manual_seed(12345)
        g2.manual_seed(12345)
        self.assertEqual(g1.initial_seed(), g2.initial_seed())

        if torch.mlu.device_count() < 2:
            return
        g1 = torch.mlu.default_generators[0]
        g2 = torch.mlu.default_generators[1]
        g1.manual_seed(12345)
        g2.manual_seed(12345)
        g1.seed()
        g2.seed()
        self.assertNotEqual(g1.initial_seed(), g2.initial_seed())

        g1_default_state = torch.mlu.default_generators[0].get_state()
        g2 = torch.mlu.default_generators[1]
        g2.set_state(g1_default_state)
        g2_state = g2.get_state()
        # TODO(sifengyang): device of g1_default_state is mlu:0
        # device of g2_state is mlu:1. This is deprecated.
        self.assertTensorsEqual(
            g1_default_state.cpu(), g2_state.cpu(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_get_set_rng_state_all(self):
        states = torch.mlu.get_rng_state_all()
        before0 = torch.mlu.FloatTensor(100, device=0).normal_()
        before1 = torch.mlu.FloatTensor(100, device=1).normal_()
        torch.mlu.set_rng_state_all(states)
        after0 = torch.mlu.FloatTensor(100, device=0).normal_()
        after1 = torch.mlu.FloatTensor(100, device=1).normal_()
        self.assertEqual(before0, after0, atol=0, rtol=0)
        self.assertEqual(before1, after1, atol=0, rtol=0)

    # @unittest.skip("not test")
    @testinfo()
    def test_random_rng_state(self):
        device = "mlu"
        dtype = torch.float32
        # "hard" and "not hard" should propagate same gradient.
        logits_soft = torch.zeros(
            10, 10, dtype=dtype, device=device, requires_grad=True
        )
        logits_hard = torch.zeros(
            10, 10, dtype=dtype, device=device, requires_grad=True
        )

        seed = torch.mlu.random.get_rng_state()
        y_soft = F.gumbel_softmax(logits_soft, hard=False)
        torch.mlu.random.set_rng_state(seed)
        y_hard = F.gumbel_softmax(logits_hard, hard=True)

        y_soft.sum().backward()
        y_hard.sum().backward()

        # 2eps = 1x addition + 1x subtraction.
        tol = 2 * torch.finfo(dtype).eps
        self.assertEqual(logits_soft.grad, logits_hard.grad, atol=tol, rtol=0)

    # @unittest.skip("not test")
    @testinfo()
    def test_multinomial_deterministic(self):
        gen = torch.Generator(device="mlu")

        trials = 5
        seed = 0
        n_sample = 1

        for dtype in [torch.half, torch.float32, torch.double]:
            prob_dist = torch.rand(100, 10, device="mlu", dtype=dtype)
            for i in range(trials):
                gen.manual_seed(seed)
                samples_1 = torch.multinomial(prob_dist, n_sample, True, generator=gen)

                gen.manual_seed(seed)
                samples_2 = torch.multinomial(prob_dist, n_sample, True, generator=gen)

                self.assertEqual(samples_1, samples_2)
                self.assertEqual(samples_1.dim(), 2, msg="wrong number of dimensions")
                self.assertEqual(
                    samples_1.size(1), n_sample, msg="wrong number of samples"
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_multinomial_rng_state_advance(self):
        corpus_size = 100000
        dtype = torch.float32
        freqs = torch.ones(corpus_size, dtype=torch.float, device="mlu")
        n_sample = 100
        samples1 = torch.multinomial(freqs, n_sample, replacement=True)
        samples2 = torch.multinomial(freqs, n_sample, replacement=True)
        samples = torch.cat([samples1, samples2])
        # expect no more than 1 repeating elements generated in 2 attempts
        # the probability of at least element being repeated is surprisingly large, 18%
        self.assertLessEqual(2 * n_sample - samples.unique().size(0), 2)
        samples1 = torch.multinomial(freqs, n_sample, replacement=False)
        samples2 = torch.multinomial(freqs, n_sample, replacement=False)
        samples = torch.cat([samples1, samples2])
        # expect no more than 1 repeating elements generated in 2 attempts
        self.assertLessEqual(2 * n_sample - samples.unique().size(0), 1)

    # @unittest.skip("not test")
    @testinfo()
    def test_generator_mlu(self):
        # test default generators are equal
        self.assertEqual(torch.default_generator, torch.default_generator)

        # tests Generator API
        # manual_seed, seed, initial_seed, get_state, set_state
        g1 = torch.Generator(0)
        g2 = torch.Generator(0)
        g1.manual_seed(12345)
        g2.manual_seed(12345)
        self.assertEqual(g1.initial_seed(), g2.initial_seed())

        g1.seed()
        g2.seed()
        self.assertNotEqual(g1.initial_seed(), g2.initial_seed())

        g1 = torch.Generator(0)
        g2_state = g2.get_state()
        g2_randn = torch.randn(1, device="mlu", generator=g2)
        g1.set_state(g2_state)
        g1_randn = torch.randn(1, device="mlu", generator=g1)
        self.assertEqual(g1_randn, g2_randn)

        default_state = torch.mlu.default_generators[0].get_state()
        q = torch.empty(100).mlu()
        g1_normal = q.normal_()
        g2 = torch.Generator(0)
        g2.set_state(default_state)
        g2_normal = q.normal_(generator=g2)
        self.assertEqual(g1_normal, g2_normal)

    # @unittest.skip("not test")
    @testinfo()
    def test_torch_manual_seed_seeds_mlu_devices(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().mlu()
            torch.manual_seed(2)
            self.assertEqual(torch.mlu.initial_seed(), 2)
            x.uniform_()
            torch.manual_seed(2)
            y = x.clone().uniform_()
            self.assertEqual(x, y)
            self.assertEqual(torch.mlu.initial_seed(), 2)

    # @unittest.skip("not test")
    @testinfo()
    def test_manual_seed(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().mlu()
            torch.mlu.manual_seed(2)
            self.assertEqual(torch.mlu.initial_seed(), 2)
            x.uniform_()
            a = torch.bernoulli(torch.full_like(x, 0.5))
            torch.mlu.manual_seed(2)
            y = x.clone().uniform_()
            b = torch.bernoulli(torch.full_like(x, 0.5))
            self.assertEqual(x, y)
            self.assertEqual(a, b)
            self.assertEqual(torch.mlu.initial_seed(), 2)

    # @unittest.skip("not test")
    @testinfo()
    def test_get_set_rng_state_all(self):
        if torch.mlu.device_count() <= 1:
            return
        states = torch.mlu.get_rng_state_all()
        before0 = torch.mlu.FloatTensor(100, device=0).normal_()
        before1 = torch.mlu.FloatTensor(100, device=1).normal_()
        torch.mlu.set_rng_state_all(states)
        after0 = torch.mlu.FloatTensor(100, device=0).normal_()
        after1 = torch.mlu.FloatTensor(100, device=1).normal_()
        self.assertEqual(before0, after0, atol=0, rtol=0)
        self.assertEqual(before1, after1, atol=0, rtol=0)

    # @unittest.skip("not test")
    @testinfo()
    def test_RNGState(self):
        state = torch.mlu.get_rng_state()
        stateCloned = state.clone()
        before = torch.rand(1000, device="mlu")

        self.assertEqual(state.ne(stateCloned).long().sum(), 0, atol=0, rtol=0)

        torch.mlu.set_rng_state(state)
        after = torch.rand(1000, device="mlu")
        self.assertEqual(before, after, atol=0, rtol=0)

    # @unittest.skip("not test")
    @testinfo()
    def test_boxMullerState(self):
        torch.manual_seed(123)
        odd_number = 101
        seeded = torch.randn(odd_number, device="mlu")
        state = torch.mlu.get_rng_state()
        midstream = torch.randn(odd_number, device="mlu")
        torch.mlu.set_rng_state(state)
        repeat_midstream = torch.randn(odd_number, device="mlu")
        torch.mlu.manual_seed(123)
        reseeded = torch.randn(odd_number, device="mlu")
        self.assertEqual(
            midstream,
            repeat_midstream,
            atol=0,
            rtol=0,
            msg="get_rng_state/set_rng_state not generating same sequence of normally distributed numbers",
        )
        self.assertEqual(
            seeded,
            reseeded,
            atol=0,
            rtol=0,
            msg="repeated calls to manual_seed not generating same sequence of normally distributed numbers",
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_manual_seed(self):
        rng_state = torch.mlu.get_rng_state()
        torch.manual_seed(2)
        x = torch.randn(100, device="mlu")
        self.assertEqual(torch.mlu.initial_seed(), 2)
        torch.manual_seed(2)
        y = torch.randn(100, device="mlu")
        self.assertEqual(x, y)

        max_int64 = 0x7FFF_FFFF_FFFF_FFFF
        min_int64 = -max_int64 - 1
        max_uint64 = 0xFFFF_FFFF_FFFF_FFFF
        # Check all boundary cases of valid seed value inputs
        test_cases = [
            # (seed, expected_initial_seed)
            # Positive seeds should be unchanged
            (max_int64, max_int64),
            (max_int64 + 1, max_int64 + 1),
            (max_uint64, max_uint64),
            (0, 0),
            # Negative seeds wrap around starting from the largest seed value
            (-1, max_uint64),
            (min_int64, max_int64 + 1),
        ]
        for seed, expected_initial_seed in test_cases:
            torch.manual_seed(seed)
            actual_initial_seed = torch.mlu.initial_seed()
            msg = (
                "expected initial_seed() = %x after calling manual_seed(%x), but got %x instead"
                % (expected_initial_seed, seed, actual_initial_seed)
            )
            self.assertEqual(expected_initial_seed, actual_initial_seed, msg=msg)
        for invalid_seed in [min_int64 - 1, max_uint64 + 1]:
            with self.assertRaisesRegex(RuntimeError, r"Overflow when unpacking long"):
                torch.manual_seed(invalid_seed)

        torch.mlu.set_rng_state(rng_state)

    # @unittest.skip("not test")
    @testinfo()
    def test_RNGStateAliasing(self):
        # Fork the random number stream at this point
        gen = torch.Generator(0)
        gen.set_state(torch.mlu.get_rng_state())
        self.assertEqual(gen.get_state(), torch.mlu.get_rng_state())

        target_value = torch.rand(1000, device="mlu")
        # Dramatically alter the internal state of the main generator
        _ = torch.rand(100000, device="mlu")
        forked_value = torch.rand(1000, generator=gen, device="mlu")
        self.assertEqual(
            target_value,
            forked_value,
            atol=0,
            rtol=0,
            msg="RNG has not forked correctly.",
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_random_split(self):
        from torch.utils.data import random_split

        random_split(range(30), [0.3, 0.3, 0.4])
        # This should run without error

    # @unittest.skip("not test")
    @testinfo()
    def test_default_generator(self):
        from torch import default_generator

        msg = r"Expected a \'mlu\' device type for generator but found \'cpu\'"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.randn(10, generator=default_generator, device="mlu")


if __name__ == "__main__":
    unittest.main()
