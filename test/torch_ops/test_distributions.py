"""
Note [Randomized statistical tests]
-----------------------------------

This note describes how to maintain tests in this file as random sources
change. This file contains two types of randomized tests:

1. The easier type of randomized test are tests that should always pass but are
   initialized with random data. If these fail something is wrong, but it's
   fine to use a fixed seed by inheriting from common.TestCase.

2. The trickier tests are statistical tests. These tests explicitly call
   set_rng_seed(n) and are marked "see Note [Randomized statistical tests]".
   These statistical tests have a known positive failure rate
   (we set failure_rate=1e-3 by default). We need to balance strength of these
   tests with annoyance of false alarms. One way that works is to specifically
   set seeds in each of the randomized tests. When a random generator
   occasionally changes (as in #4312 vectorizing the Box-Muller sampler), some
   of these statistical tests may (rarely) fail. If one fails in this case,
   it's fine to increment the seed of the failing test (but you shouldn't need
   to increment it more than once; otherwise something is probably actually
   wrong).
"""

import math
import numbers
import unittest
from collections import namedtuple
from itertools import product
from random import shuffle

import torch
import torch_mlu

# TODO: remove this global setting
# Distributions tests use double as the default dtype
torch.set_default_dtype(torch.float32)

import math
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    set_rng_seed,
    TEST_WITH_UBSAN,
    load_tests,
    gradcheck,
)  # pylint: disable=C0411, C0413
from torch.autograd import grad  # pylint: disable=C0411, C0413
from torch.distributions import (
    Bernoulli,
    Categorical,
    Cauchy,
    ContinuousBernoulli,
    Distribution,
    Exponential,
    ExponentialFamily,
    Geometric,
    Gumbel,
    HalfCauchy,
    HalfNormal,
    Independent,
    Laplace,
    LogisticNormal,
    LogNormal,
    LowRankMultivariateNormal,
    MixtureSameFamily,
    Multinomial,
    MultivariateNormal,
    Normal,
    OneHotCategorical,
    OneHotCategoricalStraightThrough,
    Pareto,
    RelaxedBernoulli,
    TransformedDistribution,
    Uniform,
    VonMises,
    kl_divergence,
)  # pylint: disable=C0411, C0413
from torch.distributions.constraints import Constraint, is_dependent
from torch.distributions.kl import _kl_expfamily_expfamily
from torch.distributions.transforms import (
    AffineTransform,
    CatTransform,
    ExpTransform,
    StackTransform,
    identity_transform,
)
from torch.distributions.utils import probs_to_logits, lazy_property
from torch.nn.functional import softmax

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

TEST_NUMPY = True
try:
    import numpy as np
    import scipy.stats
    import scipy.special
except ImportError:
    TEST_NUMPY = False

inf = math.inf


def pairwise(Dist, *params):
    """
    Creates a pair of distributions `Dist` initialized to test each element of
    param with each other.
    """
    params1 = [torch.tensor([p] * len(p)).to("mlu") for p in params]
    params2 = [p.transpose(0, 1) for p in params1]
    return Dist(*params1), Dist(*params2)


def is_all_nan(tensor):
    """
    Checks if all entries of a tensor is nan.
    """
    return torch.isnan(tensor).all()


ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU = (
    "Could not run '.*' with arguments from the 'MLU' backend."
)

# Register all distributions for generic tests.
Example = namedtuple("Example", ["Dist", "params"])
EXAMPLES = [
    Example(
        Bernoulli,
        [
            {"probs": torch.tensor([0.7, 0.2, 0.4], requires_grad=True).to("mlu")},
            {"probs": torch.tensor([0.3], requires_grad=True).to("mlu")},
            {"logits": torch.tensor([0.0], requires_grad=True).to("mlu")},
        ],
    ),
    Example(
        Geometric,
        [
            {"probs": torch.tensor([0.7, 0.2, 0.4], requires_grad=True).to("mlu")},
            {"probs": torch.tensor([0.3], requires_grad=True).to("mlu")},
        ],
    ),
    Example(
        Categorical,
        [
            {
                "probs": torch.tensor(
                    [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                ).to("mlu")
            },
            {
                "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True).to(
                    "mlu"
                )
            },
            {
                "logits": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True).to(
                    "mlu"
                )
            },
        ],
    ),
    Example(
        Exponential,
        [
            {"rate": torch.randn(5, 5).to("mlu").abs().requires_grad_()},
            {"rate": torch.randn(1).to("mlu").abs().requires_grad_()},
        ],
    ),
    Example(
        Gumbel,
        [
            {
                "loc": torch.randn(5, 5, requires_grad=True).to("mlu"),
                "scale": torch.randn(5, 5).to("mlu").abs().requires_grad_(),
            },
            {
                "loc": torch.randn(1, requires_grad=True).to("mlu"),
                "scale": torch.randn(1).abs().requires_grad_().to("mlu"),
            },
        ],
    ),
    Example(
        HalfNormal,
        [
            {"scale": torch.randn(5, 5).abs().requires_grad_().to("mlu")},
            {"scale": torch.randn(1).abs().requires_grad_().to("mlu")},
            {"scale": torch.tensor([1e-5, 1e-5], requires_grad=True).to("mlu")},
        ],
    ),
    Example(
        Independent,
        [
            {
                "base_distribution": Normal(
                    torch.randn(2, 3, requires_grad=True).to("mlu"),
                    torch.randn(2, 3).to("mlu").abs().requires_grad_(),
                ),
                "reinterpreted_batch_ndims": 0,
            },
            {
                "base_distribution": Normal(
                    torch.randn(2, 3, requires_grad=True).to("mlu"),
                    torch.randn(2, 3).to("mlu").abs().requires_grad_(),
                ),
                "reinterpreted_batch_ndims": 1,
            },
            {
                "base_distribution": Normal(
                    torch.randn(2, 3, requires_grad=True).to("mlu"),
                    torch.randn(2, 3).to("mlu").abs().requires_grad_(),
                ),
                "reinterpreted_batch_ndims": 2,
            },
            {
                "base_distribution": Normal(
                    torch.randn(2, 3, 5, requires_grad=True).to("mlu"),
                    torch.randn(2, 3, 5).to("mlu").abs().requires_grad_(),
                ),
                "reinterpreted_batch_ndims": 2,
            },
            {
                "base_distribution": Normal(
                    torch.randn(2, 3, 5, requires_grad=True).to("mlu"),
                    torch.randn(2, 3, 5).to("mlu").abs().requires_grad_(),
                ),
                "reinterpreted_batch_ndims": 3,
            },
        ],
    ),
    Example(
        Laplace,
        [
            {
                "loc": torch.randn(5, 5, requires_grad=True).to("mlu"),
                "scale": torch.randn(5, 5).to("mlu").abs().requires_grad_(),
            },
            {
                "loc": torch.randn(1, requires_grad=True).to("mlu"),
                "scale": torch.randn(1).to("mlu").abs().requires_grad_(),
            },
            {
                "loc": torch.tensor([1.0, 0.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([1e-5, 1e-5], requires_grad=True).to("mlu"),
            },
        ],
    ),
    Example(
        LogNormal,
        [
            {
                "loc": torch.randn(5, 5, requires_grad=True).to("mlu"),
                "scale": torch.randn(5, 5).to("mlu").abs().requires_grad_(),
            },
            {
                "loc": torch.randn(1, requires_grad=True).to("mlu"),
                "scale": torch.randn(1).to("mlu").abs().requires_grad_(),
            },
            {
                "loc": torch.tensor([1.0, 0.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([1e-5, 1e-5], requires_grad=True).to("mlu"),
            },
        ],
    ),
    Example(
        LogisticNormal,
        [
            {
                "loc": torch.randn(5, 5).to("mlu").requires_grad_(),
                "scale": torch.randn(5, 5).to("mlu").abs().requires_grad_(),
            },
            {
                "loc": torch.randn(1).to("mlu").requires_grad_(),
                "scale": torch.randn(1).to("mlu").abs().requires_grad_(),
            },
            {
                "loc": torch.tensor([1.0, 0.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([1e-5, 1e-5], requires_grad=True).to("mlu"),
            },
        ],
    ),
    Example(
        Normal,
        [
            {
                "loc": torch.randn(5, 5, requires_grad=True).to("mlu"),
                "scale": torch.randn(5, 5).to("mlu").abs().requires_grad_(),
            },
            {
                "loc": torch.randn(1, requires_grad=True).to("mlu"),
                "scale": torch.randn(1).to("mlu").abs().requires_grad_(),
            },
            {
                "loc": torch.tensor([1.0, 0.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([1e-5, 1e-5], requires_grad=True).to("mlu"),
            },
        ],
    ),
    Example(
        OneHotCategorical,
        [
            {
                "probs": torch.tensor(
                    [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                ).to("mlu")
            },
            {
                "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True).to(
                    "mlu"
                )
            },
            {
                "logits": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True).to(
                    "mlu"
                )
            },
        ],
    ),
    Example(
        OneHotCategoricalStraightThrough,
        [
            {
                "probs": torch.tensor(
                    [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                ).to("mlu")
            },
            {
                "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True).to(
                    "mlu"
                )
            },
            {
                "logits": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True).to(
                    "mlu"
                )
            },
        ],
    ),
    Example(
        Pareto,
        [
            {
                "scale": torch.randn(5, 5).to("mlu").abs().requires_grad_(),
                "alpha": torch.randn(5, 5).to("mlu").abs().requires_grad_(),
            },
            {"scale": torch.tensor([1.0]).to("mlu"), "alpha": 1.0},
        ],
    ),
    Example(
        RelaxedBernoulli,
        [
            {
                "temperature": torch.tensor([0.5], requires_grad=True).to("mlu"),
                "probs": torch.tensor([0.7, 0.2, 0.4], requires_grad=True).to("mlu"),
            },
            {
                "temperature": torch.tensor([2.0]).to("mlu"),
                "probs": torch.tensor([0.3]).to("mlu"),
            },
            {
                "temperature": torch.tensor([7.2]).to("mlu"),
                "logits": torch.tensor([-2.0, 2.0, 1.0, 5.0]).to("mlu"),
            },
        ],
    ),
    Example(
        TransformedDistribution,
        [
            {
                "base_distribution": Normal(
                    torch.randn(2, 3, requires_grad=True).to("mlu"),
                    torch.randn(2, 3).to("mlu").abs().requires_grad_(),
                ),
                "transforms": [],
            },
            {
                "base_distribution": Normal(
                    torch.randn(2, 3, requires_grad=True).to("mlu"),
                    torch.randn(2, 3).to("mlu").abs().requires_grad_(),
                ),
                "transforms": ExpTransform(),
            },
            {
                "base_distribution": Normal(
                    torch.randn(2, 3, 5, requires_grad=True).to("mlu"),
                    torch.randn(2, 3, 5).to("mlu").abs().requires_grad_(),
                ),
                "transforms": [
                    AffineTransform(
                        torch.randn(3, 5).to("mlu"), torch.randn(3, 5).to("mlu")
                    ),
                    ExpTransform(),
                ],
            },
            {
                "base_distribution": Normal(
                    torch.randn(2, 3, 5, requires_grad=True).to("mlu"),
                    torch.randn(2, 3, 5).to("mlu").abs().requires_grad_(),
                ),
                "transforms": AffineTransform(1, 2),
            },
        ],
    ),
    Example(
        Uniform,
        [
            {
                "low": torch.zeros(5, 5, requires_grad=True).to("mlu"),
                "high": torch.ones(5, 5, requires_grad=True).to("mlu"),
            },
            {
                "low": torch.zeros(1, requires_grad=True).to("mlu"),
                "high": torch.ones(1, requires_grad=True).to("mlu"),
            },
            {
                "low": torch.tensor([1.0, 1.0], requires_grad=True).to("mlu"),
                "high": torch.tensor([2.0, 3.0], requires_grad=True).to("mlu"),
            },
        ],
    ),
    Example(
        VonMises,
        [
            {
                "loc": torch.tensor(1.0, requires_grad=True).to("mlu"),
                "concentration": torch.tensor(10.0, requires_grad=True).to("mlu"),
            },
            {
                "loc": torch.tensor([0.0, math.pi / 2], requires_grad=True).to("mlu"),
                "concentration": torch.tensor([1.0, 10.0], requires_grad=True).to(
                    "mlu"
                ),
            },
        ],
    ),
    Example(
        ContinuousBernoulli,
        [
            {"probs": torch.tensor([0.7, 0.2, 0.4], requires_grad=True).to("mlu")},
            {"probs": torch.tensor([0.3], requires_grad=True).to("mlu")},
            {"logits": torch.tensor([0.0], requires_grad=True).to("mlu")},
        ],
    ),
]

BAD_EXAMPLES = [
    Example(
        Bernoulli,
        [
            {"probs": torch.tensor([1.1, 0.2, 0.4], requires_grad=True).to("mlu")},
            {"probs": torch.tensor([-0.5], requires_grad=True).to("mlu")},
        ],
    ),
    Example(
        Geometric,
        [
            {"probs": torch.tensor([1.1, 0.2, 0.4], requires_grad=True).to("mlu")},
            {"probs": torch.tensor([-0.3], requires_grad=True).to("mlu")},
        ],
    ),
    Example(
        Categorical,
        [
            {
                "probs": torch.tensor(
                    [[-0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                ).to("mlu")
            },
            {
                "probs": torch.tensor(
                    [[-1.0, 10.0], [0.0, -1.0]], requires_grad=True
                ).to("mlu")
            },
        ],
    ),
    Example(
        Exponential,
        [
            {"rate": torch.tensor([0.0, 0.0], requires_grad=True).to("mlu")},
            {"rate": torch.tensor([-2.0], requires_grad=True).to("mlu")},
        ],
    ),
    Example(
        Gumbel,
        [
            {
                "loc": torch.tensor([1.0, 1.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([0.0, 1.0], requires_grad=True).to("mlu"),
            },
            {
                "loc": torch.tensor([1.0, 1.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([1.0, -1.0], requires_grad=True).to("mlu"),
            },
        ],
    ),
    Example(
        HalfNormal,
        [
            {"scale": torch.tensor([0.0, 1.0], requires_grad=True).to("mlu")},
            {"scale": torch.tensor([1.0, -1.0], requires_grad=True).to("mlu")},
        ],
    ),
    Example(
        Laplace,
        [
            {
                "loc": torch.tensor([1.0, 1.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([0.0, 1.0], requires_grad=True).to("mlu"),
            },
            {
                "loc": torch.tensor([1.0, 1.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([1.0, -1.0], requires_grad=True).to("mlu"),
            },
        ],
    ),
    Example(
        LogNormal,
        [
            {
                "loc": torch.tensor([1.0, 1.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([0.0, 1.0], requires_grad=True).to("mlu"),
            },
            {
                "loc": torch.tensor([1.0, 1.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([1.0, -1.0], requires_grad=True).to("mlu"),
            },
        ],
    ),
    Example(
        Normal,
        [
            {
                "loc": torch.tensor([1.0, 1.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([0.0, 1.0], requires_grad=True).to("mlu"),
            },
            {
                "loc": torch.tensor([1.0, 1.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([1.0, -1.0], requires_grad=True).to("mlu"),
            },
            {
                "loc": torch.tensor([1.0, 0.0], requires_grad=True).to("mlu"),
                "scale": torch.tensor([1e-5, -1e-5], requires_grad=True).to("mlu"),
            },
        ],
    ),
    Example(
        OneHotCategorical,
        [
            {
                "probs": torch.tensor(
                    [[0.1, 0.2, 0.3], [0.1, -10.0, 0.2]], requires_grad=True
                ).to("mlu")
            },
            {
                "probs": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True).to(
                    "mlu"
                )
            },
        ],
    ),
    Example(
        OneHotCategoricalStraightThrough,
        [
            {
                "probs": torch.tensor(
                    [[0.1, 0.2, 0.3], [0.1, -10.0, 0.2]], requires_grad=True
                ).to("mlu")
            },
            {
                "probs": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True).to(
                    "mlu"
                )
            },
        ],
    ),
    Example(
        Pareto,
        [
            {
                "scale": torch.tensor([0.0, 0.0], requires_grad=True).to("mlu"),
                "alpha": torch.tensor([-1e-5, 0.0], requires_grad=True).to("mlu"),
            },
            {"scale": torch.tensor([1.0]).to("mlu"), "alpha": -1.0},
        ],
    ),
    Example(
        RelaxedBernoulli,
        [
            {
                "temperature": torch.tensor([1.5], requires_grad=True).to("mlu"),
                "probs": torch.tensor([1.7, 0.2, 0.4], requires_grad=True).to("mlu"),
            },
            {
                "temperature": torch.tensor([2.0]).to("mlu"),
                "probs": torch.tensor([-1.0]).to("mlu"),
            },
        ],
    ),
    Example(
        Uniform,
        [
            {
                "low": torch.tensor([2.0], requires_grad=True).to("mlu"),
                "high": torch.tensor([2.0], requires_grad=True).to("mlu"),
            },
            {
                "low": torch.tensor([0.0], requires_grad=True).to("mlu"),
                "high": torch.tensor([0.0], requires_grad=True).to("mlu"),
            },
            {
                "low": torch.tensor([1.0], requires_grad=True).to("mlu"),
                "high": torch.tensor([0.0], requires_grad=True).to("mlu"),
            },
        ],
    ),
    Example(
        ContinuousBernoulli,
        [
            {"probs": torch.tensor([1.1, 0.2, 0.4], requires_grad=True).to("mlu")},
            {"probs": torch.tensor([-0.5], requires_grad=True).to("mlu")},
        ],
    ),
]


class TestDistributions(TestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _gradcheck_log_prob(self, dist_ctor, ctor_params):
        # TODO(fwg):Check this case when mlu support double type.
        # There is accuracy error when use float32 type on mlu, same on cpu.
        return
        # # performs gradient checks on log_prob
        # distribution = dist_ctor(*ctor_params)
        # s = distribution.sample()
        # if not distribution.support.is_discrete:
        #     s = s.detach().requires_grad_()

        # expected_shape = distribution.batch_shape + distribution.event_shape
        # self.assertEqual(s.size(), expected_shape)

        # def apply_fn(s, *params):
        #     return dist_ctor(*params).log_prob(s)

        # gradcheck(apply_fn, (s,) + tuple(ctor_params), raise_exception=True)

    def _check_log_prob(self, dist, asset_fn):
        # checks that the log_prob matches a reference function
        s = dist.sample()
        log_probs = dist.log_prob(s)
        log_probs_data_flat = log_probs.view(-1)
        s_data_flat = s.view(len(log_probs_data_flat), -1)
        for i, (val, log_prob) in enumerate(zip(s_data_flat, log_probs_data_flat)):
            asset_fn(i, val.squeeze(), log_prob)

    def _check_sampler_sampler(
        self,
        torch_dist,
        ref_dist,
        message,
        multivariate=False,
        circular=False,
        num_samples=10000,
        failure_rate=1e-3,
    ):
        # Checks that the .sample() method matches a reference function.
        torch_samples = torch_dist.sample((num_samples,)).squeeze()
        torch_samples = torch_samples.cpu().numpy()
        ref_samples = ref_dist.rvs(num_samples).astype(np.float64)
        if multivariate:
            # Project onto a random axis.
            axis = np.random.normal(size=torch_samples.shape[-1])
            axis /= np.linalg.norm(axis)
            torch_samples = np.dot(torch_samples, axis)
            ref_samples = np.dot(ref_samples, axis)
        samples = [(x, +1) for x in torch_samples] + [(x, -1) for x in ref_samples]
        if circular:
            samples = [(np.cos(x), v) for (x, v) in samples]
        shuffle(
            samples
        )  # necessary to prevent stable sort from making uneven bins for discrete
        samples.sort(key=lambda x: x[0])
        samples = np.array(samples)[:, 1]

        # Aggregate into bins filled with roughly zero-mean unit-variance RVs.
        num_bins = 10
        samples_per_bin = len(samples) // num_bins
        bins = samples.reshape((num_bins, samples_per_bin)).mean(axis=1)
        stddev = samples_per_bin**-0.5
        threshold = stddev * scipy.special.erfinv(1 - 2 * failure_rate / num_bins)
        message = "{}.sample() is biased:\n{}".format(message, bins)
        for bias in bins:
            self.assertLess(-threshold, bias, message)
            self.assertLess(bias, threshold, message)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def _check_sampler_discrete(
        self, torch_dist, ref_dist, message, num_samples=10000, failure_rate=1e-3
    ):
        """Runs a Chi2-test for the support, but ignores tail instead of combining"""
        torch_samples = torch_dist.sample((num_samples,)).squeeze()
        torch_samples = torch_samples.cpu().numpy()
        unique, counts = np.unique(torch_samples, return_counts=True)
        pmf = ref_dist.pmf(unique)
        pmf = pmf / pmf.sum()  # renormalize to 1.0 for chisq test
        msk = (counts > 5) & ((pmf * num_samples) > 5)
        self.assertGreater(
            pmf[msk].sum(),
            0.9,
            "Distribution is too sparse for test; try increasing num_samples",
        )
        # Add a remainder bucket that combines counts for all values
        # below threshold, if such values exist (i.e. mask has False entries).
        if not msk.all():
            counts = np.concatenate([counts[msk], np.sum(counts[~msk], keepdims=True)])
            pmf = np.concatenate([pmf[msk], np.sum(pmf[~msk], keepdims=True)])
        chisq, p = scipy.stats.chisquare(counts, pmf * num_samples)
        self.assertGreater(p, failure_rate, message)

    def _check_enumerate_support(self, dist, examples):
        for params, expected in examples:
            params = {k: torch.tensor(v).to("mlu") for k, v in params.items()}
            expected = torch.tensor(expected).to("mlu")
            d = dist(**params)
            actual = d.enumerate_support(expand=False)
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(actual, expected)
            actual = d.enumerate_support(expand=True)
            expected_with_expand = expected.expand(
                (-1,) + d.batch_shape + d.event_shape
            )
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(actual, expected_with_expand)

    def test_repr(self):
        for Dist, params in EXAMPLES:
            for param in params:
                dist = Dist(**param)
                self.assertTrue(repr(dist).startswith(dist.__class__.__name__))

    def test_sample_detached(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                variable_params = [
                    p for p in param.values() if getattr(p, "requires_grad", False)
                ]
                if not variable_params:
                    continue
                dist = Dist(**param)
                sample = dist.sample()
                self.assertFalse(
                    sample.requires_grad,
                    msg="{} example {}/{}, .sample() is not detached".format(
                        Dist.__name__, i + 1, len(params)
                    ),
                )

    def test_rsample_requires_grad(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                if not any(getattr(p, "requires_grad", False) for p in param.values()):
                    continue
                dist = Dist(**param)
                if not dist.has_rsample:
                    continue
                sample = dist.rsample()
                self.assertTrue(
                    sample.requires_grad,
                    msg="{} example {}/{}, .rsample() does not require grad".format(
                        Dist.__name__, i + 1, len(params)
                    ),
                )

    def test_enumerate_support_type(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    self.assertTrue(
                        type(dist.sample()) is type(dist.enumerate_support()),
                        msg=(
                            "{} example {}/{}, return type mismatch between "
                            + "sample and enumerate_support."
                        ).format(Dist.__name__, i + 1, len(params)),
                    )
                except NotImplementedError as e:
                    self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)

    def test_lazy_property_grad(self):
        x = torch.randn(1, requires_grad=True)
        x_mlu = x.to("mlu")

        class Dummy(object):
            @lazy_property
            def y(self):
                return x_mlu + 1

        def test():
            x.grad = None
            Dummy().y.backward()
            self.assertEqual(x.grad, torch.ones(1))

        test()
        with torch.no_grad():
            test()

    @unittest.skip("not support all distributinos on mlu")
    def test_has_examples(self):
        distributions_with_examples = {e.Dist for e in EXAMPLES}
        for Dist in globals().values():
            if (
                isinstance(Dist, type)
                and issubclass(Dist, Distribution)
                and Dist is not Distribution
                and Dist is not ExponentialFamily
            ):
                self.assertIn(
                    Dist,
                    distributions_with_examples,
                    "Please add {} to the EXAMPLES list in test_distributions.py".format(
                        Dist.__name__
                    ),
                )

    def test_support_attributes(self):
        for Dist, params in EXAMPLES:
            for param in params:
                d = Dist(**param)
                event_dim = len(d.event_shape)
                self.assertEqual(d.support.event_dim, event_dim)
                try:
                    self.assertEqual(Dist.support.event_dim, event_dim)
                except NotImplementedError as e:
                    self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)
                is_discrete = d.support.is_discrete
                try:
                    self.assertEqual(Dist.support.is_discrete, is_discrete)
                except NotImplementedError as e:
                    self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)

    def test_distribution_expand(self):
        shapes = [torch.Size(), torch.Size((2,)), torch.Size((2, 1))]
        for Dist, params in EXAMPLES:
            for param in params:
                for shape in shapes:
                    d = Dist(**param)
                    expanded_shape = shape + d.batch_shape
                    original_shape = d.batch_shape + d.event_shape
                    expected_shape = shape + original_shape
                    expanded = d.expand(batch_shape=list(expanded_shape))
                    sample = expanded.sample()
                    actual_shape = expanded.sample().shape
                    self.assertEqual(expanded.__class__, d.__class__)
                    self.assertEqual(d.sample().shape, original_shape)
                    self.assertEqual(expanded.log_prob(sample), d.log_prob(sample))
                    self.assertEqual(actual_shape, expected_shape)
                    self.assertEqual(expanded.batch_shape, expanded_shape)
                    try:
                        self.assertEqual(
                            expanded.mean, d.mean.expand(expanded_shape + d.event_shape)
                        )
                        self.assertEqual(
                            expanded.variance,
                            d.variance.expand(expanded_shape + d.event_shape),
                        )
                    except NotImplementedError as e:
                        self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)

    def test_distribution_subclass_expand(self):
        expand_by = torch.Size((2,))
        for Dist, params in EXAMPLES:

            class SubClass(Dist):
                pass

            for param in params:
                d = SubClass(**param)
                expanded_shape = expand_by + d.batch_shape
                original_shape = d.batch_shape + d.event_shape
                expected_shape = expand_by + original_shape
                expanded = d.expand(batch_shape=expanded_shape)
                sample = expanded.sample()
                actual_shape = expanded.sample().shape
                self.assertEqual(expanded.__class__, d.__class__)
                self.assertEqual(d.sample().shape, original_shape)
                self.assertEqual(expanded.log_prob(sample), d.log_prob(sample))
                self.assertEqual(actual_shape, expected_shape)

    def test_bernoulli(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        p_mlu = p.to("mlu")
        r = torch.tensor(0.3, requires_grad=True)
        r_mlu = r.to("mlu")
        self.assertEqual(Bernoulli(p_mlu).sample((8,)).size(), (8, 3))
        self.assertFalse(Bernoulli(p_mlu).sample().requires_grad)
        self.assertEqual(Bernoulli(r_mlu).sample((8,)).size(), (8,))
        self.assertEqual(Bernoulli(r_mlu).sample().size(), ())
        self.assertEqual(
            Bernoulli(r_mlu).sample((3, 2)).size(),
            (
                3,
                2,
            ),
        )
        self._gradcheck_log_prob(Bernoulli, (p_mlu,))

        def ref_log_prob(idx, val, log_prob):
            prob = p[idx]
            self.assertEqual(log_prob, math.log(prob if val else 1 - prob))

        self._check_log_prob(Bernoulli(p_mlu), ref_log_prob)
        self._check_log_prob(
            Bernoulli(logits=p_mlu.log() - (-p_mlu).log1p()), ref_log_prob
        )
        self.assertRaises(NotImplementedError, Bernoulli(r_mlu).rsample)

        # check entropy computation
        self.assertEqual(
            Bernoulli(p).entropy(),
            torch.tensor([0.6108, 0.5004, 0.6730]),
            atol=1e-4,
            rtol=0,
        )
        self.assertEqual(
            Bernoulli(torch.tensor([0.0]).to("mlu")).entropy(), torch.tensor([0.0])
        )

    def test_bernoulli_enumerate_support(self):
        examples = [
            ({"probs": [0.1]}, [[0], [1]]),
            ({"probs": [0.1, 0.9]}, [[0], [1]]),
            ({"probs": [[0.1, 0.2], [0.3, 0.4]]}, [[[0]], [[1]]]),
        ]
        self._check_enumerate_support(Bernoulli, examples)

    def test_bernoulli_3d(self):
        p = torch.full((2, 3, 5), 0.5).to("mlu").requires_grad_()
        self.assertEqual(Bernoulli(p).sample().size(), (2, 3, 5))
        self.assertEqual(
            Bernoulli(p).sample(sample_shape=(2, 5)).size(), (2, 5, 2, 3, 5)
        )
        self.assertEqual(Bernoulli(p).sample((2,)).size(), (2, 2, 3, 5))

    def test_geometric(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        p_mlu = p.to("mlu")
        r = torch.tensor(0.3, requires_grad=True)
        r_mlu = r.to("mlu")
        self.assertEqual(Geometric(p_mlu).sample((8,)).size(), (8, 3))
        self.assertFalse(Geometric(p_mlu).sample().requires_grad)
        self.assertEqual(Geometric(r_mlu).sample((8,)).size(), (8,))
        self.assertEqual(Geometric(r_mlu).sample().size(), ())
        self.assertEqual(Geometric(r_mlu).sample((3, 2)).size(), (3, 2))
        self._gradcheck_log_prob(Geometric, (p_mlu,))
        self.assertRaises(NotImplementedError, Geometric(r_mlu).rsample)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_geometric_log_prob_and_entropy(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        p_mlu = p.to("mlu")

        def ref_log_prob(idx, val, log_prob):
            prob = p[idx].detach()
            self.assertEqual(log_prob, scipy.stats.geom(prob, loc=-1).logpmf(val.cpu()))

        self._check_log_prob(Geometric(p_mlu), ref_log_prob)
        self._check_log_prob(
            Geometric(logits=p_mlu.log() - (-p_mlu).log1p()), ref_log_prob
        )

        # check entropy computation
        self.assertEqual(
            Geometric(p_mlu).entropy(),
            scipy.stats.geom(p.detach().numpy(), loc=-1).entropy().astype("float32"),
            atol=1e-3,
            rtol=0,
        )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_geometric_sample(self):
        set_rng_seed(123)  # see Note [Randomized statistical tests]
        for prob in [0.01, 0.18, 0.8]:
            prob_mlu = torch.tensor(prob, device="mlu")
            self._check_sampler_discrete(
                Geometric(prob_mlu),
                scipy.stats.geom(p=prob, loc=-1),
                "Geometric(prob={})".format(prob),
            )

    def test_categorical_1d(self):
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        p_mlu = p.to("mlu")
        self.assertTrue(is_all_nan(Categorical(p_mlu).mean))
        self.assertTrue(is_all_nan(Categorical(p_mlu).variance))
        self.assertEqual(Categorical(p_mlu).sample().size(), ())
        self.assertFalse(Categorical(p_mlu).sample().requires_grad)
        self.assertEqual(Categorical(p_mlu).sample((2, 2)).size(), (2, 2))
        self.assertEqual(Categorical(p_mlu).sample((1,)).size(), (1,))
        self._gradcheck_log_prob(Categorical, (p_mlu,))
        self.assertRaises(NotImplementedError, Categorical(p_mlu).rsample)

    def test_categorical_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        p = torch.tensor(probabilities, requires_grad=True)
        p_mlu = p.to("mlu")
        s = torch.tensor(probabilities_1, requires_grad=True)
        s_mlu = s.to("mlu")
        self.assertEqual(Categorical(p_mlu).mean.size(), (2,))
        self.assertEqual(Categorical(p_mlu).variance.size(), (2,))
        self.assertTrue(is_all_nan(Categorical(p_mlu).mean))
        self.assertTrue(is_all_nan(Categorical(p_mlu).variance))
        self.assertEqual(Categorical(p_mlu).sample().size(), (2,))
        self.assertEqual(
            Categorical(p_mlu).sample(sample_shape=(3, 4)).size(), (3, 4, 2)
        )
        self.assertEqual(Categorical(p_mlu).sample((6,)).size(), (6, 2))
        self._gradcheck_log_prob(Categorical, (p_mlu,))

        # sample check for extreme value of probs
        set_rng_seed(0)
        self.assertEqual(
            Categorical(s_mlu).sample(sample_shape=(2,)), torch.tensor([[0, 1], [0, 1]])
        )

        def ref_log_prob(idx, val, log_prob):
            sample_prob = p[idx][val] / p[idx].sum()
            self.assertEqual(log_prob, math.log(sample_prob))

        self._check_log_prob(Categorical(p_mlu), ref_log_prob)
        self._check_log_prob(Categorical(logits=p_mlu.log()), ref_log_prob)

        # check entropy computation
        self.assertEqual(
            Categorical(p_mlu).entropy(),
            torch.tensor([1.0114, 1.0297]),
            atol=1e-4,
            rtol=0,
        )
        self.assertEqual(Categorical(s_mlu).entropy(), torch.tensor([0.0, 0.0]))
        # issue gh-40553
        logits = p_mlu.log()
        logits[1, 1] = logits[0, 2] = float("-inf")
        e = Categorical(logits=logits).entropy()
        self.assertEqual(e, torch.tensor([0.6365, 0.5983]), atol=1e-4, rtol=0)

    def test_categorical_enumerate_support(self):
        examples = [
            ({"probs": [0.1, 0.2, 0.7]}, [0, 1, 2]),
            ({"probs": [[0.1, 0.9], [0.3, 0.7]]}, [[0], [1]]),
        ]
        self._check_enumerate_support(Categorical, examples)

    def test_one_hot_categorical_1d(self):
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        p_mlu = p.to("mlu")
        self.assertEqual(OneHotCategorical(p_mlu).sample().size(), (3,))
        self.assertFalse(OneHotCategorical(p_mlu).sample().requires_grad)
        self.assertEqual(OneHotCategorical(p_mlu).sample((2, 2)).size(), (2, 2, 3))
        self.assertEqual(OneHotCategorical(p_mlu).sample((1,)).size(), (1, 3))
        self._gradcheck_log_prob(OneHotCategorical, (p_mlu,))
        self.assertRaises(NotImplementedError, OneHotCategorical(p_mlu).rsample)

    def test_one_hot_categorical_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        p = torch.tensor(probabilities, requires_grad=True)
        p_mlu = p.to("mlu")
        self.assertEqual(OneHotCategorical(p_mlu).sample().size(), (2, 3))
        self.assertEqual(
            OneHotCategorical(p_mlu).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3)
        )
        self.assertEqual(OneHotCategorical(p_mlu).sample((6,)).size(), (6, 2, 3))
        self._gradcheck_log_prob(OneHotCategorical, (p_mlu,))

        dist = OneHotCategorical(p_mlu)
        x = dist.sample()
        self.assertEqual(dist.log_prob(x), Categorical(p_mlu).log_prob(x.max(-1)[1]))

    def test_one_hot_categorical_enumerate_support(self):
        examples = [
            ({"probs": [0.1, 0.2, 0.7]}, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ({"probs": [[0.1, 0.9], [0.3, 0.7]]}, [[[1, 0]], [[0, 1]]]),
        ]
        self._check_enumerate_support(OneHotCategorical, examples)

    def test_relaxed_bernoulli(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        p_mlu = p.to("mlu")
        r = torch.tensor(0.3, requires_grad=True)
        r_mlu = r.to("mlu")
        temp = torch.tensor(0.67, requires_grad=True)
        temp_mlu = temp.to("mlu")
        self.assertEqual(RelaxedBernoulli(temp_mlu, p_mlu).sample((8,)).size(), (8, 3))
        self.assertFalse(RelaxedBernoulli(temp_mlu, p_mlu).sample().requires_grad)
        self.assertEqual(RelaxedBernoulli(temp_mlu, r_mlu).sample((8,)).size(), (8,))
        self.assertEqual(RelaxedBernoulli(temp_mlu, r_mlu).sample().size(), ())
        self.assertEqual(
            RelaxedBernoulli(temp_mlu, r_mlu).sample((3, 2)).size(),
            (
                3,
                2,
            ),
        )
        self._gradcheck_log_prob(RelaxedBernoulli, (temp_mlu, p_mlu))
        self._gradcheck_log_prob(RelaxedBernoulli, (temp_mlu, r_mlu))

        # test that rsample doesn't fail
        s = RelaxedBernoulli(temp_mlu, p_mlu).rsample()
        s.backward(torch.ones_like(s))

    # TODO(sifengyang): this problem will be solved by PYTORCH-10150.
    @unittest.skip("not test")
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_rounded_relaxed_bernoulli(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]

        class Rounded(object):
            def __init__(self, dist):
                self.dist = dist

            def sample(self, *args, **kwargs):
                return torch.round(self.dist.sample(*args, **kwargs))

        for probs, temp in product([0.1, 0.2, 0.8], [0.1, 1.0, 10.0]):
            probs_mlu = torch.tensor(probs).to("mlu")
            temp_mlu = torch.tensor(temp).to("mlu")
            self._check_sampler_discrete(
                Rounded(RelaxedBernoulli(temp_mlu, probs_mlu)),
                scipy.stats.bernoulli(probs),
                "Rounded(RelaxedBernoulli(temp={}, probs={}))".format(temp, probs),
                failure_rate=1e-3,
            )

        for probs in [0.001, 0.2, 0.999]:
            equal_probs = torch.tensor(0.5)
            probs_mlu = torch.tensor(probs).to("mlu")
            dist = RelaxedBernoulli(1e10, probs_mlu)
            s = dist.rsample()
            self.assertEqual(equal_probs, s)

    def test_uniform(self):
        low = torch.zeros(5, 5, requires_grad=True)
        low_mlu = low.to("mlu")
        high = (torch.ones(5, 5) * 3).requires_grad_()
        high_mlu = high.to("mlu")
        low_1d = torch.zeros(1, requires_grad=True)
        low_1d_mlu = low_1d.to("mlu")
        high_1d = (torch.ones(1) * 3).requires_grad_()
        high_1d_mlu = high_1d.to("mlu")
        self.assertEqual(Uniform(low_mlu, high_mlu).sample().size(), (5, 5))
        self.assertEqual(Uniform(low_mlu, high_mlu).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Uniform(low_1d_mlu, high_1d_mlu).sample().size(), (1,))
        self.assertEqual(Uniform(low_1d_mlu, high_1d_mlu).sample((1,)).size(), (1, 1))

        # Check log_prob computation when value outside range
        uniform = Uniform(low_1d_mlu, high_1d_mlu, validate_args=False)
        above_high = torch.tensor([4.0])
        below_low = torch.tensor([-1.0])
        self.assertEqual(uniform.log_prob(above_high.mlu()).item(), -inf)
        self.assertEqual(uniform.log_prob(below_low.mlu()).item(), -inf)

        # check cdf computation when value outside range
        self.assertEqual(uniform.cdf(below_low.mlu()).item(), 0)
        self.assertEqual(uniform.cdf(above_high.mlu()).item(), 1)

        set_rng_seed(1)
        self._gradcheck_log_prob(Uniform, (low_mlu, high_mlu))
        self._gradcheck_log_prob(Uniform, (low_mlu, 1.0))
        self._gradcheck_log_prob(Uniform, (0.0, high_mlu))

        set_rng_seed(1)
        rand = low_mlu.new(low_mlu.size()).uniform_()
        set_rng_seed(1)
        u = Uniform(low_mlu, high_mlu).rsample()
        u.backward(torch.ones_like(u))
        self.assertEqual(low.grad, 1 - rand)
        self.assertEqual(high.grad, rand)
        low.grad.zero_()
        high.grad.zero_()

    # @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    # def test_vonmises_sample(self):
    #     for loc in [0.0, math.pi / 2.0]:
    #         for concentration in [0.03, 0.3, 1.0, 10.0, 100.0]:
    #             loc_mlu = torch.tensor(loc).to('mlu')
    #             concentration_mlu = torch.tensor(concentration).to('mlu')
    #             self._check_sampler_sampler(VonMises(loc_mlu, concentration_mlu),
    #                           scipy.stats.vonmises(loc=loc, kappa=concentration),
    #                           "VonMises(loc={}, concentration={})".format(loc, concentration),
    #                           num_samples=int(1e5), circular=True)

    def test_vonmises_logprob(self):
        concentrations = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
        for concentration in concentrations:
            concentration_mlu = torch.tensor(concentration).to("mlu")
            grid = torch.arange(0.0, 2 * math.pi, 1e-4)
            grid_mlu = grid.to("mlu")
            prob = VonMises(0.0, concentration_mlu).log_prob(grid_mlu).exp()
            norm = prob.mean().item() * 2 * math.pi
            self.assertLess(abs(norm - 1), 1e-3)

    def test_halfnormal(self):
        std = torch.randn(5, 5).abs().requires_grad_()
        std_mlu = std.to("mlu")
        std_1d = torch.randn(1).abs().requires_grad_()
        std_1d_mlu = std_1d.to("mlu")
        std_delta = torch.tensor([1e-5, 1e-5])
        std_delta_mlu = std_delta.to("mlu")
        self.assertEqual(HalfNormal(std_mlu).sample().size(), (5, 5))
        self.assertEqual(HalfNormal(std_mlu).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(HalfNormal(std_1d_mlu).sample((1,)).size(), (1, 1))
        self.assertEqual(HalfNormal(std_1d_mlu).sample().size(), (1,))

        # sample check for extreme value of std
        set_rng_seed(1)
        self.assertEqual(
            HalfNormal(std_delta_mlu).sample(sample_shape=(1, 2)),
            torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
            atol=1e-4,
            rtol=0,
        )

        self._gradcheck_log_prob(HalfNormal, (std_mlu,))

        # check .log_prob() can broadcast.
        dist = HalfNormal(torch.ones(2, 1, 4).mlu())
        log_prob = dist.log_prob(torch.ones(3, 1).mlu())
        self.assertEqual(log_prob.shape, (2, 3, 4))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_halfnormal_logprob(self):
        std = torch.randn(5, 1).abs().requires_grad_()
        std_mlu = std.to("mlu")

        def ref_log_prob(idx, x, log_prob):
            s = std.view(-1)[idx].detach()
            expected = scipy.stats.halfnorm(scale=s).logpdf(x.cpu())
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(HalfNormal(std_mlu), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_halfnormal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for std in [0.1, 1.0, 10.0]:
            std_mlu = torch.tensor(std).to("mlu")
            self._check_sampler_sampler(
                HalfNormal(std_mlu),
                scipy.stats.halfnorm(scale=std),
                "HalfNormal(scale={})".format(std),
            )

    def test_lognormal(self):
        mean = torch.randn(5, 5, requires_grad=True)
        mean_mlu = mean.to("mlu")
        std = torch.randn(5, 5).abs().requires_grad_()
        std_mlu = std.to("mlu")
        mean_1d = torch.randn(1, requires_grad=True)
        mean_1d_mlu = mean_1d.to("mlu")
        std_1d = torch.randn(1).abs().requires_grad_()
        std_1d_mlu = std_1d.to("mlu")
        mean_delta = torch.tensor([1.0, 0.0])
        mean_delta_mlu = mean_delta.to("mlu")
        std_delta = torch.tensor([1e-5, 1e-5])
        std_delta_mlu = std_delta.to("mlu")
        self.assertEqual(LogNormal(mean_mlu, std_mlu).sample().size(), (5, 5))
        self.assertEqual(LogNormal(mean_mlu, std_mlu).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(LogNormal(mean_1d_mlu, std_1d_mlu).sample((1,)).size(), (1, 1))
        self.assertEqual(LogNormal(mean_1d_mlu, std_1d_mlu).sample().size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(
            LogNormal(mean_delta_mlu, std_delta_mlu).sample(sample_shape=(1, 2)),
            torch.tensor([[[math.exp(1), 1.0], [math.exp(1), 1.0]]]),
            atol=1e-4,
            rtol=0,
        )

        self._gradcheck_log_prob(LogNormal, (mean_mlu, std_mlu))
        self._gradcheck_log_prob(LogNormal, (mean_mlu, 1.0))
        self._gradcheck_log_prob(LogNormal, (0.0, std_mlu))

        # check .log_prob() can broadcast.
        dist = LogNormal(torch.zeros(4).mlu(), torch.ones(2, 1, 1).mlu())
        log_prob = dist.log_prob(torch.ones(3, 1).mlu())
        self.assertEqual(log_prob.shape, (2, 3, 4))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lognormal_logprob(self):
        mean = torch.randn(5, 1, requires_grad=True)
        mean_mlu = mean.to("mlu")
        std = torch.randn(5, 1).abs().requires_grad_()
        std_mlu = std.to("mlu")

        def ref_log_prob(idx, x, log_prob):
            m = mean.view(-1)[idx].detach()
            s = std.view(-1)[idx].detach()
            expected = scipy.stats.lognorm(s=s, scale=math.exp(m)).logpdf(x.cpu())
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(LogNormal(mean_mlu, std_mlu), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lognormal_sample(self):
        # It's not passed for some MLU device type when seed=0.
        # See [CTR-4620] for more information.
        set_rng_seed(123)  # see Note [Randomized statistical tests]
        for mean, std in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            mean_mlu = torch.tensor(mean).to("mlu")
            std_mlu = torch.tensor(std).to("mlu")
            self._check_sampler_sampler(
                LogNormal(mean_mlu, std_mlu),
                scipy.stats.lognorm(scale=math.exp(mean), s=std),
                "LogNormal(loc={}, scale={})".format(mean, std),
            )

    def test_logisticnormal(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        mean = torch.randn(5, 5).requires_grad_()
        mean_mlu = mean.to("mlu")
        std = torch.randn(5, 5).abs().requires_grad_()
        std_mlu = std.to("mlu")
        mean_1d = torch.randn(1).requires_grad_()
        mean_1d_mlu = mean_1d.to("mlu")
        std_1d = torch.randn(1).abs().requires_grad_()
        std_1d_mlu = std_1d.to("mlu")
        mean_delta = torch.tensor([1.0, 0.0])
        mean_delta_mlu = mean_delta.to("mlu")
        std_delta = torch.tensor([1e-5, 1e-5])
        std_delta_mlu = std_delta.to("mlu")
        self.assertEqual(LogisticNormal(mean_mlu, std_mlu).sample().size(), (5, 6))
        self.assertEqual(
            LogisticNormal(mean_mlu, std_mlu).sample((7,)).size(), (7, 5, 6)
        )
        self.assertEqual(
            LogisticNormal(mean_1d_mlu, std_1d_mlu).sample((1,)).size(), (1, 2)
        )
        self.assertEqual(LogisticNormal(mean_1d_mlu, std_1d_mlu).sample().size(), (2,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(
            LogisticNormal(mean_delta_mlu, std_delta_mlu).sample(),
            torch.tensor(
                [
                    math.exp(1) / (1.0 + 1.0 + math.exp(1)),
                    1.0 / (1.0 + 1.0 + math.exp(1)),
                    1.0 / (1.0 + 1.0 + math.exp(1)),
                ]
            ),
            atol=1e-4,
            rtol=0,
        )

        # TODO: gradcheck seems to mutate the sample values so that the simplex
        # constraint fails by a very small margin.
        self._gradcheck_log_prob(
            lambda m, s: LogisticNormal(m, s, validate_args=False), (mean_mlu, std_mlu)
        )
        self._gradcheck_log_prob(
            lambda m, s: LogisticNormal(m, s, validate_args=False), (mean_mlu, 1.0)
        )
        self._gradcheck_log_prob(
            lambda m, s: LogisticNormal(m, s, validate_args=False), (0.0, std_mlu)
        )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_logisticnormal_logprob(self):
        mean = torch.randn(5, 7).requires_grad_()
        mean_mlu = mean.to("mlu")
        std = torch.randn(5, 7).abs().requires_grad_()
        std_mlu = std.to("mlu")

        # Smoke test for now
        # TODO: Once _check_log_prob works with multidimensional distributions,
        #       add proper testing of the log probabilities.
        dist = LogisticNormal(mean_mlu, std_mlu)
        assert dist.log_prob(dist.sample()).detach().cpu().numpy().shape == (5,)

    def _get_logistic_normal_ref_sampler(self, base_dist):
        def _sampler(num_samples):
            x = base_dist.rvs(num_samples)
            offset = np.log((x.shape[-1] + 1) - np.ones_like(x).cumsum(-1))
            z = 1.0 / (1.0 + np.exp(offset - x))
            z_cumprod = np.cumprod(1 - z, axis=-1)
            y1 = np.pad(z, ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
            y2 = np.pad(
                z_cumprod, ((0, 0), (1, 0)), mode="constant", constant_values=1.0
            )
            return y1 * y2

        return _sampler

    # TODO(huangqipeng) Not all MLU can pass with seed=0,
    # which highly rely on the initial value of the random generator
    @unittest.skip("not test")
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_logisticnormal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        means = map(np.asarray, [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)])
        covs = map(np.diag, [(0.1, 0.1), (1.0, 1.0), (10.0, 10.0)])
        for mean, cov in product(means, covs):
            base_dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            ref_dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            ref_dist.rvs = self._get_logistic_normal_ref_sampler(base_dist)
            mean_th = torch.tensor(mean).to("mlu")
            std_th = torch.tensor(np.sqrt(np.diag(cov))).to("mlu")
            self._check_sampler_sampler(
                LogisticNormal(mean_th, std_th),
                ref_dist,
                "LogisticNormal(loc={}, scale={})".format(mean_th, std_th),
                multivariate=True,
            )

    def test_mixture_same_family_shape(self):
        normal_case_1d = MixtureSameFamily(
            Categorical(torch.rand(5).mlu()),
            Normal(torch.randn(5).mlu(), torch.rand(5).mlu()),
        )
        normal_case_1d_batch = MixtureSameFamily(
            Categorical(torch.rand(3, 5).mlu()),
            Normal(torch.randn(3, 5).mlu(), torch.rand(3, 5).mlu()),
        )
        normal_case_1d_multi_batch = MixtureSameFamily(
            Categorical(torch.rand(4, 3, 5).mlu()),
            Normal(torch.randn(4, 3, 5).mlu(), torch.rand(4, 3, 5).mlu()),
        )
        normal_case_2d = MixtureSameFamily(
            Categorical(torch.rand(5).mlu()),
            Independent(Normal(torch.randn(5, 2).mlu(), torch.rand(5, 2).mlu()), 1),
        )
        normal_case_2d_batch = MixtureSameFamily(
            Categorical(torch.rand(3, 5).mlu()),
            Independent(
                Normal(torch.randn(3, 5, 2).mlu(), torch.rand(3, 5, 2).mlu()), 1
            ),
        )
        normal_case_2d_multi_batch = MixtureSameFamily(
            Categorical(torch.rand(4, 3, 5).mlu()),
            Independent(
                Normal(torch.randn(4, 3, 5, 2).mlu(), torch.rand(4, 3, 5, 2).mlu()), 1
            ),
        )

        self.assertEqual(normal_case_1d.sample().size(), ())
        self.assertEqual(normal_case_1d.sample((2,)).size(), (2,))
        self.assertEqual(normal_case_1d.sample((2, 7)).size(), (2, 7))
        self.assertEqual(normal_case_1d_batch.sample().size(), (3,))
        self.assertEqual(normal_case_1d_batch.sample((2,)).size(), (2, 3))
        self.assertEqual(normal_case_1d_batch.sample((2, 7)).size(), (2, 7, 3))
        self.assertEqual(normal_case_1d_multi_batch.sample().size(), (4, 3))
        self.assertEqual(normal_case_1d_multi_batch.sample((2,)).size(), (2, 4, 3))
        self.assertEqual(normal_case_1d_multi_batch.sample((2, 7)).size(), (2, 7, 4, 3))

        self.assertEqual(normal_case_2d.sample().size(), (2,))
        self.assertEqual(normal_case_2d.sample((2,)).size(), (2, 2))
        self.assertEqual(normal_case_2d.sample((2, 7)).size(), (2, 7, 2))
        self.assertEqual(normal_case_2d_batch.sample().size(), (3, 2))
        self.assertEqual(normal_case_2d_batch.sample((2,)).size(), (2, 3, 2))
        self.assertEqual(normal_case_2d_batch.sample((2, 7)).size(), (2, 7, 3, 2))
        self.assertEqual(normal_case_2d_multi_batch.sample().size(), (4, 3, 2))
        self.assertEqual(normal_case_2d_multi_batch.sample((2,)).size(), (2, 4, 3, 2))
        self.assertEqual(
            normal_case_2d_multi_batch.sample((2, 7)).size(), (2, 7, 4, 3, 2)
        )

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_mixture_same_family_log_prob(self):
        probs = torch.rand(5, 5).softmax(dim=-1)
        probs_mlu = probs.to("mlu")
        loc = torch.randn(5, 5)
        loc_mlu = loc.to("mlu")
        scale = torch.rand(5, 5)
        scale_mlu = scale.to("mlu")

        def ref_log_prob(idx, x, log_prob):
            p = probs[idx].numpy()
            m = loc[idx].numpy()
            s = scale[idx].numpy()
            mix = scipy.stats.multinomial(1, p)
            comp = scipy.stats.norm(m, s)
            expected = scipy.special.logsumexp(comp.logpdf(x.cpu()) + np.log(mix.p))
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(
            MixtureSameFamily(Categorical(probs=probs_mlu), Normal(loc_mlu, scale_mlu)),
            ref_log_prob,
        )

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_mixture_same_family_sample(self):
        probs = torch.rand(5).softmax(dim=-1)
        probs_mlu = probs.to("mlu")
        loc = torch.randn(5)
        loc_mlu = loc.to("mlu")
        scale = torch.rand(5)
        scale_mlu = scale.to("mlu")

        class ScipyMixtureNormal(object):
            def __init__(self, probs, mu, std):
                self.probs = probs
                self.mu = mu
                self.std = std

            def rvs(self, n_sample):
                comp_samples = [
                    scipy.stats.norm(m, s).rvs(n_sample)
                    for m, s in zip(self.mu, self.std)
                ]
                mix_samples = scipy.stats.multinomial(1, self.probs).rvs(n_sample)
                samples = []
                for i in range(n_sample):
                    samples.append(comp_samples[mix_samples[i].argmax()][i])
                return np.asarray(samples)

        self._check_sampler_sampler(
            MixtureSameFamily(Categorical(probs=probs_mlu), Normal(loc_mlu, scale_mlu)),
            ScipyMixtureNormal(probs.numpy(), loc.numpy(), scale.numpy()),
            """MixtureSameFamily(Categorical(probs={}),
            Normal(loc={}, scale={}))""".format(
                probs, loc, scale
            ),
        )

    def test_normal(self):
        loc = torch.randn(5, 5, requires_grad=True)
        loc_mlu = loc.to("mlu")
        scale = torch.randn(5, 5).abs().requires_grad_()
        scale_mlu = scale.to("mlu")
        loc_1d = torch.randn(1, requires_grad=True)
        loc_1d_mlu = loc_1d.to("mlu")
        scale_1d = torch.randn(1).abs().requires_grad_()
        scale_1d_mlu = scale_1d.to("mlu")
        loc_delta = torch.tensor([1.0, 0.0])
        loc_delta_mlu = loc_delta.to("mlu")
        scale_delta = torch.tensor([1e-5, 1e-5])
        scale_delta_mlu = scale_delta.to("mlu")
        self.assertEqual(Normal(loc_mlu, scale_mlu).sample().size(), (5, 5))
        self.assertEqual(Normal(loc_mlu, scale_mlu).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Normal(loc_1d_mlu, scale_1d_mlu).sample((1,)).size(), (1, 1))
        self.assertEqual(Normal(loc_1d_mlu, scale_1d_mlu).sample().size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(
            Normal(loc_delta_mlu, scale_delta_mlu).sample(sample_shape=(1, 2)),
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            atol=1e-4,
            rtol=0,
        )

        self._gradcheck_log_prob(Normal, (loc_mlu, scale_mlu))
        self._gradcheck_log_prob(Normal, (loc_mlu, 1.0))
        self._gradcheck_log_prob(Normal, (0.0, scale_mlu))

        set_rng_seed(1)
        eps = torch.normal(torch.zeros_like(loc).mlu(), torch.ones_like(scale).mlu())
        set_rng_seed(1)
        z = Normal(loc_mlu, scale_mlu).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(loc.grad, torch.ones_like(loc))
        self.assertEqual(scale.grad, eps)
        loc.grad.zero_()
        scale.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        def ref_log_prob(idx, x, log_prob):
            m = loc.view(-1)[idx]
            s = scale.view(-1)[idx]
            expected = math.exp(-((x - m) ** 2) / (2 * s**2)) / math.sqrt(
                2 * math.pi * s**2
            )
            self.assertEqual(log_prob, math.log(expected), atol=1e-3, rtol=0)

        self._check_log_prob(Normal(loc_mlu, scale_mlu), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_normal_sample(self):
        # It's not passed for some MLU device type when seed=0.
        # See [CTR-4620] for more information.
        set_rng_seed(123)  # see Note [Randomized statistical tests]
        for loc, scale in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            loc_mlu = torch.tensor(loc).to("mlu")
            scale_mlu = torch.tensor(scale).to("mlu")
            self._check_sampler_sampler(
                Normal(loc_mlu, scale_mlu),
                scipy.stats.norm(loc=loc, scale=scale),
                "Normal(mean={}, std={})".format(loc, scale),
            )

    def test_exponential(self):
        rate = torch.randn(5, 5).abs().requires_grad_()
        rate_mlu = rate.to("mlu")
        rate_1d = torch.randn(1).abs().requires_grad_()
        rate_1d_mlu = rate_1d.to("mlu")
        self.assertEqual(Exponential(rate_mlu).sample().size(), (5, 5))
        self.assertEqual(Exponential(rate_mlu).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Exponential(rate_1d_mlu).sample((1,)).size(), (1, 1))
        self.assertEqual(Exponential(rate_1d_mlu).sample().size(), (1,))

        self._gradcheck_log_prob(Exponential, (rate_mlu,))
        set_rng_seed(1)
        eps = rate_mlu.new(rate_mlu.size()).exponential_()
        set_rng_seed(1)
        z = Exponential(rate_mlu).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(rate.grad, -eps.cpu() / rate**2)
        rate.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        def ref_log_prob(idx, x, log_prob):
            m = rate.view(-1)[idx]
            expected = math.log(m) - m * x.cpu()
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Exponential(rate_mlu), ref_log_prob)

    # TODO(huangqipeng) Not all MLU can pass with seed=1,
    # which highly rely on the initial value of the random generator
    @unittest.skip("not test")
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_exponential_sample(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for rate in [1e-5, 1.0, 10.0]:
            rate_mlu = torch.tensor(rate).to("mlu")
            self._check_sampler_sampler(
                Exponential(rate_mlu),
                scipy.stats.expon(scale=1.0 / rate),
                "Exponential(rate={})".format(rate),
            )

    def test_laplace(self):
        loc = torch.randn(5, 5, requires_grad=True)
        loc_mlu = loc.to("mlu")
        scale = torch.randn(5, 5).abs().requires_grad_()
        scale_mlu = scale.to("mlu")
        loc_1d = torch.randn(1, requires_grad=True)
        loc_1d_mlu = loc_1d.to("mlu")
        scale_1d = torch.randn(1, requires_grad=True)
        scale_1d_mlu = scale_1d.to("mlu")
        loc_delta = torch.tensor([1.0, 0.0])
        loc_delta_mlu = loc_delta.to("mlu")
        scale_delta = torch.tensor([1e-5, 1e-5])
        scale_delta_mlu = scale_delta.to("mlu")
        self.assertEqual(Laplace(loc_mlu, scale_mlu).sample().size(), (5, 5))
        self.assertEqual(Laplace(loc_mlu, scale_mlu).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Laplace(loc_1d_mlu, scale_1d_mlu).sample((1,)).size(), (1, 1))
        self.assertEqual(Laplace(loc_1d_mlu, scale_1d_mlu).sample().size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(0)
        self.assertEqual(
            Laplace(loc_delta_mlu, scale_delta_mlu).sample(sample_shape=(1, 2)),
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            atol=1e-4,
            rtol=0,
        )

        self._gradcheck_log_prob(Laplace, (loc_mlu, scale_mlu))
        self._gradcheck_log_prob(Laplace, (loc_mlu, 1.0))
        self._gradcheck_log_prob(Laplace, (0.0, scale_mlu))

        set_rng_seed(1)
        eps = torch.ones_like(loc_mlu).uniform_(-0.5, 0.5)
        set_rng_seed(1)
        z = Laplace(loc_mlu, scale_mlu).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(loc.grad, torch.ones_like(loc))
        self.assertEqual(
            scale.grad, -eps.cpu().sign() * torch.log1p(-2 * eps.cpu().abs())
        )
        loc.grad.zero_()
        scale.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        def ref_log_prob(idx, x, log_prob):
            m = loc.view(-1)[idx]
            s = scale.view(-1)[idx]
            expected = -math.log(2 * s) - abs(x.cpu() - m) / s
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Laplace(loc_mlu, scale_mlu), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_pareto(self):
        scale = torch.randn(2, 3).abs().requires_grad_()
        scale_mlu = scale.to("mlu")
        alpha = torch.randn(2, 3).abs().requires_grad_()
        alpha_mlu = alpha.to("mlu")
        scale_1d = torch.randn(1).abs().requires_grad_()
        scale_1d_mlu = scale_1d.to("mlu")
        alpha_1d = torch.randn(1).abs().requires_grad_()
        alpha_1d_mlu = alpha_1d.to("mlu")
        self.assertEqual(Pareto(scale_1d_mlu, 0.5).mean, inf)
        self.assertEqual(Pareto(scale_1d_mlu, 0.5).variance, inf)
        self.assertEqual(Pareto(scale_mlu, alpha_mlu).sample().size(), (2, 3))
        self.assertEqual(Pareto(scale_mlu, alpha_mlu).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Pareto(scale_1d_mlu, alpha_1d_mlu).sample((1,)).size(), (1, 1))
        self.assertEqual(Pareto(scale_1d_mlu, alpha_1d_mlu).sample().size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            s = scale.view(-1)[idx].detach()
            a = alpha.view(-1)[idx].detach()
            expected = scipy.stats.pareto.logpdf(x.cpu(), a, scale=s)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Pareto(scale_mlu, alpha_mlu), ref_log_prob)

    # TODO(huangqipeng) Not all MLU can pass with seed=1,
    # which highly rely on the initial value of the random generator
    @unittest.skip("not test")
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_pareto_sample(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for scale, alpha in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            scale_mlu = torch.tensor(scale).to("mlu")
            alpha_mlu = torch.tensor(alpha).to("mlu")
            self._check_sampler_sampler(
                Pareto(scale_mlu, alpha_mlu),
                scipy.stats.pareto(alpha, scale=scale),
                "Pareto(scale={}, alpha={})".format(scale, alpha),
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gumbel(self):
        loc = torch.randn(2, 3, requires_grad=True)
        loc_mlu = loc.to("mlu")
        scale = torch.randn(2, 3).abs().requires_grad_()
        scale_mlu = scale.to("mlu")
        loc_1d = torch.randn(1, requires_grad=True)
        loc_1d_mlu = loc_1d.to("mlu")
        scale_1d = torch.randn(1).abs().requires_grad_()
        scale_1d_mlu = scale_1d.to("mlu")
        self.assertEqual(Gumbel(loc_mlu, scale_mlu).sample().size(), (2, 3))
        self.assertEqual(Gumbel(loc_mlu, scale_mlu).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Gumbel(loc_1d_mlu, scale_1d_mlu).sample().size(), (1,))
        self.assertEqual(Gumbel(loc_1d_mlu, scale_1d_mlu).sample((1,)).size(), (1, 1))

        def ref_log_prob(idx, x, log_prob):
            l = loc.view(-1)[idx].detach()
            s = scale.view(-1)[idx].detach()
            expected = scipy.stats.gumbel_r.logpdf(x.cpu(), loc=l, scale=s)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Gumbel(loc_mlu, scale_mlu), ref_log_prob)

    def test_continuous_bernoulli(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        p_mlu = p.to("mlu")
        r = torch.tensor(0.3, requires_grad=True)
        r_mlu = r.to("mlu")
        self.assertEqual(ContinuousBernoulli(p_mlu).sample((8,)).size(), (8, 3))
        self.assertFalse(ContinuousBernoulli(p_mlu).sample().requires_grad)
        self.assertEqual(ContinuousBernoulli(r_mlu).sample((8,)).size(), (8,))
        self.assertEqual(ContinuousBernoulli(r_mlu).sample().size(), ())
        self.assertEqual(
            ContinuousBernoulli(r_mlu).sample((3, 2)).size(),
            (
                3,
                2,
            ),
        )
        self._gradcheck_log_prob(ContinuousBernoulli, (p_mlu,))

        def ref_log_prob(idx, val, log_prob):
            prob = p[idx]
            if prob > 0.499 and prob < 0.501:  # using default value of lim here
                log_norm_const = (
                    math.log(2.0)
                    + 4.0 / 3.0 * math.pow(prob - 0.5, 2)
                    + 104.0 / 45.0 * math.pow(prob - 0.5, 4)
                )
            else:
                log_norm_const = math.log(
                    2.0 * math.atanh(1.0 - 2.0 * prob) / (1.0 - 2.0 * prob)
                )
            res = (
                val * math.log(prob) + (1.0 - val) * math.log1p(-prob) + log_norm_const
            )
            self.assertEqual(log_prob, res)

        self._check_log_prob(ContinuousBernoulli(p_mlu), ref_log_prob)
        self._check_log_prob(
            ContinuousBernoulli(logits=p_mlu.log() - (-p_mlu).log1p()), ref_log_prob
        )

        # check entropy computation
        self.assertEqual(
            ContinuousBernoulli(p_mlu).entropy(),
            torch.tensor([-0.02938, -0.07641, -0.00682]),
            atol=1e-4,
            rtol=0,
        )
        # entropy below corresponds to the clamped value of prob when using float 64
        # the value for float32 should be -1.76898
        self.assertEqual(
            ContinuousBernoulli(torch.tensor([0.0]).mlu()).entropy(),
            torch.tensor([-1.76898]),
            atol=1e-5,
            rtol=0,
        )

    def test_continuous_bernoulli_3d(self):
        p = torch.full((2, 3, 5), 0.5).requires_grad_()
        p_mlu = p.to("mlu")
        self.assertEqual(ContinuousBernoulli(p_mlu).sample().size(), (2, 3, 5))
        self.assertEqual(
            ContinuousBernoulli(p_mlu).sample(sample_shape=(2, 5)).size(),
            (2, 5, 2, 3, 5),
        )
        self.assertEqual(ContinuousBernoulli(p_mlu).sample((2,)).size(), (2, 2, 3, 5))

    def test_independent_shape(self):
        for Dist, params in EXAMPLES:
            for param in params:
                base_dist = Dist(**param)
                x = base_dist.sample()
                base_log_prob_shape = base_dist.log_prob(x).shape
                for reinterpreted_batch_ndims in range(len(base_dist.batch_shape) + 1):
                    indep_dist = Independent(base_dist, reinterpreted_batch_ndims)
                    indep_log_prob_shape = base_log_prob_shape[
                        : len(base_log_prob_shape) - reinterpreted_batch_ndims
                    ]
                    self.assertEqual(indep_dist.log_prob(x).shape, indep_log_prob_shape)
                    self.assertEqual(
                        indep_dist.sample().shape, base_dist.sample().shape
                    )
                    self.assertEqual(indep_dist.has_rsample, base_dist.has_rsample)
                    if indep_dist.has_rsample:
                        self.assertEqual(
                            indep_dist.sample().shape, base_dist.sample().shape
                        )
                    try:
                        self.assertEqual(
                            indep_dist.enumerate_support().shape,
                            base_dist.enumerate_support().shape,
                        )
                        self.assertEqual(indep_dist.mean.shape, base_dist.mean.shape)
                    except NotImplementedError as e:
                        self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)
                    try:
                        self.assertEqual(
                            indep_dist.variance.shape, base_dist.variance.shape
                        )
                    except NotImplementedError as e:
                        self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)
                    try:
                        self.assertEqual(
                            indep_dist.entropy().shape, indep_log_prob_shape
                        )
                    except NotImplementedError as e:
                        self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)

    def test_independent_expand(self):
        for Dist, params in EXAMPLES:
            for param in params:
                base_dist = Dist(**param)
                for reinterpreted_batch_ndims in range(len(base_dist.batch_shape) + 1):
                    for s in [torch.Size(), torch.Size((2,)), torch.Size((2, 3))]:
                        indep_dist = Independent(base_dist, reinterpreted_batch_ndims)
                        expanded_shape = s + indep_dist.batch_shape
                        expanded = indep_dist.expand(expanded_shape)
                        expanded_sample = expanded.sample()
                        expected_shape = expanded_shape + indep_dist.event_shape
                        self.assertEqual(expanded_sample.shape, expected_shape)
                        self.assertEqual(
                            expanded.log_prob(expanded_sample),
                            indep_dist.log_prob(expanded_sample),
                        )
                        self.assertEqual(expanded.event_shape, indep_dist.event_shape)
                        self.assertEqual(expanded.batch_shape, expanded_shape)

    # TODO(huangqipeng) Not all MLU can pass with seed=0,
    # which highly rely on the initial value of the random generator
    @unittest.skip("not test")
    def test_cdf_icdf_inverse(self):
        # Tests the invertibility property on the distributions
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                samples = dist.sample(sample_shape=(20,))
                try:
                    cdf = dist.cdf(samples)
                    actual = dist.icdf(cdf)
                except NotImplementedError as e:
                    self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)
                    continue
                rel_error = torch.abs(actual - samples) / (1e-10 + torch.abs(samples))
                # MLU modify default type from double to float32, so modify threshold to 1e-3.
                self.assertLess(
                    rel_error.max(),
                    1e-3,
                    msg="\n".join(
                        [
                            "{} example {}/{}, icdf(cdf(x)) != x".format(
                                Dist.__name__, i + 1, len(params)
                            ),
                            "x = {}".format(samples),
                            "cdf(x) = {}".format(cdf),
                            "icdf(cdf(x)) = {}".format(actual),
                        ]
                    ),
                )

    # TODO(sifengyang): this problem will be solved by PYTORCH-10150.
    @unittest.skip("not test")
    def test_cdf_log_prob(self):
        # Tests if the differentiation of the CDF gives the PDF at a given value
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                samples = dist.sample()
                if not dist.support.is_discrete:
                    samples.requires_grad_()
                try:
                    cdfs = dist.cdf(samples)
                    pdfs = dist.log_prob(samples).exp()
                except NotImplementedError as e:
                    self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)
                    continue
                cdfs_derivative = grad(cdfs.sum(), [samples])[
                    0
                ]  # this should not be wrapped in torch.abs()
                self.assertEqual(
                    cdfs_derivative,
                    pdfs,
                    msg="\n".join(
                        [
                            "{} example {}/{}, d(cdf)/dx != pdf(x)".format(
                                Dist.__name__, i + 1, len(params)
                            ),
                            "x = {}".format(samples),
                            "cdf = {}".format(cdfs),
                            "pdf = {}".format(pdfs),
                            "grad(cdf) = {}".format(cdfs_derivative),
                        ]
                    ),
                )

    def test_valid_parameter_broadcasting(self):
        # Test correct broadcasting of parameter sizes for distributions that have multiple
        # parameters.
        # example type (distribution instance, expected sample shape)
        valid_examples = [
            (Normal(loc=torch.tensor([0.0, 0.0]).to("mlu"), scale=1), (2,)),
            (Normal(loc=0, scale=torch.tensor([1.0, 1.0]).to("mlu")), (2,)),
            (
                Normal(
                    loc=torch.tensor([0.0, 0.0]).to("mlu"),
                    scale=torch.tensor([1.0]).to("mlu"),
                ),
                (2,),
            ),
            (
                Normal(
                    loc=torch.tensor([0.0, 0.0]).to("mlu"),
                    scale=torch.tensor([[1.0], [1.0]]).to("mlu"),
                ),
                (2, 2),
            ),
            (
                Normal(
                    loc=torch.tensor([0.0, 0.0]).to("mlu"),
                    scale=torch.tensor([[1.0]]).to("mlu"),
                ),
                (1, 2),
            ),
            (
                Normal(
                    loc=torch.tensor([0.0]).to("mlu"),
                    scale=torch.tensor([[1.0]]).to("mlu"),
                ),
                (1, 1),
            ),
            (Gumbel(loc=torch.tensor([0.0, 0.0]).to("mlu"), scale=1), (2,)),
            (Gumbel(loc=0, scale=torch.tensor([1.0, 1.0]).to("mlu")), (2,)),
            (
                Gumbel(
                    loc=torch.tensor([0.0, 0.0]).to("mlu"),
                    scale=torch.tensor([1.0]).to("mlu"),
                ),
                (2,),
            ),
            (
                Gumbel(
                    loc=torch.tensor([0.0, 0.0]).to("mlu"),
                    scale=torch.tensor([[1.0], [1.0]]).to("mlu"),
                ),
                (2, 2),
            ),
            (
                Gumbel(
                    loc=torch.tensor([0.0, 0.0]).to("mlu"),
                    scale=torch.tensor([[1.0]]).to("mlu"),
                ),
                (1, 2),
            ),
            (
                Gumbel(
                    loc=torch.tensor([0.0]).to("mlu"),
                    scale=torch.tensor([[1.0]]).to("mlu"),
                ),
                (1, 1),
            ),
            (Laplace(loc=torch.tensor([0.0, 0.0]).to("mlu"), scale=1), (2,)),
            (Laplace(loc=0, scale=torch.tensor([1.0, 1.0]).to("mlu")), (2,)),
            (
                Laplace(
                    loc=torch.tensor([0.0, 0.0]).to("mlu"),
                    scale=torch.tensor([1.0]).to("mlu"),
                ),
                (2,),
            ),
            (
                Laplace(
                    loc=torch.tensor([0.0, 0.0]).to("mlu"),
                    scale=torch.tensor([[1.0], [1.0]]).to("mlu"),
                ),
                (2, 2),
            ),
            (
                Laplace(
                    loc=torch.tensor([0.0, 0.0]).to("mlu"),
                    scale=torch.tensor([[1.0]]).to("mlu"),
                ),
                (1, 2),
            ),
            (
                Laplace(
                    loc=torch.tensor([0.0]).to("mlu"),
                    scale=torch.tensor([[1.0]]).to("mlu"),
                ),
                (1, 1),
            ),
            (Pareto(scale=torch.tensor([1.0, 1.0]).to("mlu"), alpha=1), (2,)),
            (Pareto(scale=1, alpha=torch.tensor([1.0, 1.0]).to("mlu")), (2,)),
            (
                Pareto(
                    scale=torch.tensor([1.0, 1.0]).to("mlu"),
                    alpha=torch.tensor([1.0]).to("mlu"),
                ),
                (2,),
            ),
            (
                Pareto(
                    scale=torch.tensor([1.0, 1.0]).to("mlu"),
                    alpha=torch.tensor([[1.0], [1.0]]).to("mlu"),
                ),
                (2, 2),
            ),
            (
                Pareto(
                    scale=torch.tensor([1.0, 1.0]).to("mlu"),
                    alpha=torch.tensor([[1.0]]).to("mlu"),
                ),
                (1, 2),
            ),
            (
                Pareto(
                    scale=torch.tensor([1.0]).to("mlu"),
                    alpha=torch.tensor([[1.0]]).to("mlu"),
                ),
                (1, 1),
            ),
        ]

        for dist, expected_size in valid_examples:
            actual_size = dist.sample().size()
            self.assertEqual(
                actual_size,
                expected_size,
                msg="{} actual size: {} != expected size: {}".format(
                    dist, actual_size, expected_size
                ),
            )

            sample_shape = torch.Size((2,))
            expected_size = sample_shape + expected_size
            actual_size = dist.sample(sample_shape).size()
            self.assertEqual(
                actual_size,
                expected_size,
                msg="{} actual size: {} != expected size: {}".format(
                    dist, actual_size, expected_size
                ),
            )

    def test_invalid_parameter_broadcasting(self):
        # invalid broadcasting cases; should throw error
        # example type (distribution class, distribution params)
        invalid_examples = [
            (
                Normal,
                {
                    "loc": torch.tensor([[0, 0]]).to("mlu"),
                    "scale": torch.tensor([1, 1, 1, 1]).to("mlu"),
                },
            ),
            (
                Normal,
                {
                    "loc": torch.tensor([[[0, 0, 0], [0, 0, 0]]]).to("mlu"),
                    "scale": torch.tensor([1, 1]).to("mlu"),
                },
            ),
            (
                Gumbel,
                {
                    "loc": torch.tensor([[0, 0]]).to("mlu"),
                    "scale": torch.tensor([1, 1, 1, 1]).to("mlu"),
                },
            ),
            (
                Gumbel,
                {
                    "loc": torch.tensor([[[0, 0, 0], [0, 0, 0]]]).to("mlu"),
                    "scale": torch.tensor([1, 1]).to("mlu"),
                },
            ),
            (
                Laplace,
                {
                    "loc": torch.tensor([0, 0]).to("mlu"),
                    "scale": torch.tensor([1, 1, 1]).to("mlu"),
                },
            ),
            (
                Pareto,
                {
                    "scale": torch.tensor([1, 1]).to("mlu"),
                    "alpha": torch.tensor([1, 1, 1]).to("mlu"),
                },
            ),
        ]

        for dist, kwargs in invalid_examples:
            self.assertRaises(RuntimeError, dist, **kwargs)


class TestDistributionShapes(TestCase):
    def setUp(self):
        super(TestDistributionShapes, self).setUp()
        self.scalar_sample = 1
        self.tensor_sample_1 = torch.ones(3, 2).to("mlu")
        self.tensor_sample_2 = torch.ones(3, 2, 3).to("mlu")

    def tearDown(self):
        super(TestDistributionShapes, self).tearDown()

    def test_entropy_shape(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(validate_args=False, **param)
                try:
                    actual_shape = dist.entropy().size()
                    expected_shape = (
                        dist.batch_shape if dist.batch_shape else torch.Size()
                    )
                    message = "{} example {}/{}, shape mismatch. expected {}, actual {}".format(
                        Dist.__name__, i + 1, len(params), expected_shape, actual_shape
                    )
                    self.assertEqual(actual_shape, expected_shape, msg=message)
                except NotImplementedError as e:
                    self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)

    def test_bernoulli_shape_scalar_params(self):
        bernoulli = Bernoulli(torch.tensor(0.3).to("mlu"))
        self.assertEqual(bernoulli._batch_shape, torch.Size())
        self.assertEqual(bernoulli._event_shape, torch.Size())
        self.assertEqual(bernoulli.sample().size(), torch.Size())
        self.assertEqual(bernoulli.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, bernoulli.log_prob, self.scalar_sample)
        self.assertEqual(
            bernoulli.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            bernoulli.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_bernoulli_shape_tensor_params(self):
        bernoulli = Bernoulli(
            torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]).to("mlu")
        )
        self.assertEqual(bernoulli._batch_shape, torch.Size((3, 2)))
        self.assertEqual(bernoulli._event_shape, torch.Size(()))
        self.assertEqual(bernoulli.sample().size(), torch.Size((3, 2)))
        self.assertEqual(bernoulli.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(
            bernoulli.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, bernoulli.log_prob, self.tensor_sample_2)
        self.assertEqual(
            bernoulli.log_prob(torch.ones(3, 1, 1).mlu()).size(), torch.Size((3, 3, 2))
        )

    def test_geometric_shape_scalar_params(self):
        geometric = Geometric(torch.tensor(0.3).to("mlu"))
        self.assertEqual(geometric._batch_shape, torch.Size())
        self.assertEqual(geometric._event_shape, torch.Size())
        self.assertEqual(geometric.sample().size(), torch.Size())
        self.assertEqual(geometric.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, geometric.log_prob, self.scalar_sample)
        self.assertEqual(
            geometric.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            geometric.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_geometric_shape_tensor_params(self):
        geometric = Geometric(
            torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]).to("mlu")
        )
        self.assertEqual(geometric._batch_shape, torch.Size((3, 2)))
        self.assertEqual(geometric._event_shape, torch.Size(()))
        self.assertEqual(geometric.sample().size(), torch.Size((3, 2)))
        self.assertEqual(geometric.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(
            geometric.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, geometric.log_prob, self.tensor_sample_2)
        self.assertEqual(
            geometric.log_prob(torch.ones(3, 1, 1).mlu()).size(), torch.Size((3, 3, 2))
        )

    def test_categorical_shape(self):
        # unbatched
        dist = Categorical(torch.tensor([0.6, 0.3, 0.1]).to("mlu"))
        self.assertEqual(dist._batch_shape, torch.Size(()))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size())
        self.assertEqual(
            dist.sample((3, 2)).size(),
            torch.Size(
                (
                    3,
                    2,
                )
            ),
        )
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )
        self.assertEqual(
            dist.log_prob(torch.ones(3, 1).mlu()).size(), torch.Size((3, 1))
        )
        # batched
        dist = Categorical(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]).mlu())
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size((3,)))
        self.assertEqual(
            dist.sample((3, 2)).size(),
            torch.Size(
                (
                    3,
                    2,
                    3,
                )
            ),
        )
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_1)
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )
        self.assertEqual(
            dist.log_prob(torch.ones(3, 1).mlu()).size(), torch.Size((3, 3))
        )

    def test_one_hot_categorical_shape(self):
        # unbatched
        dist = OneHotCategorical(torch.tensor([0.6, 0.3, 0.1]).mlu())
        self.assertEqual(dist._batch_shape, torch.Size(()))
        self.assertEqual(dist._event_shape, torch.Size((3,)))
        self.assertEqual(dist.sample().size(), torch.Size((3,)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_1)
        sample = torch.tensor([0.0, 1.0, 0.0]).to("mlu").expand(3, 2, 3)
        self.assertEqual(
            dist.log_prob(sample).size(),
            torch.Size(
                (
                    3,
                    2,
                )
            ),
        )
        self.assertEqual(
            dist.log_prob(dist.enumerate_support()).size(), torch.Size((3,))
        )
        sample = torch.eye(3).to("mlu")
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3,)))
        # batched
        dist = OneHotCategorical(
            torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]).mlu()
        )
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        sample = torch.tensor([0.0, 1.0]).to("mlu")
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3,)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        self.assertEqual(
            dist.log_prob(dist.enumerate_support()).size(), torch.Size((2, 3))
        )
        sample = torch.tensor([0.0, 1.0]).to("mlu").expand(3, 1, 2)
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3, 3)))

    def test_mixture_same_family_shape(self):
        dist = MixtureSameFamily(
            Categorical(torch.rand(5).to("mlu")),
            Normal(torch.randn(5).to("mlu"), torch.rand(5).to("mlu")),
        )
        self.assertEqual(dist._batch_shape, torch.Size())
        self.assertEqual(dist._event_shape, torch.Size())
        self.assertEqual(dist.sample().size(), torch.Size())
        self.assertEqual(dist.sample((5, 4)).size(), torch.Size((5, 4)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_vonmises_shape_tensor_params(self):
        von_mises = VonMises(
            torch.tensor([0.0, 0.0]).to("mlu"), torch.tensor([1.0, 1.0]).to("mlu")
        )
        self.assertEqual(von_mises._batch_shape, torch.Size((2,)))
        self.assertEqual(von_mises._event_shape, torch.Size(()))
        self.assertEqual(von_mises.sample().size(), torch.Size((2,)))
        self.assertEqual(
            von_mises.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2))
        )
        self.assertEqual(
            von_mises.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            von_mises.log_prob(torch.ones(2, 1).to("mlu")).size(), torch.Size((2, 2))
        )

    def test_normal_shape_tensor_params(self):
        normal = Normal(
            torch.tensor([0.0, 0.0]).to("mlu"), torch.tensor([1.0, 1.0]).to("mlu")
        )
        self.assertEqual(normal._batch_shape, torch.Size((2,)))
        self.assertEqual(normal._event_shape, torch.Size(()))
        self.assertEqual(normal.sample().size(), torch.Size((2,)))
        self.assertEqual(normal.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(
            normal.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, normal.log_prob, self.tensor_sample_2)
        self.assertEqual(
            normal.log_prob(torch.ones(2, 1).to("mlu")).size(), torch.Size((2, 2))
        )

    def test_uniform_shape_tensor_params(self):
        uniform = Uniform(
            torch.tensor([0.0, 0.0]).to("mlu"), torch.tensor([1.0, 1.0]).to("mlu")
        )
        self.assertEqual(uniform._batch_shape, torch.Size((2,)))
        self.assertEqual(uniform._event_shape, torch.Size(()))
        self.assertEqual(uniform.sample().size(), torch.Size((2,)))
        self.assertEqual(
            uniform.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2))
        )
        self.assertEqual(
            uniform.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, uniform.log_prob, self.tensor_sample_2)
        self.assertEqual(
            uniform.log_prob(torch.ones(2, 1).to("mlu")).size(), torch.Size((2, 2))
        )

    def test_pareto_shape_scalar_params(self):
        pareto = Pareto(torch.tensor(1.0).to("mlu"), torch.tensor(1.0).to("mlu"))
        self.assertEqual(pareto._batch_shape, torch.Size())
        self.assertEqual(pareto._event_shape, torch.Size())
        self.assertEqual(pareto.sample().size(), torch.Size())
        self.assertEqual(pareto.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(
            pareto.log_prob(self.tensor_sample_1 + 1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            pareto.log_prob(self.tensor_sample_2 + 1).size(), torch.Size((3, 2, 3))
        )

    def test_exponential_shape_tensor_param(self):
        expon = Exponential(torch.tensor([1.0, 1.0]).to("mlu"))
        self.assertEqual(expon._batch_shape, torch.Size((2,)))
        self.assertEqual(expon._event_shape, torch.Size(()))
        self.assertEqual(expon.sample().size(), torch.Size((2,)))
        self.assertEqual(expon.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(
            expon.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, expon.log_prob, self.tensor_sample_2)
        self.assertEqual(
            expon.log_prob(torch.ones(2, 2).to("mlu")).size(), torch.Size((2, 2))
        )

    def test_laplace_shape_tensor_params(self):
        laplace = Laplace(
            torch.tensor([0.0, 0.0]).to("mlu"), torch.tensor([1.0, 1.0]).to("mlu")
        )
        self.assertEqual(laplace._batch_shape, torch.Size((2,)))
        self.assertEqual(laplace._event_shape, torch.Size(()))
        self.assertEqual(laplace.sample().size(), torch.Size((2,)))
        self.assertEqual(laplace.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(
            laplace.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, laplace.log_prob, self.tensor_sample_2)
        self.assertEqual(
            laplace.log_prob(torch.ones(2, 1).to("mlu")).size(), torch.Size((2, 2))
        )

    def test_continuous_bernoulli_shape_tensor_params(self):
        continuous_bernoulli = ContinuousBernoulli(
            torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]).to("mlu")
        )
        self.assertEqual(continuous_bernoulli._batch_shape, torch.Size((3, 2)))
        self.assertEqual(continuous_bernoulli._event_shape, torch.Size(()))
        self.assertEqual(continuous_bernoulli.sample().size(), torch.Size((3, 2)))
        self.assertEqual(
            continuous_bernoulli.sample((3, 2)).size(), torch.Size((3, 2, 3, 2))
        )
        self.assertEqual(
            continuous_bernoulli.log_prob(self.tensor_sample_1).size(),
            torch.Size((3, 2)),
        )
        self.assertRaises(
            ValueError, continuous_bernoulli.log_prob, self.tensor_sample_2
        )
        self.assertEqual(
            continuous_bernoulli.log_prob(torch.ones(3, 1, 1).to("mlu")).size(),
            torch.Size((3, 3, 2)),
        )


class TestKL(TestCase):
    def setUp(self):
        super(TestKL, self).setUp()

        categorical = pairwise(
            Categorical,
            [[0.4, 0.3, 0.3], [0.2, 0.7, 0.1], [0.33, 0.33, 0.34], [0.2, 0.2, 0.6]],
        )
        exponential = pairwise(Exponential, [1.0, 2.5, 5.0, 10.0])
        gumbel = pairwise(Gumbel, [-2.0, 4.0, -3.0, 6.0], [1.0, 2.5, 1.0, 2.5])
        halfnormal = pairwise(HalfNormal, [1.0, 2.0, 1.0, 2.0])
        laplace = pairwise(Laplace, [-2.0, 4.0, -3.0, 6.0], [1.0, 2.5, 1.0, 2.5])
        lognormal = pairwise(LogNormal, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
        normal = pairwise(Normal, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
        independent = (Independent(normal[0], 1), Independent(normal[1], 1))
        onehotcategorical = pairwise(
            OneHotCategorical,
            [[0.4, 0.3, 0.3], [0.2, 0.7, 0.1], [0.33, 0.33, 0.34], [0.2, 0.2, 0.6]],
        )
        pareto = (
            Pareto(
                torch.tensor([2.5, 4.0, 2.5, 4.0]).to("mlu").expand(4, 4),
                torch.tensor([2.25, 3.75, 2.25, 3.75]).to("mlu").expand(4, 4),
            ),
            Pareto(
                torch.tensor([2.25, 3.75, 2.25, 3.8]).to("mlu").expand(4, 4),
                torch.tensor([2.25, 3.75, 2.25, 3.75]).to("mlu").expand(4, 4),
            ),
        )
        uniform_within_unit = pairwise(
            Uniform, [0.1, 0.9, 0.2, 0.75], [0.15, 0.95, 0.25, 0.8]
        )
        uniform_positive = pairwise(Uniform, [1, 1.5, 2, 4], [1.2, 2.0, 3, 7])
        uniform_real = pairwise(Uniform, [-2.0, -1, 0, 2], [-1.0, 1, 1, 4])
        uniform_pareto = pairwise(Uniform, [6.5, 7.5, 6.5, 8.5], [7.5, 8.5, 9.5, 9.5])
        continuous_bernoulli = pairwise(ContinuousBernoulli, [0.1, 0.2, 0.5, 0.9])

        # These tests should pass with precision = 0.01, but that makes tests very expensive.
        # Instead, we test with precision = 0.1 and only test with higher precision locally
        # when adding a new KL implementation.
        # The following pairs are not tested due to very high variance of the monte carlo
        # estimator; their implementations have been reviewed with extra care:
        # - (pareto, normal)
        self.precision = 0.1  # Set this to 0.01 when testing a new KL implementation.
        self.max_samples = int(1e07)  # Increase this when testing at smaller precision.
        self.samples_per_batch = int(1e04)
        self.finite_examples = [
            (categorical, categorical),
            (exponential, exponential),
            (exponential, gumbel),
            (exponential, normal),
            (gumbel, normal),
            (halfnormal, halfnormal),
            (independent, independent),
            (laplace, laplace),
            (lognormal, lognormal),
            (laplace, normal),
            (normal, gumbel),
            (normal, normal),
            (onehotcategorical, onehotcategorical),
            (pareto, pareto),
            (pareto, exponential),
            (uniform_positive, exponential),
            (uniform_real, gumbel),
            (uniform_real, normal),
            (uniform_pareto, pareto),
            (continuous_bernoulli, continuous_bernoulli),
            (continuous_bernoulli, exponential),
            (continuous_bernoulli, normal),
        ]

        self.infinite_examples = [
            (
                Categorical(torch.tensor([0.9, 0.1]).to("mlu")),
                Categorical(torch.tensor([1.0, 0.0]).to("mlu")),
            ),
            (
                Categorical(torch.tensor([[0.9, 0.1], [0.9, 0.1]]).to("mlu")),
                Categorical(torch.tensor([1.0, 0.0]).to("mlu")),
            ),
        ]

    def test_kl_monte_carlo(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for (p, _), (_, q) in self.finite_examples:
            actual = kl_divergence(p, q)
            numerator = 0
            denominator = 0
            while denominator < self.max_samples:
                x = p.sample(sample_shape=(self.samples_per_batch,))
                numerator += (p.log_prob(x) - q.log_prob(x)).sum(0)
                denominator += x.size(0)
                expected = numerator / denominator
                error = torch.abs(expected - actual) / (1 + expected)
                if error[~torch.isnan(error)].max() < self.precision:
                    break
            self.assertLess(
                error[~torch.isnan(error)].max(),
                self.precision,
                "\n".join(
                    [
                        "Incorrect KL({}, {}).".format(
                            type(p).__name__, type(q).__name__
                        ),
                        "Expected ({} Monte Carlo samples): {}".format(
                            denominator, expected
                        ),
                        "Actual (analytic): {}".format(actual),
                    ]
                ),
            )

    def test_kl_exponential_family(self):
        for (p, _), (_, q) in self.finite_examples:
            if type(p) == type(q) and issubclass(type(p), ExponentialFamily):
                actual = kl_divergence(p, q)
                expected = _kl_expfamily_expfamily(p, q)
                self.assertEqual(
                    actual,
                    expected,
                    msg="\n".join(
                        [
                            "Incorrect KL({}, {}).".format(
                                type(p).__name__, type(q).__name__
                            ),
                            "Expected (using Bregman Divergence) {}".format(expected),
                            "Actual (analytic) {}".format(actual),
                            "max error = {}".format(torch.abs(actual - expected).max()),
                        ]
                    ),
                )

    def test_kl_infinite(self):
        for p, q in self.infinite_examples:
            self.assertTrue(
                (kl_divergence(p, q) == inf).all(),
                "Incorrect KL({}, {})".format(type(p).__name__, type(q).__name__),
            )

    def test_kl_edgecases(self):
        self.assertEqual(
            kl_divergence(
                Categorical(torch.tensor([0.0, 1.0]).to("mlu")),
                Categorical(torch.tensor([0.0, 1.0]).to("mlu")),
            ),
            0,
        )

    def test_kl_shape(self):
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                # TODO(fwg): remove this when kl_divergence support (Gumbel,Gumbel)
                if isinstance(dist, Gumbel):
                    continue
                try:
                    kl = kl_divergence(dist, dist)
                except NotImplementedError as e:
                    self.assertNotRegex(
                        str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU, str(dist)
                    )
                    continue
                expected_shape = dist.batch_shape if dist.batch_shape else torch.Size()
                self.assertEqual(
                    kl.shape,
                    expected_shape,
                    msg="\n".join(
                        [
                            "{} example {}/{}".format(
                                Dist.__name__, i + 1, len(params)
                            ),
                            "Expected {}".format(expected_shape),
                            "Actual {}".format(kl.shape),
                        ]
                    ),
                )

    def test_kl_transformed(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/34859
        scale = torch.ones(2, 3).to("mlu")
        loc = torch.zeros(2, 3).to("mlu")
        normal = Normal(loc=loc, scale=scale)
        diag_normal = Independent(normal, reinterpreted_batch_ndims=1)
        trans_dist = TransformedDistribution(
            diag_normal, AffineTransform(loc=0.0, scale=2.0)
        )
        self.assertEqual(kl_divergence(diag_normal, diag_normal).shape, (2,))
        self.assertEqual(kl_divergence(trans_dist, trans_dist).shape, (2,))

    def test_entropy_monte_carlo(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    actual = dist.entropy()
                except NotImplementedError as e:
                    self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)
                    continue
                x = dist.sample(sample_shape=(60000,))
                expected = -dist.log_prob(x).mean(0)
                ignore = (expected == inf) | (expected == -inf)
                expected[ignore] = actual[ignore]
                self.assertEqual(
                    actual,
                    expected,
                    atol=0.2,
                    rtol=0,
                    msg="\n".join(
                        [
                            "{} example {}/{}, incorrect .entropy().".format(
                                Dist.__name__, i + 1, len(params)
                            ),
                            "Expected (monte carlo) {}".format(expected),
                            "Actual (analytic) {}".format(actual),
                            "max error = {}".format(torch.abs(actual - expected).max()),
                        ]
                    ),
                )


class TestConstraints(TestCase):
    def test_params_constraints(self):
        normalize_probs_dists = (
            Categorical,
            OneHotCategorical,
            OneHotCategoricalStraightThrough,
        )

        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                for name, value in param.items():
                    if isinstance(value, numbers.Number):
                        value = torch.tensor([value]).to("mlu")
                    if Dist in normalize_probs_dists and name == "probs":
                        # These distributions accept positive probs, but elsewhere we
                        # use a stricter constraint to the simplex.
                        value = value / value.sum(-1, True)
                    try:
                        constraint = dist.arg_constraints[name]
                    except KeyError:
                        continue  # ignore optional parameters

                    # Check param shape is compatible with distribution shape.
                    self.assertGreaterEqual(value.dim(), constraint.event_dim)
                    value_batch_shape = value.shape[
                        : value.dim() - constraint.event_dim
                    ]
                    torch.broadcast_shapes(dist.batch_shape, value_batch_shape)

                    if is_dependent(constraint):
                        continue

                    message = "{} example {}/{} parameter {} = {}".format(
                        Dist.__name__, i + 1, len(params), name, value
                    )
                    self.assertTrue(constraint.check(value).all(), msg=message)

    def test_support_constraints(self):
        for Dist, params in EXAMPLES:
            self.assertIsInstance(Dist.support, Constraint)
            for i, param in enumerate(params):
                dist = Dist(**param)
                value = dist.sample()
                constraint = dist.support
                message = "{} example {}/{} sample = {}".format(
                    Dist.__name__, i + 1, len(params), value
                )
                self.assertEqual(
                    constraint.event_dim, len(dist.event_shape), msg=message
                )
                ok = constraint.check(value)
                self.assertEqual(ok.shape, dist.batch_shape, msg=message)
                self.assertTrue(ok.all(), msg=message)


class TestNumericalStability(TestCase):
    def _test_pdf_score(
        self,
        dist_class,
        x,
        expected_value,
        probs=None,
        logits=None,
        expected_gradient=None,
        atol=1e-5,
    ):
        if probs is not None:
            p = probs.detach().requires_grad_()
            p_mlu = p.to("mlu")
            dist = dist_class(p_mlu)
        else:
            p = logits.detach().requires_grad_()
            p_mlu = p.to("mlu")
            dist = dist_class(logits=p_mlu)
        log_pdf = dist.log_prob(x.mlu())
        log_pdf.sum().backward()
        self.assertEqual(
            log_pdf,
            expected_value,
            atol=atol,
            rtol=0,
            msg="Incorrect value for tensor type: {}. Expected = {}, Actual = {}".format(
                type(x), expected_value, log_pdf
            ),
        )
        if expected_gradient is not None:
            self.assertEqual(
                p.grad,
                expected_gradient,
                atol=atol,
                rtol=0,
                msg="Incorrect gradient for tensor type: \
                                {}. Expected = {}, Actual = {}".format(
                    type(x), expected_gradient, p.grad
                ),
            )

    def test_bernoulli_gradient(self):
        for tensor_type in [torch.FloatTensor]:
            self._test_pdf_score(
                dist_class=Bernoulli,
                probs=tensor_type([0]),
                x=tensor_type([0]),
                expected_value=tensor_type([0]),
                expected_gradient=tensor_type([0]),
            )

            self._test_pdf_score(
                dist_class=Bernoulli,
                probs=tensor_type([0]),
                x=tensor_type([1]),
                expected_value=tensor_type(
                    [torch.finfo(tensor_type([]).dtype).eps]
                ).log(),
                expected_gradient=tensor_type([0]),
            )

            self._test_pdf_score(
                dist_class=Bernoulli,
                probs=tensor_type([1e-4]),
                x=tensor_type([1]),
                expected_value=tensor_type([math.log(1e-4)]),
                expected_gradient=tensor_type([10000]),
            )

            # Lower precision due to:
            # >>> 1 / (1 - torch.FloatTensor([0.9999]))
            # 9998.3408
            # [torch.FloatTensor of size 1]
            self._test_pdf_score(
                dist_class=Bernoulli,
                probs=tensor_type([1 - 1e-4]),
                x=tensor_type([0]),
                expected_value=tensor_type([math.log(1e-4)]),
                expected_gradient=tensor_type([-10000]),
                atol=2,
            )

            self._test_pdf_score(
                dist_class=Bernoulli,
                logits=tensor_type([math.log(9999)]),
                x=tensor_type([0]),
                expected_value=tensor_type([math.log(1e-4)]),
                expected_gradient=tensor_type([-1]),
                atol=1e-3,
            )

    def test_bernoulli_with_logits_underflow(self):
        for tensor_type, lim in [(torch.FloatTensor, -1e38)]:
            self._test_pdf_score(
                dist_class=Bernoulli,
                logits=tensor_type([lim]),
                x=tensor_type([0]),
                expected_value=tensor_type([0]),
                expected_gradient=tensor_type([0]),
            )

    def test_bernoulli_with_logits_overflow(self):
        for tensor_type, lim in [(torch.FloatTensor, 1e38)]:
            self._test_pdf_score(
                dist_class=Bernoulli,
                logits=tensor_type([lim]),
                x=tensor_type([1]),
                expected_value=tensor_type([0]),
                expected_gradient=tensor_type([0]),
            )

    def test_categorical_log_prob(self):
        for dtype in [torch.float]:
            p = torch.tensor([0, 1], dtype=dtype, requires_grad=True).to("mlu")
            categorical = OneHotCategorical(p)
            log_pdf = categorical.log_prob(torch.tensor([0, 1], dtype=dtype).to("mlu"))
            self.assertEqual(log_pdf.item(), 0)

    def test_categorical_log_prob_with_logits(self):
        for dtype in [torch.float]:
            p = torch.tensor([-inf, 0], dtype=dtype, requires_grad=True).to("mlu")
            categorical = OneHotCategorical(logits=p)
            log_pdf_prob_1 = categorical.log_prob(
                torch.tensor([0, 1], dtype=dtype).mlu()
            )
            self.assertEqual(log_pdf_prob_1.item(), 0)
            log_pdf_prob_0 = categorical.log_prob(
                torch.tensor([1, 0], dtype=dtype).mlu()
            )
            self.assertEqual(log_pdf_prob_0.item(), -inf)

    def test_continuous_bernoulli_gradient(self):
        def expec_val(x, probs=None, logits=None):
            assert not (probs is None and logits is None)
            if logits is not None:
                probs = 1.0 / (1.0 + math.exp(-logits))
            bern_log_lik = x * math.log(probs) + (1.0 - x) * math.log1p(-probs)
            if probs < 0.499 or probs > 0.501:  # using default values of lims here
                log_norm_const = (
                    math.log(math.fabs(math.atanh(1.0 - 2.0 * probs)))
                    - math.log(math.fabs(1.0 - 2.0 * probs))
                    + math.log(2.0)
                )
            else:
                aux = math.pow(probs - 0.5, 2)
                log_norm_const = math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * aux) * aux
            log_lik = bern_log_lik + log_norm_const
            return log_lik

        def expec_grad(x, probs=None, logits=None):
            assert not (probs is None and logits is None)
            if logits is not None:
                probs = 1.0 / (1.0 + math.exp(-logits))
            grad_bern_log_lik = x / probs - (1.0 - x) / (1.0 - probs)
            if probs < 0.499 or probs > 0.501:  # using default values of lims here
                grad_log_c = (
                    2.0 * probs
                    - 4.0 * (probs - 1.0) * probs * math.atanh(1.0 - 2.0 * probs)
                    - 1.0
                )
                grad_log_c /= (
                    2.0
                    * (probs - 1.0)
                    * probs
                    * (2.0 * probs - 1.0)
                    * math.atanh(1.0 - 2.0 * probs)
                )
            else:
                grad_log_c = 8.0 / 3.0 * (probs - 0.5) + 416.0 / 45.0 * math.pow(
                    probs - 0.5, 3
                )
            grad = grad_bern_log_lik + grad_log_c
            if logits is not None:
                grad *= 1.0 / (1.0 + math.exp(logits)) - 1.0 / math.pow(
                    1.0 + math.exp(logits), 2
                )
            return grad

        for tensor_type in [torch.FloatTensor]:
            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                probs=tensor_type([0.1]),
                x=tensor_type([0.1]),
                expected_value=tensor_type([expec_val(0.1, probs=0.1)]),
                expected_gradient=tensor_type([expec_grad(0.1, probs=0.1)]),
            )

            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                probs=tensor_type([0.1]),
                x=tensor_type([1.0]),
                expected_value=tensor_type([expec_val(1.0, probs=0.1)]),
                expected_gradient=tensor_type([expec_grad(1.0, probs=0.1)]),
            )

            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                probs=tensor_type([0.4999]),
                x=tensor_type([0.9]),
                expected_value=tensor_type([expec_val(0.9, probs=0.4999)]),
                expected_gradient=tensor_type([expec_grad(0.9, probs=0.4999)]),
            )

            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                probs=tensor_type([1e-4]),
                x=tensor_type([1]),
                expected_value=tensor_type([expec_val(1, probs=1e-4)]),
                expected_gradient=tensor_type(tensor_type([expec_grad(1, probs=1e-4)])),
                atol=1e-3,
            )

            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                probs=tensor_type([1 - 1e-4]),
                x=tensor_type([0.1]),
                expected_value=tensor_type([expec_val(0.1, probs=1 - 1e-4)]),
                expected_gradient=tensor_type([expec_grad(0.1, probs=1 - 1e-4)]),
                atol=2,
            )

            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                logits=tensor_type([math.log(9999)]),
                x=tensor_type([0]),
                expected_value=tensor_type([expec_val(0, logits=math.log(9999))]),
                expected_gradient=tensor_type([expec_grad(0, logits=math.log(9999))]),
                atol=1e-3,
            )

            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                logits=tensor_type([0.001]),
                x=tensor_type([0.5]),
                expected_value=tensor_type([expec_val(0.5, logits=0.001)]),
                expected_gradient=tensor_type([expec_grad(0.5, logits=0.001)]),
            )

    def test_continuous_bernoulli_with_logits_underflow(self):
        for tensor_type, lim, expected in [(torch.FloatTensor, -1e38, 2.76898)]:
            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                logits=tensor_type([lim]),
                x=tensor_type([0]),
                expected_value=tensor_type([expected]),
                expected_gradient=tensor_type([0.0]),
            )

    def test_continuous_bernoulli_with_logits_overflow(self):
        for tensor_type, lim, expected in [(torch.FloatTensor, 1e38, 2.76898)]:
            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                logits=tensor_type([lim]),
                x=tensor_type([1]),
                expected_value=tensor_type([expected]),
                expected_gradient=tensor_type([0.0]),
            )


# TODO: make this a pytest parameterized test
class TestLazyLogitsInitialization(TestCase):
    def setUp(self):
        super(TestLazyLogitsInitialization, self).setUp()
        # ContinuousBernoulli is not tested because log_prob is not computed simply
        # from 'logits', but 'probs' is also needed
        self.examples = [
            e for e in EXAMPLES if e.Dist in (Categorical, OneHotCategorical, Bernoulli)
        ]

    def test_lazy_logits_initialization(self):
        for Dist, params in self.examples:
            param = params[0].copy()
            if "probs" in param:
                probs = param.pop("probs")
                param["logits"] = probs_to_logits(probs)
                dist = Dist(**param)
                # Create new instance to generate a valid sample
                dist.log_prob(Dist(**param).sample())
                message = "Failed for {} example 0/{}".format(
                    Dist.__name__, len(params)
                )
                self.assertFalse("probs" in vars(dist), msg=message)
                try:
                    dist.enumerate_support()
                except NotImplementedError as e:
                    self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)
                self.assertFalse("probs" in vars(dist), msg=message)
                batch_shape, event_shape = dist.batch_shape, dist.event_shape
                self.assertFalse("probs" in vars(dist), msg=message)

    def test_lazy_probs_initialization(self):
        for Dist, params in self.examples:
            param = params[0].copy()
            if "probs" in param:
                dist = Dist(**param)
                dist.sample()
                message = "Failed for {} example 0/{}".format(
                    Dist.__name__, len(params)
                )
                self.assertFalse("logits" in vars(dist), msg=message)
                try:
                    dist.enumerate_support()
                except NotImplementedError as e:
                    self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)
                self.assertFalse("logits" in vars(dist), msg=message)
                self.assertFalse("logits" in vars(dist), msg=message)


@unittest.skipIf(not TEST_NUMPY, "NumPy not found")
class TestAgainstScipy(TestCase):
    def setUp(self):
        super(TestAgainstScipy, self).setUp()
        positive_var = torch.randn(20).to("mlu").exp()
        positive_var2 = torch.randn(20).to("mlu").exp()
        random_var = torch.randn(20).to("mlu")
        simplex_tensor = softmax(torch.randn(20).to("mlu"), dim=-1)
        self.distribution_pairs = [
            (Bernoulli(simplex_tensor), scipy.stats.bernoulli(simplex_tensor.cpu())),
            (
                Exponential(positive_var),
                scipy.stats.expon(scale=positive_var.cpu().reciprocal()),
            ),
            (Geometric(simplex_tensor), scipy.stats.geom(simplex_tensor.cpu(), loc=-1)),
            (
                Gumbel(random_var, positive_var2),
                scipy.stats.gumbel_r(random_var.cpu(), positive_var2.cpu()),
            ),
            (
                HalfNormal(positive_var2),
                scipy.stats.halfnorm(scale=positive_var2.cpu()),
            ),
            (
                Laplace(random_var, positive_var2),
                scipy.stats.laplace(random_var.cpu(), positive_var2.cpu()),
            ),
            (
                # Tests fail 1e-5 threshold if scale > 3
                LogNormal(random_var, positive_var.clamp(max=3)),
                scipy.stats.lognorm(
                    s=positive_var.cpu().clamp(max=3), scale=random_var.cpu().exp()
                ),
            ),
            (
                Normal(random_var, positive_var2),
                scipy.stats.norm(random_var.cpu(), positive_var2.cpu()),
            ),
            (
                OneHotCategorical(simplex_tensor),
                scipy.stats.multinomial(1, simplex_tensor.cpu()),
            ),
            (
                Pareto(positive_var, 2 + positive_var2),
                scipy.stats.pareto(2 + positive_var2.cpu(), scale=positive_var.cpu()),
            ),
            (
                Uniform(random_var, random_var + positive_var),
                scipy.stats.uniform(random_var.cpu(), positive_var.cpu()),
            ),
            (
                VonMises(random_var, positive_var),
                scipy.stats.vonmises(positive_var.cpu(), loc=random_var.cpu()),
            ),
        ]

    def test_mean(self):
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            if isinstance(pytorch_dist, (Cauchy, HalfCauchy)):
                # Cauchy, HalfCauchy distributions' mean is nan, skipping check
                continue
            elif isinstance(
                pytorch_dist, (LowRankMultivariateNormal, MultivariateNormal)
            ):
                self.assertEqual(pytorch_dist.mean, scipy_dist.mean, msg=pytorch_dist)
            else:
                self.assertEqual(
                    pytorch_dist.mean,
                    scipy_dist.mean().astype("float32"),
                    msg=pytorch_dist,
                )

    def test_variance_stddev(self):
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            if isinstance(pytorch_dist, (Cauchy, HalfCauchy, VonMises)):
                # Cauchy, HalfCauchy distributions' standard deviation is nan, skipping check
                # VonMises variance is circular and scipy doesn't produce a correct result
                continue
            elif isinstance(pytorch_dist, (Multinomial, OneHotCategorical)):
                self.assertEqual(
                    pytorch_dist.variance,
                    np.diag(scipy_dist.cov().astype("float32")),
                    msg=pytorch_dist,
                )
                self.assertEqual(
                    pytorch_dist.stddev,
                    np.diag(scipy_dist.cov().astype("float32")) ** 0.5,
                    msg=pytorch_dist,
                )
            elif isinstance(
                pytorch_dist, (LowRankMultivariateNormal, MultivariateNormal)
            ):
                self.assertEqual(
                    pytorch_dist.variance, np.diag(scipy_dist.cov), msg=pytorch_dist
                )
                self.assertEqual(
                    pytorch_dist.stddev,
                    np.diag(scipy_dist.cov) ** 0.5,
                    msg=pytorch_dist,
                )
            else:
                self.assertEqual(
                    pytorch_dist.variance,
                    scipy_dist.var().astype("float32"),
                    msg=pytorch_dist,
                )
                self.assertEqual(
                    pytorch_dist.stddev,
                    scipy_dist.var().astype("float32") ** 0.5,
                    msg=pytorch_dist,
                )

    def test_cdf(self):
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            samples = pytorch_dist.sample((5,))
            try:
                cdf = pytorch_dist.cdf(samples)
            except NotImplementedError as e:
                self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)
                continue
            self.assertEqual(
                cdf, scipy_dist.cdf(samples.cpu()).astype("float32"), msg=pytorch_dist
            )

    def test_icdf(self):
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            samples = torch.rand((5,) + pytorch_dist.batch_shape)
            try:
                icdf = pytorch_dist.icdf(samples.mlu())
            except NotImplementedError as e:
                self.assertNotRegex(str(e), ERR_MSG_OP_NOT_IMPLEMENTED_ON_MLU)
                continue
            self.assertEqual(
                icdf,
                scipy_dist.ppf(samples).astype("float32"),
                atol=1e-4,
                rtol=0,
                msg=pytorch_dist,
            )


class TestFunctors(TestCase):
    def test_cat_transform(self):
        x1 = -1 * torch.arange(1, 101, dtype=torch.float).mlu().view(-1, 100)
        x2 = (torch.arange(1, 101, dtype=torch.float).mlu().view(-1, 100) - 1) / 100
        x3 = torch.arange(1, 101, dtype=torch.float).mlu().view(-1, 100)
        t1, t2, t3 = ExpTransform(), AffineTransform(1, 100), identity_transform
        dim = 0
        x = torch.cat([x1, x2, x3], dim=dim)
        t = CatTransform([t1, t2, t3], dim=dim)
        actual_dom_check = t.domain.check(x)
        expected_dom_check = torch.cat(
            [t1.domain.check(x1), t2.domain.check(x2), t3.domain.check(x3)], dim=dim
        )
        self.assertEqual(expected_dom_check, actual_dom_check)
        actual = t(x)
        expected = torch.cat([t1(x1), t2(x2), t3(x3)], dim=dim)
        self.assertEqual(expected, actual)
        y1 = torch.arange(1, 101, dtype=torch.float).mlu().view(-1, 100)
        y2 = torch.arange(1, 101, dtype=torch.float).mlu().view(-1, 100)
        y3 = torch.arange(1, 101, dtype=torch.float).mlu().view(-1, 100)
        y = torch.cat([y1, y2, y3], dim=dim)
        actual_cod_check = t.codomain.check(y)
        expected_cod_check = torch.cat(
            [t1.codomain.check(y1), t2.codomain.check(y2), t3.codomain.check(y3)],
            dim=dim,
        )
        self.assertEqual(actual_cod_check, expected_cod_check)
        actual_inv = t.inv(y)
        expected_inv = torch.cat([t1.inv(y1), t2.inv(y2), t3.inv(y3)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = torch.cat(
            [
                t1.log_abs_det_jacobian(x1, y1),
                t2.log_abs_det_jacobian(x2, y2),
                t3.log_abs_det_jacobian(x3, y3),
            ],
            dim=dim,
        )
        self.assertEqual(actual_jac, expected_jac)

    def test_cat_transform_non_uniform(self):
        x1 = -1 * torch.arange(1, 101, dtype=torch.float).mlu().view(-1, 100)
        x2 = torch.cat(
            [
                (torch.arange(1, 101, dtype=torch.float).mlu().view(-1, 100) - 1) / 100,
                torch.arange(1, 101, dtype=torch.float).mlu().view(-1, 100),
            ]
        )
        t1 = ExpTransform()
        t2 = CatTransform([AffineTransform(1, 100), identity_transform], dim=0)
        dim = 0
        x = torch.cat([x1, x2], dim=dim)
        t = CatTransform([t1, t2], dim=dim, lengths=[1, 2])
        actual_dom_check = t.domain.check(x)
        expected_dom_check = torch.cat(
            [t1.domain.check(x1), t2.domain.check(x2)], dim=dim
        )
        self.assertEqual(expected_dom_check, actual_dom_check)
        actual = t(x)
        expected = torch.cat([t1(x1), t2(x2)], dim=dim)
        self.assertEqual(expected, actual)
        y1 = torch.arange(1, 101, dtype=torch.float).mlu().view(-1, 100)
        y2 = torch.cat(
            [
                torch.arange(1, 101, dtype=torch.float).mlu().view(-1, 100),
                torch.arange(1, 101, dtype=torch.float).mlu().view(-1, 100),
            ]
        )
        y = torch.cat([y1, y2], dim=dim)
        actual_cod_check = t.codomain.check(y)
        expected_cod_check = torch.cat(
            [t1.codomain.check(y1), t2.codomain.check(y2)], dim=dim
        )
        self.assertEqual(actual_cod_check, expected_cod_check)
        actual_inv = t.inv(y)
        expected_inv = torch.cat([t1.inv(y1), t2.inv(y2)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = torch.cat(
            [t1.log_abs_det_jacobian(x1, y1), t2.log_abs_det_jacobian(x2, y2)], dim=dim
        )
        self.assertEqual(actual_jac, expected_jac)

    def test_cat_event_dim(self):
        t1 = AffineTransform(0, 2 * torch.ones(2).mlu(), event_dim=1)
        t2 = AffineTransform(0, 2 * torch.ones(2).mlu(), event_dim=1)
        dim = 1
        bs = 16
        x1 = torch.randn(bs, 2).mlu()
        x2 = torch.randn(bs, 2).mlu()
        x = torch.cat([x1, x2], dim=1)
        t = CatTransform([t1, t2], dim=dim, lengths=[2, 2])
        y1 = t1(x1)
        y2 = t2(x2)
        y = t(x)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = sum(
            [t1.log_abs_det_jacobian(x1, y1), t2.log_abs_det_jacobian(x2, y2)]
        )

    def test_stack_transform(self):
        x1 = -1 * torch.arange(1, 101, dtype=torch.float).mlu()
        x2 = (torch.arange(1, 101, dtype=torch.float).mlu() - 1) / 100
        x3 = torch.arange(1, 101, dtype=torch.float).mlu()
        t1, t2, t3 = ExpTransform(), AffineTransform(1, 100), identity_transform
        dim = 0
        x = torch.stack([x1, x2, x3], dim=dim)
        t = StackTransform([t1, t2, t3], dim=dim)
        actual_dom_check = t.domain.check(x)
        expected_dom_check = torch.stack(
            [t1.domain.check(x1), t2.domain.check(x2), t3.domain.check(x3)], dim=dim
        )
        self.assertEqual(expected_dom_check, actual_dom_check)
        actual = t(x)
        expected = torch.stack([t1(x1), t2(x2), t3(x3)], dim=dim)
        self.assertEqual(expected, actual)
        y1 = torch.arange(1, 101, dtype=torch.float).mlu()
        y2 = torch.arange(1, 101, dtype=torch.float).mlu()
        y3 = torch.arange(1, 101, dtype=torch.float).mlu()
        y = torch.stack([y1, y2, y3], dim=dim)
        actual_cod_check = t.codomain.check(y)
        expected_cod_check = torch.stack(
            [t1.codomain.check(y1), t2.codomain.check(y2), t3.codomain.check(y3)],
            dim=dim,
        )
        self.assertEqual(actual_cod_check, expected_cod_check)
        actual_inv = t.inv(x)
        expected_inv = torch.stack([t1.inv(x1), t2.inv(x2), t3.inv(x3)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = torch.stack(
            [
                t1.log_abs_det_jacobian(x1, y1),
                t2.log_abs_det_jacobian(x2, y2),
                t3.log_abs_det_jacobian(x3, y3),
            ],
            dim=dim,
        )
        self.assertEqual(actual_jac, expected_jac)


class TestValidation(TestCase):
    def setUp(self):
        super(TestCase, self).setUp()

    def test_valid(self):
        for Dist, params in EXAMPLES:
            for param in params:
                Dist(validate_args=True, **param)

    def test_invalid_log_probs_arg(self):
        # Check that validation errors are indeed disabled,
        # but they might raise another error
        for Dist, params in EXAMPLES:
            if Dist == TransformedDistribution:
                # TransformedDistribution has a distribution instance
                # as the argument, so we cannot do much about that
                continue
            for param in params:
                d_nonval = Dist(validate_args=False, **param)
                d_val = Dist(validate_args=True, **param)
                for v in torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).mlu():
                    # samples with incorrect shape must throw ValueError only
                    try:
                        log_prob = d_val.log_prob(v)
                    except ValueError:
                        pass

                    # get sample of correct shape
                    val = torch.full(d_val.batch_shape + d_val.event_shape, v)
                    # check samples with incorrect support
                    try:
                        log_prob = d_val.log_prob(val.mlu())
                    except ValueError as e:
                        if e.args and "must be within the support" in e.args[0]:
                            try:
                                log_prob = d_nonval.log_prob(val)
                            except RuntimeError:
                                pass

    @unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    def test_invalid(self):
        for Dist, params in BAD_EXAMPLES:
            for i, param in enumerate(params):
                try:
                    with self.assertRaises(ValueError):
                        Dist(validate_args=True, **param)
                except AssertionError as e:
                    fail_string = "ValueError not raised for {} example {}/{}"
                    raise AssertionError(
                        fail_string.format(Dist.__name__, i + 1, len(params))
                    ) from e


if __name__ == "__main__" and torch._C.has_lapack:
    run_tests()
