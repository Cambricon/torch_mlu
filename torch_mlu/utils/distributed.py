import sys
import torch
import torch.distributed as dist
from torch.distributed import Reducer
import torch_mlu


def _ddp_init_helper(
    self,
    parameters,
    expect_sparse_gradient,
    param_to_name_mapping,
    static_graph,
):
    """
    DDP init helper function to manage parameters, grad hooks, logging, and SyncBatchNorm.

    Initialization helper function that does the following:
    (1) bucketing the parameters for reductions
    (2) resetting the bucketing states
    (3) registering the grad hooks
    (4) Logging construction-time DDP logging data
    (5) passing a handle of DDP to SyncBatchNorm Layer
    """
    # Notice, the parameters order is not in the order in which they are used,
    # especially in models with control flow.
    #
    # Alongside parameters are not presented in the real execution order,
    # if a certain model happens to also
    #   1) have other collectives comm ops in its backward graph.
    #   2) have unused parameter in subset ranks of the whole world.
    # bucketing could insert ALL-REDUCE comm op too early on the rank with unused parameter,
    # matching up with other collectives comm ops on other ranks unexpectedly.
    #
    # In order to handle this corner case, when the parameters are not in the real execution order,
    # we don't do bucketing, thus only one ALL-REDUCE is inserted after all the gradients
    # of the whole graph are computed.
    #
    # Notice, here we only disable bucketing for the first iteration.
    # After the first iteration, it's OK to rebuild buckets,
    # because "bucket rebuild" bucketizes parameters based on its real
    # execution order in backward graph.

    # Can remove this branching once #73732 is landed.
    if static_graph is True or self.find_unused_parameters is False:
        bucket_size_limits = [sys.maxsize]
    else:
        bucket_size_limits = [
            dist._DEFAULT_FIRST_BUCKET_BYTES,
            self.bucket_bytes_cap,
        ]
    (
        bucket_indices,
        per_bucket_size_limits,
    ) = dist._compute_bucket_assignment_by_size(
        parameters,
        bucket_size_limits,
        expect_sparse_gradient,
    )

    # Remember index for parameters if we are in mixed precision, as we
    # need to pass in index to Reducer's autograd hook via python.
    if self.mixed_precision is not None:
        for i, p in enumerate(parameters):
            p._idx = i

    # Note: reverse list of buckets because we want to approximate the
    # order in which their gradients are produced, and assume they
    # are used in the forward pass in the order they are defined.
    self.reducer = torch_mlu._MLUC._c10d_mlu.Reducer(
        parameters,
        list(reversed(bucket_indices)),
        list(reversed(per_bucket_size_limits)),
        self.process_group,
        expect_sparse_gradient,
        # The bucket size limit is specified in the constructor.
        # Additionally, we allow for a single small bucket for parameters
        # that are defined first, such that their gradients don't spill into
        # a much larger bucket, adding unnecessary latency after gradient
        # computation finishes. Experiments showed 1MB is a reasonable value.
        self.bucket_bytes_cap,
        self.find_unused_parameters,
        self.gradient_as_bucket_view,
        param_to_name_mapping,
        # User can set dist._DEFAULT_FIRST_BUCKET_BYTES to tune DDP first
        # bucket.
        dist._DEFAULT_FIRST_BUCKET_BYTES,
    )

    self.logger = torch_mlu._MLUC._c10d_mlu.Logger(self.reducer)
    # Set as a weak reference to avoid reference cycle between
    # logger and reducer.
    self.reducer.set_logger(self.logger)

    has_sync_bn = False
    for submodule in self.module.modules():
        if isinstance(submodule, torch.nn.SyncBatchNorm):
            has_sync_bn = True
            break

    # Set logging data that can be got during construction time.
    self.logger.set_construction_data_and_log(
        self.module.__class__.__name__,
        [] if self.device_ids is None else self.device_ids,
        -1 if self.output_device is None else self.output_device,
        self.broadcast_buffers,
        has_sync_bn,
        static_graph,
    )

    # passing a handle to torch.nn.SyncBatchNorm layer
    self._passing_sync_batchnorm_handle(self.module)


def apply_ddp_patch():
    torch.nn.parallel.DistributedDataParallel._ddp_init_helper = _ddp_init_helper
    torch.distributed._register_comm_hook = (
        torch_mlu._MLUC._c10d_mlu._register_comm_hook
    )
    torch.distributed._register_builtin_comm_hook = (
        torch_mlu._MLUC._c10d_mlu._register_builtin_comm_hook
    )
