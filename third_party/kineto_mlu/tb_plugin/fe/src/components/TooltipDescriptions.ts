/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

export const StepTimeBreakDownTooltip = `The time spent on each step is broken down into multiple categories as follows:
Kernel: Kernels execution time on GPU/MLU device;
Memcpy: GPU/MLU involved memory copy time (either D2D, D2H or H2D);
Memset: GPU/MLU involved memory set time;
Runtime: CUDA/MLU runtime execution time on host side; Such as cudaLaunchKernel, cudaMemcpyAsync, cudaStreamSynchronize, cnInvokeKernel, cnrtMemcpyAsync...
DataLoader: The data loading time spent in PyTorch DataLoader object;
CPU Exec: Host compute time, including every PyTorch operator running time;
Other: The time not included in any of the above.`

export const DeviceSelfTimeTooltip = `The accumulated time spent on Device, not including this operator’s child operators.`

export const DeviceTotalTimeTooltip = `The accumulated time spent on Device, including this operator’s child operators.`

export const HostSelfTimeTooltip = `The accumulated time spent on Host, not including this operator’s child operators.`

export const HostTotalTimeTooltip = `The accumulated time spent on Host, including this operator’s child operators.`

export const GPUKernelTotalTimeTooltip = `The accumulated time of all calls of this kernel.`

export const TensorCoresPieChartTooltip = `The accumulated time of all kernels using or not using Tensor Cores.`

export const DistributedGpuInfoTableTooltip = `Information about Device hardware used during the run.`

export const DistributedOverlapGraphTooltip = `The time spent on computation vs communication.`

export const DistributedWaittimeGraphTooltip = `The time spent waiting vs communicating between devices.`

export const DistributedCommopsTableTooltip = `Statistics for operations managing communications between nodes.`
