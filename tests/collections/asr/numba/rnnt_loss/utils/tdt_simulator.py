# Numba CUDA simulator harness for the TDT kernels, shared by the CPU tests of
# the fp16 table and of the chunked long-clip path. Made with Claude Code.
#
# NUMBA_ENABLE_CUDASIM must be set before numba.cuda is first imported, so a
# test runs its simulated part in a subprocess (run_in_simulator) that calls
# install() before touching the kernels.

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# tests/collections/asr/numba/rnnt_loss/utils/<this file>
REPO_ROOT = Path(__file__).resolve().parents[6]

DeviceBuffer = None  # set by install()


def run_in_simulator(script, params, tmp_path):
    """Run `script` with json(params + out) as argv[1] under the simulator; returns the npz it saved."""
    out = tmp_path / "sim.npz"
    env = dict(
        os.environ,
        NUMBA_ENABLE_CUDASIM="1",
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]),
    )
    args = json.dumps(dict(params, out=str(out)))
    proc = subprocess.run([sys.executable, str(script), args], env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return dict(np.load(out))


def install():
    """Patch what the simulator lacks. Call once, in the simulator process."""
    global DeviceBuffer
    from numba import cuda
    from numba.cuda.simulator.cudadrv.devicearray import FakeCUDAArray

    from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import gpu_rnnt

    # The simulator has no __cuda_array_interface__ bridge: hand it host arrays.
    cuda.as_cuda_array = lambda x, *args, **kwargs: x

    # The simulator cannot run the shuffle-based log-softmax reduction, so the
    # denominators are computed on the host (the GPU tests cover that kernel).
    def host_log_softmax(self, acts, denom):
        rows = np.asarray(acts, dtype=np.float64).reshape(-1, self.alphabet_size_)
        top = rows.max(axis=1)
        denom[:] = -(top + np.log(np.exp(rows - top[:, None]).sum(axis=1)))

    gpu_rnnt.GPURNNT.log_softmax = host_log_softmax

    class _DeviceBuffer(FakeCUDAArray):
        # A host array is copied in and out per argument, so passing one twice
        # would not alias; one fake device array passed twice does. GPUTDT
        # tells "in place" from the data pointer, as on torch tensors.
        def data_ptr(self):
            return self._ary.__array_interface__["data"][0]

        def __imul__(self, other):
            self._ary *= other
            return self

    DeviceBuffer = _DeviceBuffer


def tdt_loss_gpu(
    label_acts,
    duration_acts,
    labels,
    input_lengths,
    label_lengths,
    costs,
    label_grads,
    duration_grads,
    blank_label,
    durations,
    fastemit_lambda,
    clamp,
    num_threads,
    sigma,
    omega,
    log_probs_input=False,
):
    """rnnt.tdt_loss_gpu on CPU torch tensors in the simulator: same arguments, no stream or device.

    Results land in costs / label_grads / duration_grads through numpy views.
    """
    from numba import cuda

    from nemo.collections.asr.parts.numba.rnnt_loss.utils import rnnt_helper
    from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import gpu_rnnt

    B, T, U, V = label_acts.shape
    size, _ = rnnt_helper.get_workspace_size(T, U, B, gpu=True)
    wrapper = gpu_rnnt.GPUTDT(
        minibatch=B,
        maxT=T,
        maxU=U,
        alphabet_size=V,
        workspace=np.zeros(size, dtype=np.float32),
        tdt_workspace=np.array(durations, dtype=np.int64),
        num_durations=len(durations),
        blank=blank_label,
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        num_threads=1,
        stream=cuda.default_stream(),
        sigma=sigma,
        omega=omega,
        log_probs_input=log_probs_input,
    )

    def flat(x):
        return x.detach().numpy().reshape(-1)

    common = dict(
        costs=costs.numpy(),
        pad_labels=labels.numpy(),
        label_lengths=label_lengths.numpy(),
        input_lengths=input_lengths.numpy(),
    )
    acts = DeviceBuffer(flat(label_acts))
    if label_grads is None:
        wrapper.score_forward(label_acts=acts, duration_acts=flat(duration_acts), **common)
    else:
        wrapper.cost_and_grad(
            label_acts=acts,
            duration_acts=flat(duration_acts),
            label_grads=DeviceBuffer(flat(label_grads)),
            duration_grads=flat(duration_grads),
            **common,
        )
    return True
