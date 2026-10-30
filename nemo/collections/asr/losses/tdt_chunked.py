# TDT joint + loss over a compact lattice, for clips too long for the dense
# [B, T, U+1, V] table. Made with Claude Code.
#
# The dense path materialises joint logits for every (t, u) cell: a 105 s clip
# is T = 1313 frames by U+1 ~ 700, i.e. ~7 G cells times V = 8198, tens of GB.
# This path never holds more than one T-chunk of logits.
#
# Why two slots are enough. In compute_tdt_grad_kernel, the gradient w.r.t.
# the logit of token v at node (t, u) is
#
#     exp(alpha + beta + logp(v) - ll) - transitions(v)
#
# where only blank (duration > 0) and label[u] (any duration) have transition
# terms (FastEmit off). The first term is softmax(v) * post(t, u), with
# post = exp(alpha + beta - ll) the node posterior: sigma and the duration
# log-probs enter alpha, beta and ll only. So the alpha/beta/grad kernels need
# no more than two log-probs per node, both normalised over the full V:
#
#     compact[t, u, 0] = logp(label[u])    compact[t, u, 1] = logp(blank)
#
# run with labels remapped to 0, blank index 1, alphabet 2 and the log-softmax
# skipped (log_probs_input). The kernel then returns, for those two columns,
# exactly the full-V gradient of the dense run (its softmax(v) * post term uses
# the full-V logp held in the slot), and every other column is
# softmax(v) * post. The node posterior needs no workspace access either: the
# kernel's duration gradients are minus the posterior mass of each outgoing
# transition (w.r.t. the duration log-probs), and every lattice path leaves a
# node it visits exactly once, so post = -sum_d duration_grad[t, u, d].
#
# Backward recomputes each chunk's logits (replaying the joint's dropout mask
# from the RNG state saved at forward, as torch.utils.checkpoint does), builds
# the logit gradient from the saved kernel outputs, and backprops it through
# joint_net into f, g and the joint weights.

from typing import List, Sequence

import torch
from torch.utils.checkpoint import get_device_states, set_device_states

from nemo.collections.asr.parts.numba.rnnt_loss import rnnt as numba_rnnt

# Compact table layout.
LABEL_SLOT, BLANK_SLOT = 0, 1


def _chunk_logits(joint_net: torch.nn.Module, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """joint_net(f + g) for a T-chunk f [B, Tc, H] and g [B, U+1, H], upcast to fp32 like RNNTLoss does."""
    return joint_net(f.unsqueeze(2) + g.unsqueeze(1)).float()


def _label_index(labels: torch.Tensor, label_lens: torch.Tensor, num_nodes: int, blank: int) -> torch.Tensor:
    """[B, U+1] vocabulary index of label[u], blank where there is none (u >= label length)."""
    idx = torch.full((labels.shape[0], num_nodes), blank, dtype=torch.long, device=labels.device)
    u = torch.arange(num_nodes - 1, device=labels.device)
    idx[:, :-1] = torch.where(u < label_lens[:, None], labels[:, : num_nodes - 1], blank)
    return idx


class ChunkedTDTJointLoss(torch.autograd.Function):
    """Per-sample TDT loss of joint_net(relu(f + g)) without the dense logit table.

    Same objective and gradients as the dense numba TDT loss with omega 0 and
    FastEmit off, given f, g already projected (and g masked) as
    RNNTJoint.joint_after_projection receives them. Returns costs [B] (fp32),
    like TDTLossNumba with reduction None. Call through ``chunked_tdt_joint_loss``.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        f: torch.Tensor,
        g: torch.Tensor,
        labels: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
        joint_net: torch.nn.Module,
        blank: int,
        durations: List[int],
        sigma: float,
        clamp: float,
        chunk_frames: int,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = f.shape
        num_nodes = g.shape[1]  # U + 1
        num_durations = len(durations)
        idx = _label_index(labels, label_lens, num_nodes, blank)

        # Dropout in joint_net draws from these; backward replays them.
        cpu_rng = torch.get_rng_state()
        devices, device_rng = get_device_states(f, g)

        compact = torch.empty(B, T, num_nodes, 2, dtype=torch.float32, device=f.device)
        duration_logprobs = torch.empty(B, T, num_nodes, num_durations, dtype=torch.float32, device=f.device)
        for t0 in range(0, T, chunk_frames):
            t1 = min(t0 + chunk_frames, T)
            logits = _chunk_logits(joint_net, f[:, t0:t1], g)
            tokens = logits[..., :-num_durations]
            denom = torch.logsumexp(tokens, dim=-1)
            gather = idx[:, None, :, None].expand(B, t1 - t0, num_nodes, 1)
            compact[:, t0:t1, :, LABEL_SLOT] = tokens.gather(-1, gather).squeeze(-1) - denom
            compact[:, t0:t1, :, BLANK_SLOT] = tokens[..., blank] - denom
            duration_logprobs[:, t0:t1] = torch.log_softmax(logits[..., -num_durations:], dim=-1)
            del logits, tokens, denom

        training = any(ctx.needs_input_grad)
        costs = torch.zeros(B, dtype=torch.float32, device=f.device)
        label_grads = torch.zeros_like(compact) if training else None
        duration_grads = torch.zeros_like(duration_logprobs) if training else None
        numba_rnnt.tdt_loss_gpu(
            compact,
            duration_logprobs,
            labels=torch.zeros(B, num_nodes - 1, dtype=torch.long, device=f.device),  # every label is slot 0
            input_lengths=act_lens,
            label_lengths=label_lens,
            costs=costs,
            label_grads=label_grads,
            duration_grads=duration_grads,
            blank_label=BLANK_SLOT,
            durations=list(durations),
            fastemit_lambda=0.0,
            clamp=clamp,
            num_threads=0,
            sigma=sigma,
            omega=0.0,
            log_probs_input=True,
        )

        if training:
            ctx.save_for_backward(f, g, idx, label_grads, duration_grads)
            ctx.joint_net, ctx.blank, ctx.clamp, ctx.chunk_frames = joint_net, blank, clamp, chunk_frames
            ctx.num_durations = num_durations
            ctx.rng = (cpu_rng, devices, device_rng)
        return costs

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_costs: torch.Tensor):
        f, g, idx, label_grads, duration_grads = ctx.saved_tensors
        joint_net, blank, clamp, num_durations = ctx.joint_net, ctx.blank, ctx.clamp, ctx.num_durations
        params = list(joint_net.parameters())
        need_f, need_g = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        need_params = ctx.needs_input_grad[11:]
        B, T, _ = f.shape
        num_nodes = g.shape[1]
        scale = grad_costs.float().view(B, 1, 1, 1)

        grad_f = torch.zeros_like(f) if need_f else None
        grad_g = torch.zeros(g.shape, dtype=torch.float32, device=g.device) if need_g else None
        grad_params = [torch.zeros_like(p) if need else None for p, need in zip(params, need_params)]

        cpu_rng, devices, device_rng = ctx.rng
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            # Chunks are recomputed in forward order from the forward's starting
            # state, so each one draws the dropout mask it drew then.
            torch.set_rng_state(cpu_rng)
            set_device_states(devices, device_rng)
            g_in = g.detach().requires_grad_(need_g)
            for t0 in range(0, T, ctx.chunk_frames):
                t1 = min(t0 + ctx.chunk_frames, T)
                f_in = f[:, t0:t1].detach().requires_grad_(need_f)
                logits = _chunk_logits(joint_net, f_in, g_in)
                with torch.no_grad():
                    grad_logits = _logit_grads(
                        logits, idx, label_grads[:, t0:t1], duration_grads[:, t0:t1], blank, num_durations, clamp
                    ).mul_(scale)
                wanted = [(i, x) for i, x in enumerate([f_in, g_in] + params) if x.requires_grad]
                grads = torch.autograd.grad(logits, [x for _, x in wanted], grad_logits, allow_unused=True)
                del logits, grad_logits
                for (i, _), grad in zip(wanted, grads):
                    if grad is None:
                        continue
                    if i == 0:
                        grad_f[:, t0:t1] = grad
                    elif i == 1:
                        grad_g += grad
                    elif grad_params[i - 2] is not None:
                        grad_params[i - 2] += grad

        if grad_g is not None:
            grad_g = grad_g.to(g.dtype)
        return (grad_f, grad_g) + (None,) * 9 + tuple(grad_params)


def _logit_grads(
    logits: torch.Tensor,
    idx: torch.Tensor,
    label_grads: torch.Tensor,
    duration_grads: torch.Tensor,
    blank: int,
    num_durations: int,
    clamp: float,
) -> torch.Tensor:
    """d(cost)/d(logits) of one chunk [B, Tc, U+1, V+D], from the compact kernel's outputs."""
    B, Tc, num_nodes, _ = logits.shape
    post = -duration_grads.sum(dim=-1, keepdim=True)  # node posterior, 0 off the lattice
    tokens = torch.softmax(logits[..., :-num_durations], dim=-1).mul_(post)
    if clamp > 0.0:
        tokens.clamp_(-clamp, clamp)  # the dense kernel clamps every column
    # The kernel's two columns are the full-V gradients (transitions included).
    # Label first: past the last label idx is blank, whose slot-0 value is only
    # softmax * post, and the blank slot must win there.
    tokens.scatter_(-1, idx[:, None, :, None].expand(B, Tc, num_nodes, 1), label_grads[..., LABEL_SLOT:LABEL_SLOT + 1])
    tokens[..., blank] = label_grads[..., BLANK_SLOT]
    # Duration grads are w.r.t. the log-probs: chain through the log-softmax.
    durations = torch.softmax(logits[..., -num_durations:], dim=-1)
    durations = duration_grads - durations * duration_grads.sum(dim=-1, keepdim=True)
    return torch.cat([tokens, durations], dim=-1)


def chunked_tdt_joint_loss(
    f: torch.Tensor,
    g: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    joint_net: torch.nn.Module,
    blank: int,
    durations: Sequence[int],
    sigma: float = 0.0,
    clamp: float = 0.0,
    chunk_frames: int = 64,
) -> torch.Tensor:
    """Per-sample TDT costs [B] of joint_net(f + g), computed chunk by chunk over T.

    Args:
        f: projected encoder output [B, T, H] (RNNTJoint.project_encoder).
        g: projected (and masked) prediction output [B, U+1, H].
        labels: targets [B, U], blank-free.
        act_lens, label_lens: lengths [B].
        joint_net: the joint's activation / dropout / output Linear stack.
        blank: blank index among the token logits, the last one.
        durations: TDT durations, ascending (e.g. [0, 1, 2, 3, 4]).
        sigma: logit under-normalisation, as TDTLossNumba.
        clamp: gradient clamp, as TDTLossNumba (0 = off).
        chunk_frames: frames of logits alive at once (Tc).
    """
    if chunk_frames < 1:
        raise ValueError(f"chunk_frames must be at least 1, got {chunk_frames}")
    return ChunkedTDTJointLoss.apply(
        f,
        g,
        labels,
        act_lens,
        label_lens,
        joint_net,
        int(blank),
        list(durations),
        float(sigma),
        float(clamp),
        int(chunk_frames),
        *joint_net.parameters(),
    )
