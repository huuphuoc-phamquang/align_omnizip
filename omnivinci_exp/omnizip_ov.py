"""
Token compression for OmniVinci, called from the hook inserted by patch_model.py.

The hook fires once per 30-second segment with that segment's video and audio tokens
already separated, which is exactly the time window OmniZip operates on.

Methods:
    random         uniformly random token selection at the same budget - the control
                   group. Without it, any accuracy drop could be explained by "the model
                   tolerates fewer tokens" rather than by which tokens were chosen.
    omnizip        audio-guided: score audio tokens, keep the salient ones, and let the
                   per-segment audio saliency steer how hard the video is pruned.
    omnizip_audio  diagnostic. OmniZip's audio selection plus the audio-guided dynamic
                   allocation, but random video selection.
    omnizip_video  diagnostic. ISTM on the video tokens, random audio selection and a
                   uniform per-segment budget (the dynamic allocation needs the audio
                   saliency, so it switches off with it). This is the same cell as
                   "ISTC + Random" in Tab. 4 of the paper.

                   Why both: omnizip loses to random at every budget tested, with the
                   damage concentrated in temporal and causal reasoning rather than in
                   audio tasks, which points at the video compressor. Running the two
                   halves separately is what turns that inference into a measurement -
                   their gaps against `random` at the same budget should account for
                   the gap that full omnizip shows, and whatever is left over is
                   interaction between the two.

Budget
------
The paper uses one fixed (rho_a, rho_v) pair for every sample. That works on
Qwen2.5-Omni where each 2s window always holds 288 video : 50 audio tokens. It does not
work here: OmniVinci samples a fixed 128 frames regardless of duration while audio grows
at 750 tokens per 30s segment, so the video share slides from ~88% on a 23s clip to ~20%
on a 641s one. A fixed pair would compress different videos by different amounts and the
comparison across methods would not be FLOPs-matched.

Instead each segment is given a retention target and rho_v is solved from the tokens
actually present, keeping the paper's rule that audio is pruned more gently
(rho_a = alpha * rho_v, alpha < 1):

    keep = target * (V + A) = (1 - rho_v) * V + (1 - alpha * rho_v) * A
    =>  rho_v = (1 - target) * (V + A) / (V + alpha * A)

Dynamic allocation
------------------
OmniZip's headline idea is spending the budget unevenly across windows: information-dense
windows get pruned gently, sparse ones aggressively. The hook sees segments one at a
time, so the allocation is done online - each segment is pruned relative to the running
mean saliency, and whatever is over- or under-spent is carried into the remaining
segments so the video-level budget still lands on target.

Worth knowing when reading the results: a median WorldSense video is 78s, which is only
3 segments of 30s. Qwen2.5-Omni at 2s windows would give 39. With three decision points
the dynamic allocation has very little room to differ from uniform pruning, so a small
gap between omnizip and random here is expected and is a property of the architecture,
not evidence that the idea is wrong.

Audio saliency
--------------
The paper reads attention from inside the audio encoder (queries and keys of the last
self-attention layer). OmniVinci's audio tower does not expose attentions through
encode_sound(), so this uses self-attention over the *output* features instead:

    A = softmax(F F^T / sqrt(d)),  importance = column mean of A

Same quantity, except the query and key projections are identity rather than learned.
This is an approximation, not the paper's exact score, and any write-up should say so.
"""
import math
import os

import torch

# ---------------------------------------------------------------------------------
# State. Configured once per run, reset once per video.
# ---------------------------------------------------------------------------------

_CFG = {
    "method": "none",
    "target_retained": 1.0,
    "alpha": 0.5,          # rho_a = alpha * rho_v; 0.5 matches the paper's 0.3/0.6
    "seed": 0,
    "g": 3,                # secondary audio tokens merged per anchor
    "contextual_ratio": 0.05,
    "tokens_per_frame": 32,
    "max_rho": 0.95,
}

_STATE = {
    "budget_debt": 0.0,    # tokens still owed (+) or overspent (-) by earlier segments
    "saliency_hist": [],
    "kept": 0,
    "total": 0,
    "videos": 0,
}

_rng = torch.Generator()


def configure(method="none", target_retained=1.0, alpha=0.5, seed=0,
              g=3, contextual_ratio=0.05, tokens_per_frame=32, max_rho=0.95):
    _CFG.update(method=method, target_retained=float(target_retained),
                alpha=float(alpha), seed=int(seed), g=int(g),
                contextual_ratio=float(contextual_ratio),
                tokens_per_frame=int(tokens_per_frame), max_rho=float(max_rho))
    _rng.manual_seed(int(seed))
    reset_video()
    _STATE["kept"] = 0
    _STATE["total"] = 0
    _STATE["videos"] = 0


def reset_video():
    """Call before each sample: the budget carry-over must not leak across videos."""
    _STATE["budget_debt"] = 0.0
    _STATE["saliency_hist"] = []
    _STATE["videos"] += 1


def stats():
    total = max(_STATE["total"], 1)
    return {"tokens_kept": _STATE["kept"], "tokens_total": _STATE["total"],
            "actual_retained": _STATE["kept"] / total, "videos": _STATE["videos"],
            "video_compressor": video_path_info()}


# ---------------------------------------------------------------------------------
# Scoring and selection
# ---------------------------------------------------------------------------------

def _audio_importance(feats):
    """Per-token importance, as the column mean of self-attention over the features.

    Columns, not rows: every row of a softmax sums to 1, so row means are 1/N for every
    token and carry no signal. The column sum is how much attention a token *receives*,
    which is the quantity that separates tokens.
    """
    f = feats.float()
    d = f.shape[-1]
    scores = torch.matmul(f, f.transpose(0, 1)) / math.sqrt(d)
    attn = torch.softmax(scores, dim=-1)
    return attn.mean(dim=0)


def _keep_topk_with_anchors(feats, n_keep, importance):
    """Keep the highest scoring tokens plus evenly spaced anchors.

    Pure top-k clusters the survivors wherever the audio is loudest and leaves long
    stretches of the window unrepresented. The paper adds uniformly sampled anchors for
    the same reason; contextual_ratio controls how many.
    """
    n = feats.shape[0]
    if n_keep >= n:
        return torch.arange(n, device=feats.device)

    n_anchor = int(round(_CFG["contextual_ratio"] * n))
    n_anchor = max(0, min(n_anchor, n_keep))
    n_top = n_keep - n_anchor

    keep = torch.zeros(n, dtype=torch.bool, device=feats.device)
    if n_top > 0:
        keep[torch.topk(importance, n_top).indices] = True
    if n_anchor > 0:
        step = max(1, n // n_anchor)
        keep[torch.arange(0, n, step, device=feats.device)[:n_anchor]] = True

    idx = keep.nonzero(as_tuple=True)[0]
    if idx.numel() > n_keep:
        # Anchors can land on tokens top-k already chose. Drop the weakest of the
        # survivors rather than the last ones in time order, which would bias the
        # selection toward the start of the window.
        order = torch.topk(importance[idx], n_keep).indices
        idx = idx[order].sort().values
    elif idx.numel() < n_keep:                     # top them back up
        spare = (~keep).nonzero(as_tuple=True)[0]
        extra = spare[torch.topk(importance[spare], n_keep - idx.numel()).indices]
        idx = torch.cat([idx, extra]).sort().values
    return idx


_VIDEO_PATH = {"name": None}


def _istm(video_feature, num_tokens_per_frame, merging_ratio):
    """Verbatim port of omnizip_core.omnizip_istm from the OmniZip repo.

    Vendored rather than imported. The import used to depend on the OmniZip repo sitting
    one directory up on the server; when that failed the code silently fell back to a
    different compressor and the results gave no way to tell which had run. Carrying the
    algorithm here makes every run reproducible from this file alone.

    Interleaved spatio-temporal selection, alternating by frame:
      even frames - DPC-KNN clustering, keep the cluster centres (spatial redundancy)
      odd frames  - keep the tokens LEAST similar to the previous frame (temporal change)
    """
    num_frames = video_feature.shape[0] // num_tokens_per_frame
    mask = torch.zeros(video_feature.shape[0], dtype=torch.bool,
                       device=video_feature.device)

    def dpcknn(tokens, keep_rate=0.5, k=5):
        N = tokens.shape[0]
        num_keep = int(N * keep_rate)
        if num_keep >= N:
            return torch.arange(N, device=tokens.device)
        k = min(k, N - 1)
        with torch.no_grad():
            normed = torch.nn.functional.normalize(tokens.float(), dim=1)
            dist = 1 - torch.mm(normed, normed.T)
            dist.fill_diagonal_(float("inf"))
            knn_dist, _ = torch.topk(dist, k, dim=1, largest=False)
            rho = torch.exp(-knn_dist.mean(dim=1))
            higher_mask = rho.unsqueeze(0) > rho.unsqueeze(1)
            delta_dist = dist.clone()
            delta_dist[~higher_mask] = float("inf")
            delta = delta_dist.min(dim=1).values
            is_peak = delta.isinf()
            if is_peak.any():
                finite_dist = dist.clone()
                finite_dist.fill_diagonal_(0)
                delta[is_peak] = finite_dist[is_peak].max(dim=1).values
            score = delta * rho
            return torch.topk(score, max(num_keep, 1)).indices

    keep_ratio = 1.0 - merging_ratio
    for t in range(num_frames):
        start = t * num_tokens_per_frame
        tokens = video_feature[start:start + num_tokens_per_frame]
        num_keep = int(num_tokens_per_frame * keep_ratio)
        if num_keep >= num_tokens_per_frame:
            keep_idx = torch.arange(num_tokens_per_frame, device=tokens.device)
        elif t % 2 == 0:
            keep_idx = dpcknn(tokens, keep_rate=keep_ratio, k=5)
        else:
            prev = video_feature[start - num_tokens_per_frame:start]
            sim = torch.nn.functional.cosine_similarity(
                torch.nn.functional.normalize(tokens.float(), p=2, dim=1),
                torch.nn.functional.normalize(prev.float(), p=2, dim=1), dim=1)
            keep_idx = sim.topk(max(num_keep, 1), largest=False).indices
        mask[start + keep_idx] = True
    return mask


def video_path_info():
    return _VIDEO_PATH["name"] or "not run yet"


def _fit_to_budget(idx, n_total, n_keep, device):
    """Force a selection to hold exactly n_keep tokens.

    omnizip_istm keeps int(32 * keep_ratio) tokens per frame, so the total is a floor of
    the budget rather than the budget. Truncating with idx[:n_keep] would have discarded
    whole frames from the end of the segment; spreading the correction across the
    selection keeps temporal coverage intact.
    """
    n_sel = idx.numel()
    if n_sel == n_keep:
        return idx
    if n_sel > n_keep:
        pick = torch.linspace(0, n_sel - 1, n_keep, device=device).long()
        return idx[pick]
    chosen = torch.zeros(n_total, dtype=torch.bool, device=device)
    chosen[idx] = True
    spare = (~chosen).nonzero(as_tuple=True)[0]
    need = min(n_keep - n_sel, spare.numel())
    if need > 0:
        pick = torch.linspace(0, spare.numel() - 1, need, device=device).long()
        idx = torch.cat([idx, spare[pick]]).sort().values
    return idx


def _compress_video(feats, n_keep):
    """Reduce a segment's video tokens to n_keep using OmniZip's ISTM."""
    n = feats.shape[0]
    if n_keep >= n:
        return feats
    tpf = _CFG["tokens_per_frame"]

    if tpf > 0 and n >= 2 * tpf:
        aligned = (n // tpf) * tpf
        ratio = n_keep / n
        mask = _istm(feats[:aligned], num_tokens_per_frame=tpf,
                     merging_ratio=1.0 - ratio)
        idx = mask.nonzero(as_tuple=True)[0]
        if aligned < n:                       # tail shorter than one frame
            tail_keep = max(int(round((n - aligned) * ratio)), 1)
            tail = torch.linspace(aligned, n - 1, tail_keep,
                                  device=feats.device).long()
            idx = torch.cat([idx, tail]).sort().values
        idx = _fit_to_budget(idx, n, n_keep, feats.device)
        _VIDEO_PATH["name"] = "omnizip_istm (vendored DPC-KNN + temporal)"
        return feats[idx]

    _VIDEO_PATH["name"] = "uniform stride (segment shorter than 2 frames)"
    step = n / n_keep
    idx = torch.tensor([int(i * step) for i in range(n_keep)], device=feats.device)
    return feats[idx.clamp(max=n - 1)]


# ---------------------------------------------------------------------------------
# Entry point called by the hook in modeling_vila.py
# ---------------------------------------------------------------------------------

def compress_segment(vis, aud, segment_index=0, num_segments=1):
    method = _CFG["method"]
    if method == "none" or _CFG["target_retained"] >= 1.0:
        return vis, aud

    n_v = 0 if vis is None else vis.shape[0]
    n_a = 0 if aud is None else aud.shape[0]
    if n_v + n_a == 0:
        return vis, aud

    target = _CFG["target_retained"]
    alpha = _CFG["alpha"]

    # Budget for this segment, plus whatever earlier segments over- or under-spent.
    budget = target * (n_v + n_a) + _STATE["budget_debt"]
    budget = max(1.0, min(budget, float(n_v + n_a)))

    rho_v = (1.0 - budget / (n_v + n_a)) * (n_v + n_a) / max(n_v + alpha * n_a, 1e-6)
    rho_v = float(min(max(rho_v, 0.0), _CFG["max_rho"]))
    rho_a = min(alpha * rho_v, _CFG["max_rho"])

    audio_guided = method in ("omnizip", "omnizip_audio")

    if audio_guided and n_a > 0:
        # Audio-guided allocation: a segment whose audio is more salient than the
        # running mean keeps more of its video, and vice versa. Bounded to +/-25% so a
        # single loud segment cannot consume the whole video's budget.
        imp = _audio_importance(aud)
        saliency = float(imp.max() / (imp.mean() + 1e-9))
        hist = _STATE["saliency_hist"]
        if hist:
            ref = sum(hist) / len(hist)
            adjust = max(-0.25, min(0.25, (saliency - ref) / (ref + 1e-9)))
            rho_v = float(min(max(rho_v * (1.0 - adjust), 0.0), _CFG["max_rho"]))
        hist.append(saliency)
    else:
        imp = None

    keep_v = int(round(n_v * (1.0 - rho_v)))
    if n_v > 0:
        keep_v = max(keep_v, 1)

    # Audio takes whatever the budget has left. Normally this equals
    # n_a * (1 - alpha * rho_v), the paper's gentle audio pruning. It only diverges when
    # rho_v hits the cap and the target is still not met, which happens on
    # audio-dominant videos: at 75-80% audio, keeping audio at half the video rate makes
    # a 35% target unreachable even after removing 95% of the video. Letting audio carry
    # the shortfall keeps every method at the same FLOPs, which is what makes the
    # comparison valid. It is a deviation from the paper and belongs in the write-up.
    keep_a = int(round(budget)) - keep_v
    keep_a = max(1 if n_a > 0 else 0, min(keep_a, n_a))
    rho_a = 1.0 - (keep_a / n_a) if n_a > 0 else 0.0

    new_vis, new_aud = vis, aud

    if n_a > 0 and keep_a < n_a:
        if audio_guided:
            if imp is None:
                imp = _audio_importance(aud)
            idx = _keep_topk_with_anchors(aud, max(keep_a, 1), imp)
        else:
            perm = torch.randperm(n_a, generator=_rng)[:max(keep_a, 1)]
            idx = perm.sort().values.to(aud.device)
        new_aud = aud[idx]

    if n_v > 0 and keep_v < n_v:
        # omnizip_audio keeps OmniZip's audio selection but picks video tokens at
        # random. If accuracy recovers to the random baseline under this setting, the
        # loss came from the video compressor rather than from the method.
        # omnizip_video is the mirror image: it falls through to _compress_video below
        # while its audio was pruned at random above.
        if method in ("random", "omnizip_audio"):
            perm = torch.randperm(n_v, generator=_rng)[:max(keep_v, 1)]
            new_vis = vis[perm.sort().values.to(vis.device)]
        else:
            new_vis = _compress_video(vis, max(keep_v, 1))

    kept = (0 if new_vis is None else new_vis.shape[0]) + \
           (0 if new_aud is None else new_aud.shape[0])
    _STATE["budget_debt"] = budget - kept
    _STATE["kept"] += kept
    _STATE["total"] += n_v + n_a

    return new_vis, new_aud


def install_method(model, method="none", rho_audio=0.3, rho_video=0.6,
                   target_retained=None, **kwargs):
    """Called by eval_omnivinci.py.

    rho_audio/rho_video are accepted for compatibility with the paper's parameterisation
    and converted into a retention target, since a fixed pair does not hold a constant
    budget on OmniVinci (see the module docstring).
    """
    if target_retained is None:
        # Interpret the pair at the token mix of a median WorldSense video:
        # 4096 video tokens, 3 segments x 750 audio tokens.
        v, a = 4096.0, 2250.0
        target_retained = ((1 - rho_video) * v + (1 - rho_audio) * a) / (v + a)
    configure(method=method, target_retained=target_retained,
              alpha=(rho_audio / rho_video) if rho_video else 0.5, **kwargs)
    print("  compression: method=%s target_retained=%.3f alpha=%.2f"
          % (method, target_retained, _CFG["alpha"]))

    return model
