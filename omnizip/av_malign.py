# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_
from einops import rearrange


def norm_fea(fea, norm_flag):
    if norm_flag:
        return F.normalize(fea, p=2, dim=-1)
    return fea

class Cross_modal_interaction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.norm = nn.Identity()
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_attention(self, layer):
        """
        Share the frozen input norm and q/k/v (plus per-head q/k norms, if any) of an LLM decoder layer.
        RoPE is skipped, so scores ignore token positions.
        """
        attn = layer.self_attn
        self.norm = layer.input_layernorm
        self.q_proj, self.k_proj, self.v_proj = attn.q_proj, attn.k_proj, attn.v_proj
        self.q_norm = getattr(attn, 'q_norm', nn.Identity())
        self.k_norm = getattr(attn, 'k_norm', nn.Identity())
        head_dim = attn.head_dim
        self.num_heads = attn.q_proj.out_features // head_dim
        self.num_kv_heads = attn.k_proj.out_features // head_dim
        self.scale = head_dim**-0.5

    def _split_heads(self, x, proj, head_norm, num_heads):
        x = rearrange(proj(self.norm(x)), 'b n (h c) -> b n h c', h=num_heads)
        x = head_norm(x).transpose(1, 2) # [B, num_heads, n, head_dim]
        # GQA: repeat kv heads to match query heads
        return x.repeat_interleave(self.num_heads // num_heads, dim=1)

    def attn_logits(self, query, key):
        """
        Return:
            scaled attention logits: [B, nheads, n_query, n_key]
        """
        q = self._split_heads(query, self.q_proj, self.q_norm, self.num_heads)
        k = self._split_heads(key, self.k_proj, self.k_norm, self.num_kv_heads)
        return (q @ k.transpose(-2, -1)) * self.scale

    def forward(self, img_fea, aud_fea, aq=True, sigmoid=False, softmax=False):
        """
        Args:
            img_fea: [b, num_token, dim]
            aud_fea: [b, k, dim]
            aq: True: a is query; False: i is query
        Return
            if aq:
                fuse_fea: [b, k, dim]
            else:
                fuse_fea: [b, num_token, dim]
        """
        if len(aud_fea.shape) == 2:
            aud_fea = aud_fea.unsqueeze(1)
        if aq:
            B, N, _ = aud_fea.shape
            query = aud_fea
            key = img_fea
            value = img_fea
        else:
            B, N, _ = img_fea.shape # [B, h*w, dim]
            query = img_fea
            key = aud_fea
            value = aud_fea

        attn = self.attn_logits(query, key) # [B, nheads, h*w, 1]
        v = self._split_heads(value, self.v_proj, nn.Identity(), self.num_kv_heads) # [B, nheads, 1, head_dim]

        if sigmoid:
            attn = torch.sigmoid(attn) # [B, nheads, h*w, 1]
        if softmax:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x
    
class GumbelSoftmaxClustering(nn.Module):
    def __init__(self, num_clusters, dim, temperature=1.0):
        super(GumbelSoftmaxClustering, self).__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, dim))  # Learnable cluster centers

    def forward(self, img_fea):
        """
        Args:
            img_fea: [B, H*W, dim] -> image features
        Returns:
            cluster_assignments: [B, H*W, num_clusters] -> soft cluster assignments
            cluster_centers: [B, H*W, dim] -> recomputed cluster centers
        """
        # Compute similarity (dot product) between img_fea and cluster centers
        logits = torch.einsum('bnd,kd->bnk', img_fea, self.cluster_centers)  # [B, H*W, num_clusters]

        # Apply Gumbel Softmax for cluster assignment
        gumbel_samples = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)  # [B, H*W, num_clusters]

        cluster_centers = torch.einsum('bnk,kd->bnd', gumbel_samples, self.cluster_centers)  # [B, H*W, dim]
        # cluster_centers = torch.einsum('bnk,bnd->bkd', gumbel_samples, img_fea) # 

        return cluster_centers, gumbel_samples

class SoftKMeansClustering(nn.Module):
    """Parameter-free vMF-mixture EM (spherical soft k-means), optional Sinkhorn balancing."""
    def __init__(self, num_clusters, temperature=0.1, iters=5, sinkhorn_iters=0):
        super(SoftKMeansClustering, self).__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.iters = iters
        self.sinkhorn_iters = sinkhorn_iters

    def _assign(self, x, mu):
        logits = x @ mu.transpose(1, 2) / self.temperature  # [B, H*W, num_clusters]
        if self.sinkhorn_iters == 0:
            return logits.softmax(dim=-1)
        log_r = logits
        for _ in range(self.sinkhorn_iters):
            log_r = log_r - log_r.logsumexp(dim=1, keepdim=True)  # equal mass per cluster
            log_r = log_r - log_r.logsumexp(dim=2, keepdim=True)  # each token sums to 1
        return log_r.exp()

    def forward(self, img_fea):
        """
        Args:
            img_fea: [B, H*W, dim] -> image features
        Returns:
            cluster_centers: [B, H*W, dim] -> per-token soft-assigned center
            cluster_assignments: [B, H*W, num_clusters] -> soft cluster assignments
        """
        fea = img_fea.float()
        x = F.normalize(fea, dim=-1)
        N = x.shape[1]
        K = min(self.num_clusters, N)
        mu = x[:, torch.linspace(0, N - 1, K, device=x.device).long()]  # deterministic init, [B, K, dim]

        for _ in range(self.iters):
            r = self._assign(x, mu)
            mu = F.normalize(r.transpose(1, 2) @ x, dim=-1)  # M-step on the sphere

        r = self._assign(x, mu)
        # Centers in the raw feature scale so q/k/v projections see the usual magnitudes
        raw_mu = (r.transpose(1, 2) @ fea) / r.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)  # [B, K, dim]
        cluster_centers = r @ raw_mu  # [B, H*W, dim]

        return cluster_centers.to(img_fea.dtype), r.to(img_fea.dtype)

class AV_Aligner(nn.Module):
    def __init__(self, dim, num_heads=1, num_clusters=5):
        super(AV_Aligner, self).__init__()

        self.num_clusters = num_clusters

        self.fc = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(dim)

        self.gumbel_clustering = SoftKMeansClustering(num_clusters)

        self.av_cros_interaction = Cross_modal_interaction(dim=dim, num_heads=num_heads)
        self.va_cros_interaction = Cross_modal_interaction(dim=dim, num_heads=num_heads)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_attention(self, layer):
        """
        Reuse the frozen attention of an LLM decoder layer (e.g. thinker.model.layers[0]) for both
        cross-modal interactions. fc has no frozen counterpart, so it is dropped and audio tokens
        are weighted by their attention logits instead (see _pool_audio).
        """
        self.av_cros_interaction.load_attention(layer)
        self.va_cros_interaction.load_attention(layer)
        self.fc = None

    def _pool_audio(self, aud_fea, cluster_centers):
        """
        Args:
            aud_fea: [B, K, dim] -> audio features
            cluster_centers: [B, H*W, dim] -> clustered image features
        Returns:
            weighted_aud_fea: [B, 1, dim] -> audio features pooled by visual relevance
        """
        if self.fc is None:
            # Weight each audio token by its strongest match to the clustered image tokens
            logits = self.av_cros_interaction.attn_logits(aud_fea, cluster_centers) # [B, nheads, k, h*w]
            attention_scores = logits.amax(dim=-1).mean(dim=1).softmax(dim=1).unsqueeze(-1) # [B, k, 1]
        else:
            aud_enhance_fea = self.av_cros_interaction(cluster_centers, aud_fea, aq=True, softmax=True) # [B, k, dim]
            attention_scores = self.fc(aud_enhance_fea).softmax(dim=1) # [B, k, 1]
        return (attention_scores * aud_fea).sum(dim=1, keepdim=True)

    def forward(self, aud_fea, img_fea):
        """
        Args:
            aud_fea: [B, K, dim] -> audio features
            img_fea: [B, H*W, dim] -> image features
        Returns:
            enhanced_img_fea: [B, H*W, dim] -> enhanced image features
        """
        # Step 1: Cluster image features into K clusters
        cluster_centers, _ = self.gumbel_clustering(img_fea) # [B, img_cluster, dim]

        weighted_aud_fea = self._pool_audio(aud_fea, cluster_centers) # [B, 1, dim]

        # Step 6: Enhance image features based on the audio-visual similarity map
        enhanced_img_fea = self.va_cros_interaction(img_fea, weighted_aud_fea, aq=False, sigmoid=True)  

        # enhanced_img_fea = self.layer_norm(enhanced_img_fea + img_fea)
        enhanced_img_fea = enhanced_img_fea + img_fea

        return enhanced_img_fea

    def token_scores(self, aud_fea, img_fea):
        """
        Args:
            aud_fea: [B, K, dim] -> audio features
            img_fea: [B, H*W, dim] -> image features
        Returns:
            scores: [B, H*W] -> audio-visual relevance of each image token
        """
        cluster_centers, _ = self.gumbel_clustering(img_fea)

        weighted_aud_fea = self._pool_audio(aud_fea, cluster_centers) # [B, 1, dim]

        # Same sigmoid attention that va_cros_interaction uses to enhance img_fea
        attn = torch.sigmoid(self.va_cros_interaction.attn_logits(img_fea, weighted_aud_fea)) # [B, nheads, h*w, 1]

        return attn.mean(dim=1).squeeze(-1)