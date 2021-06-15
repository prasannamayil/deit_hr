import logging
# from math import prod
from einops import rearrange, repeat as ra, repeat

import torch
from torch import nn
from torch.nn import functional as F

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

'''
    b = batch_size
    k = num_heads
    h/w = hight/width of image at scale i
    h1/w1 = hight/width of image at scale i+1 = h/w_blocks at scale i
    h2/w2 = hight/width of a block at scale i
    h3/w3 = hight/width of a block at scale i-1 = nbr of h/w_kids of each word at scale i = h_kids

    TODOs:
        - implement weight initialization
        - implement re-weighting of attention (beware: must depend on the block-sizes)
        - implement option to use same weights at all scales
        - implement different queries/keys for parents/peers/kids
        - implement overlapping attention fields to kids (might be difficult)
        - implement recurrent traversal of hierarchy (instead of parallel computations)
'''

class MSTensor(list):
    def __add__(self, ms_y):
        assert len(self) == len(ms_y), (
            f'Both MSTensor must have same length, but {len(self)} and {len(ms_y)} given.')
        ms_z = []
        for (x, y) in zip(self, ms_y):
            ms_z.append(x+y)
        return MSTensor(ms_z)


class QKV(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3*dim, bias=bias)

    def forward(self, x):
        # dim(x) = b x h x w x feature_dim
        qkv = self.qkv(x)
        q, k, v = ra(qkv, 'b h w (i k f) -> i b k h w f', i=3, k=self.num_heads)
        return q, k, v


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop=0.):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(drop),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(drop),)

    def forward(self, z):
        return self.mlp(z)


class MSLayer(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.ms_layer = nn.ModuleList(layers)

    def forward(self, ms_input):
        ms_out = []
        for x, layer in zip(ms_input, self.ms_layer):
            ms_out.append(layer(x))
        return MSTensor(ms_out)


def compute_scales_and_block_sizes(img_size, patch_size, words_per_block):
    # Compute num_scales, listify words_per_block if needed and compute
    # the num_patches in each block
    num_patches_at_scale_i = img_size // patch_size
    num_patches = [num_patches_at_scale_i]
    if type(words_per_block) == int:
        while num_patches_at_scale_i > 1:
            num_patches_at_scale_i //= words_per_block
            num_patches.append(num_patches_at_scale_i)
        num_patches.reverse()
        num_scales = len(num_patches)
        words_per_block = [1] + [words_per_block] * (num_scales-1)
    else:
        for wpb in reversed(words_per_block[1:]):
            num_patches_at_scale_i //= wpb
            num_patches.append(num_patches_at_scale_i)
        num_patches.reverse()
        num_scales = len(num_patches)

    _logger.debug(f'nbr of scales: {num_scales}')
    _logger.debug(f'words_per_block: {words_per_block}')
    _logger.debug(f'num_patches: {num_patches}')
    assert num_scales == len(num_patches) == len(words_per_block)
    # assert prod(words_per_block) * patch_size == img_size, (
    #     f'words_per_block={words_per_block} patch_size={patch_size} '
    #     f'img_size={img_size}')

    return num_scales, words_per_block, num_patches


def listify(val_or_list, num_scales):
    if type(val_or_list) != list:
        val_or_list = [val_or_list] * num_scales
    assert len(val_or_list) == num_scales, f'val_or_list: {val_or_list}'
    return val_or_list


class MSEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, words_per_block,
                 feature_dims, drops=0., num_channels=3):

        super().__init__()
        num_scales, words_per_block, num_patches = compute_scales_and_block_sizes(
            img_size, patch_size, words_per_block)
        self.num_scales = num_scales
        self.words_per_block = words_per_block
        self.num_patches = num_patches
        feature_dims = listify(feature_dims, num_scales)
        drops = listify(drops, num_scales)

        # Make layers
        self.embeddings = nn.ModuleList([
            nn.Conv2d(
                num_channels, dim, kernel_size=patch_size, stride=patch_size)
                    for dim in feature_dims])
        self.pos = nn.ParameterList([
            nn.Parameter(torch.zeros(h, h, dim))
                for h, dim in zip(num_patches, feature_dims)])
        self.embed_drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])

    def forward(self, x):
        # Decompose x in multi-scale x
        x_cur = x
        ms_x = [x_cur]
        for wpb in reversed(self.words_per_block[1:]):
            x_cur = F.avg_pool2d(x_cur, kernel_size=wpb)
            ms_x.append(x_cur)

        # Embed each scale of x
        ms_out = []
        for i in range(self.num_scales):
            out = self.embeddings[i](ms_x.pop())
            out = ra(out, 'b f h w -> b h w f')
            ms_out.append(self.embed_drops[i](out + self.pos[i]))
        return MSTensor(ms_out)


class MSAttention(nn.Module):
    def __init__(self, words_per_block, feature_dims, num_heads, attend_to_peers=True,
                 attend_to_parents=True, qkv_bias=False, attn_drops=0., proj_drops=0.):
        '''
        words_per_block (list)    list of length `num_scales` containing the number of patches in
                                  each block
        '''
        super().__init__()
        assert type(words_per_block) == list
        self.num_scales = len(words_per_block)
        self.words_per_block = words_per_block
        feature_dims = listify(feature_dims, self.num_scales)
        attn_drops = listify(attn_drops, self.num_scales)
        proj_drops = listify(proj_drops, self.num_scales)
        self.attend_to_peers = attend_to_peers
        self.attend_to_parents = attend_to_parents

        # Remark: num_heads must be int. Cannot be different in every scale yet, given the usual
        # attention mechanism. If we move to a model where each query can be tested against all
        # `num_heads` keys (1 key per head), then we could let the num_heads vary between scales.
        # Then we could also introduce separate values for `num_key_heads` and `num_query_heads`.

        self.ms_qkv = MSLayer([QKV(dim, num_heads, qkv_bias) for dim in feature_dims])
        self.attn_drops = [nn.Dropout(attn_drop) for attn_drop in attn_drops]
        self.ms_proj = MSLayer([nn.Linear(dim,dim) for dim in feature_dims])
        self.ms_proj_drop = MSLayer([nn.Dropout(proj_drop) for proj_drop in proj_drops])

    def forward(self, ms_input):
        ms_qkv = self.ms_qkv(ms_input) 
        ms_slf_attn = []

        for i in range(self.num_scales):
            q, k, v = ms_qkv[i]  # 3 x b x num_heads x h x w x (f = c // num_heads)
            h_block = self.words_per_block[i]
            n_blocks = q.shape[3] // h_block
            n_parents = 1 if i > 0 else 0
            h_kids = self.words_per_block[i+1] if i < (self.num_scales-1) else 0
            qk_scaling = (h_block + n_parents + h_kids) ** -0.5 
            # TODO: test different qk_scalings for attentiong to parents/peers/kids
            attn = []
            val = []

            # Attention to peers
            q_peer = ra(q, 'b k (h1 h2) (w1 w2) f -> b k h1 w1 (h2 w2) f', h2=h_block, w2=h_block)
            if self.attend_to_peers:
                k_peer = ra(k, 'b k (h1 h2) (w1 w2) f -> b k h1 w1 (h2 w2) f', h2=h_block, w2=h_block)
                attn.append(torch.einsum('bkhwlf,bkhwmf->bkhwlm', q_peer, k_peer) * qk_scaling)
                v = ra(v, 'b k (h1 h2) (w1 w2) f -> b k h1 w1 (h2 w2) f', h2=h_block, w2=h_block)
                val.append(repeat(v, 'b k h1 w1 h2w2 f -> b k h1 w1 r h2w2 f', r=h_block**2))

            # Attention to parents
            if self.attend_to_parents and i > 0:
                q_par = q_peer  # query to parent
                _, k_par, v_par = ms_qkv[i-1]
                k_par = ra(k_par, 'b k h1 w1 f -> b k h1 w1 1 f')
                attn.append(torch.einsum('bkhwlf,bkhwmf->bkhwlm', q_par, k_par) * qk_scaling)
                v_par = ra(v_par, 'b k h1 w1 f -> b k h1 w1 1 f')
                val.append(repeat(v_par, 'b k h1 w1 1 f -> b k h1 w1 r 1 f', r=h_block**2))

            # Attention to kids
            if i < (self.num_scales-1):
                q_kid = ra(q, 'b k h w f -> b k h w 1 f')  # query to kid
                _, k_kid, v_kid = ms_qkv[i+1]
                k_kid = ra(k_kid, 'b k (h h3) (w w3) f -> b k h w (h3 w3) f', h3=h_kids, w3=h_kids)
                attn_kid = torch.einsum('bkhwlf,bkhwmf->bkhwlm', q_kid, k_kid) * qk_scaling
                attn.append(ra(attn_kid, 'b k (h1 h2) (w1 w2) 1 g -> b k h1 w1 (h2 w2) g', h2=h_block, w2=h_block))
                val.append(ra(v_kid, 'b k (h1 h2 h3) (w1 w2 w3) f -> b k h1 w1 (h2 w2) (h3 w3) f',
                             h2=h_block, h3=h_kids, w2=h_block, w3=h_kids))

            _logger.debug(f'shape of val peers/par/kids: {[v.shape for v in val]}')
            attn = torch.cat(attn, dim=-1)
            attn = attn.softmax(dim=-1)  # TODO: add possible multiplicative scaling here
            attn = self.attn_drops[i](attn)
            val = torch.cat(val, dim=-2)

            # Resulting attn dims
            # attn_peers.shape = b x k x h1 x w1 x h_block**2 x h_block**2
            # attn_paren.shape = b x k x h1 x w1 x h_block**2 x 1
            # attn_kids.shape  = b x k x h1 x w1 x h_block**2 x h_kids**2
            # attn.shape[5] = h_block**2 + n_parents + (h_block*h_kids)**2
            # val.shape = attn.shape x f for all vals

            slf_attn = torch.einsum('bkhwlm,bkhwlnf->bkhwlf', attn, val)
            slf_attn = ra(slf_attn, 'b k h1 w1 (h2 w2) f -> b (h1 h2) (w1 w2) (k f)', h2=h_block, w2=h_block)
            ms_slf_attn.append(slf_attn)

        ms_out = self.ms_proj(ms_slf_attn)
        ms_out = self.ms_proj_drop(ms_slf_attn)
        return ms_out


class MSTransformer(nn.Module):
    def __init__(
            self, words_per_block, feature_dims, num_heads, mlp_hidden_dims, attend_to_peers=True,
            attend_to_parents=True, qkv_bias=False, mlp_drops=0., attn_drops=0., proj_drops=0.):
        '''
        words_per_block (list)  List of length `num_scales` that starts with
                                value 1
        '''
        super().__init__()

        assert type(words_per_block) == list
        self.num_scales = len(words_per_block)
        self.words_per_block = words_per_block
        feature_dims = listify(feature_dims, self.num_scales)
        mlp_hidden_dims = listify(mlp_hidden_dims, self.num_scales)
        mlp_drops = listify(mlp_drops, self.num_scales)
        attn_drops = listify(attn_drops, self.num_scales)
        proj_drops = listify(proj_drops, self.num_scales)

        self.ms_ln1 = MSLayer([nn.LayerNorm(dim, eps=1e-6) for dim in feature_dims])
        self.ms_attn = MSAttention(words_per_block, feature_dims, num_heads, attend_to_peers,
                                   attend_to_parents, qkv_bias, attn_drops, proj_drops)
        self.ms_ln2 = MSLayer([nn.LayerNorm(dim, eps=1e-6) for dim in feature_dims])
        self.ms_mlp = MSLayer([MLP(dim, hidden_dim, drop=drop)
            for (dim, hidden_dim, drop) in zip(feature_dims, mlp_hidden_dims, mlp_drops)])

    def forward(self, ms_z):
        ms_out = ms_z[:self.num_scales]  # possibly ignore finest scales
        ms_out = self.ms_attn(self.ms_ln1(ms_z)) + ms_out
        ms_out = self.ms_mlp(self.ms_ln2(ms_z)) + ms_out
        ms_out.extend(ms_z[self.num_scales:])  # concat ignored scales back
        return ms_out


class MultiScaleViT(nn.Module):
    def __init__(
            self, img_size, num_classes, patch_size, words_per_block, feature_dims, num_heads,
            num_transformers, mlp_hidden_dims, attend_to_peers=True, attend_to_parents=True,
            wait_for_top=False, qkv_bias=False, embed_drops=0., mlp_drops=0., attn_drops=0.,
            proj_drops=0., num_channels=3):
        '''
        words_per_block (list)  List of length `num_scales` that starts with
                                value 1
        '''
        super().__init__()
        num_scales, words_per_block, num_patches = compute_scales_and_block_sizes(
            img_size, patch_size, words_per_block)

        self.img_size = img_size
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_scales = num_scales
        self.words_per_block = words_per_block
        self.num_patches = num_patches
        self.num_transformers = num_transformers
        self.feature_dims = listify(feature_dims, self.num_scales)
        self.num_heads = num_heads
        self.mlp_hidden_dims = listify(mlp_hidden_dims, self.num_scales)
        self.attend_to_peers = attend_to_peers
        self.attend_to_parents = attend_to_parents
        self.wait_for_top = wait_for_top
        self.qkv_bias = qkv_bias
        self.embed_drops = listify(embed_drops, self.num_scales)
        self.mlp_drops = listify(mlp_drops, self.num_scales)
        self.attn_drops = listify(attn_drops, self.num_scales)
        self.proj_drops = listify(proj_drops, self.num_scales)
        self.num_channels = num_channels

        self.embedding = MSEmbedding(
            self.img_size, self.patch_size, self.words_per_block,
            self.feature_dims, self.embed_drops, self.num_channels)

        transformers = []
        for l in range(self.num_transformers):
            L = self.num_transformers  # L-l+1 = last small scale layers are irrelevant for computation graph
            s = min(l+2, L-l+1) if self.wait_for_top else L-l+1
            transformers.append(MSTransformer(
                    self.words_per_block[:s], self.feature_dims[:s], self.num_heads,
                    self.mlp_hidden_dims[:s], self.attend_to_peers, self.attend_to_parents,
                    self.qkv_bias, self.mlp_drops[:s], self.attn_drops[:s], self.proj_drops[:s]))
        self.transformers = nn.Sequential(*transformers)

        self.head = nn.Linear(self.feature_dims[0], self.num_classes, bias=True)

    def forward(self, x):
        ms_z = self.embedding(x)
        ms_z = self.transformers(ms_z)
        out = ms_z[0].reshape(x.shape[0], -1)
        return self.head(out)


def small_cifar_msvit(attend_to_peers=True, attend_to_parents=True, wait_for_top=True):
    drop=.1
    return MultiScaleViT(
                img_size=32,
                num_classes=10,
                patch_size=1,
                words_per_block = [1, 2, 2, 2, 2, 2],
                feature_dims=192,
                num_heads=3,
                num_transformers=12,
                mlp_hidden_dims=4*192,
                attend_to_peers=attend_to_peers,
                attend_to_parents=attend_to_parents,
                wait_for_top=wait_for_top,
                qkv_bias=False,
                embed_drops=drop,
                mlp_drops=drop,
                attn_drops=drop,
                proj_drops=drop,
                num_channels=3)

