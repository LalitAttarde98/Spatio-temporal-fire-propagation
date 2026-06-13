# Base code from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class BiMamba(nn.Module):

    def __init__(self, input_dim, hidden_dim, downsample=[28, 32],resize_in=[112, 32],resize_back=[113, 32], 
                 output_channels=1, input_frames=5, future_frames=1):
        super().__init__()
        self.H_c, self.W_c = downsample
        self.future_frames = future_frames
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim

        self.tokenizer = nn.Sequential(
            nn.Conv2d(input_dim, input_dim * 2, kernel_size=(3,3), padding=(1,1)), # 113 -> 113
            nn.BatchNorm2d(input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim * 2, input_dim * 4, kernel_size=(3,3), padding=(1,1)), # 113 -> 113
            nn.BatchNorm2d(input_dim * 4),
            nn.ReLU(inplace=True),
            nn.Upsample(size=resize_in, mode='bilinear', align_corners=False), # 112
            nn.Conv2d(input_dim * 4, input_dim * 8, kernel_size=(4,3), stride=(2,1), padding=(1,1)), # 112 -> 56
            nn.BatchNorm2d(input_dim * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim * 8, input_dim * 16, kernel_size=(3,3), padding=(1,1)), # 56 -> 56
            nn.BatchNorm2d(input_dim * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim * 16, hidden_dim,  kernel_size=(3,3), padding=(1,1)), # 56 -> 56
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.H_c * self.W_c, hidden_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, input_frames, hidden_dim))

        self.predictor = nn.Sequential(
            nn.Upsample(size=resize_back, mode='bilinear', align_corners=False), # 56 -> 113
            nn.Conv2d(input_frames * hidden_dim, hidden_dim, kernel_size=(3,3), padding=(1,1)), # 113 -> 113
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=(3,3), padding=(1,1)), # 113 -> 113
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, future_frames * output_channels, kernel_size=(3,3), padding=(1,1)), # 113 -> 113
        )

        self.hydra = Hydra(d_model=hidden_dim,
                        d_state=hidden_dim,
                        d_conv=7,
                        expand=2,
                        headdim=64,
                        dt_minmax=[0.001, 0.1],
                        learnable_init_states=False,
                        chunk_size=self.H_c*self.W_c,
        )

    def forward(self, x):
        B, L_in, C_in, H, W = x.shape
        x = x.view(B * L_in, C_in, H, W)
        x = self.tokenizer(x)

        x = rearrange(x, "(b l) c h w -> b l (h w) c", b=B, l=L_in)        
        
        x = x + self.pos_embed_spatial + self.pos_embed_temporal.unsqueeze(2)

        x = rearrange(x, "b l hw c -> b (l hw) c")
        x = self.hydra(x)

        x = rearrange(x, "b (l h w) c -> b (l c) h w", b=B, l=L_in, h=self.H_c, w=self.W_c)
        x = self.predictor(x)

        x = rearrange(x, "b (l c) h w -> b l c h w", l=self.future_frames, c=self.output_channels)
        predictions = torch.sigmoid(x)

        return predictions


class Hydra(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=7,
        expand=2,
        headdim=64,
        ngroups=1,
        dt_minmax=[0.001, 0.1],
        learnable_init_states=False,
        chunk_size=256,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.learnable_init_states = learnable_init_states
        self.chunk_size = chunk_size

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + (4 * self.d_state) + (2 * self.nheads)
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * (2 * self.d_state)
        self.conv1d = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=d_conv,groups=conv_dim,padding=d_conv // 2,
        )

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_minmax[1]) - math.log(dt_minmax[0]))
            + math.log(dt_minmax[0])
        )
        dt = torch.clamp(dt, min=1e-4)
        
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # Inverse of softplus
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        A = torch.ones(self.nheads, dtype=torch.float32)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False)

        self.norm = RMSNormGated(self.d_inner, eps=1e-5)

        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x, seq_idx=None):

        batch, seqlen, dim = x.shape

        zxbcdt = self.in_proj(x)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log.float())  # (nheads) 
        initial_states = repeat(self.init_states, "... -> b ...", b=2*batch) if self.learnable_init_states else None

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + (4 * self.d_state), 2 * self.nheads],
            dim=-1
        )

        dt = torch.cat((dt[:, :, :self.nheads], torch.flip(dt[:, :, self.nheads:], (1,))), dim=0)
        dt = F.softplus(dt + self.dt_bias)  # (2 * B, L, nheads)

        xBC = F.silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + (4 * d_state))

        x, BC = torch.split(xBC, [self.d_inner, (4 * self.d_state)], dim=-1)
        x_og = x
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)
        BC = torch.cat(
            (BC[:, :, :2 * self.d_state],
             torch.flip(BC[:, :, 2 * self.d_state:], (1,))),
            dim=0
        )
        B, C = torch.split(BC, [self.d_state, self.d_state], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        y, ssm_state = self.ssd(
            x = x * dt.unsqueeze(-1),
            A = A * dt,
            B = rearrange(B, "b l n -> b l 1 n"),
            C = rearrange(C, "b l n -> b l 1 n"),
            chunk_size = self.chunk_size,
            initial_states = initial_states,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        #y = torch.roll(y, shifts=1, dims=1)
        #y[:, 0, :] = 0.0
        y_fw, y_bw = y[:batch], torch.flip(y[batch:], (1,))
        y = y_fw + y_bw + x_og * repeat(
            F.linear(x_og, self.fc_D.weight, bias=self.D), "b l h -> b l (h p)", p=self.headdim
        )

        y = self.norm(y, z)
        out = self.out_proj(y)

        return out
    
    def segsum(self, x):
        """Stable segment sum calculation.

        `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

        Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
        """
        T = x.size(-1)
        x = repeat(x, "... d -> ... d e", e=T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum


    def ssd(self, x, A, B, C, chunk_size, initial_states=None):
        """Structed State Space Duality (SSD) - the core of Mamba-2

        This is almost the exact same minimal SSD code from the blog post.

        Arguments
            x: (batch, seqlen, n_heads, d_head)
            A: (batch, seqlen, n_heads)
            B: (batch, seqlen, n_heads, d_state)
            C: (batch, seqlen, n_heads, d_state)
        Return
            y: (batch, seqlen, n_heads, d_head)
        """
        assert x.shape[1] % chunk_size == 0

        # Rearrange into chunks
        # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
        # This is not implemented and left as an exercise for the reader 😜
        x, A, B, C = [
            rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
        ]

        A = rearrange(A, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

        # 2. Compute the state for each intra-chunk
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

        # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
        # (middle term of factorization of off-diag blocks; A terms)
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

        return Y, final_state

class RMSNormGated(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z=None):
        if z is not None:
            x = x * F.silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


if __name__ == "__main__":
    batch_size = 2
    time_steps = 5
    channels = 5
    height = 113
    width = 32

    x = torch.randn(batch_size, 5 * 28 * 32, 64)

    model = Hydra(d_model=64,
        d_state=64,
        d_conv=7,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=5,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,)

    output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)