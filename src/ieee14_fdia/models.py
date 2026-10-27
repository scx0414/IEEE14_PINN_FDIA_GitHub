"""Task adapters using the author's complete sLSTM block stack (vanilla backend)."""

import torch
from torch import nn
from ._vendor.xlstm_core import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


def stack(dim, blocks, heads, context, conv):
    return xLSTMBlockStack(
        xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=heads,
                    conv1d_kernel_size=conv,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            embedding_dim=dim,
            num_blocks=blocks,
            context_length=context,
            slstm_at="all",
            dropout=0.0,
            add_post_blocks_norm=True,
        )
    )


class StateEstimator(nn.Module):
    def __init__(self, family, c):
        super().__init__()
        self.family = family
        self.c = c
        self.length = c["window"]
        dim = c["dim"]
        head = c.get("head", 128)
        if family == "LSTM":
            self.encoder = nn.LSTM(28, dim, c["blocks"], batch_first=True, dropout=0.0)
            features = dim
        elif family == "P_sLSTM":
            self.patch = c["patch"]
            self.stride = c["stride"]
            assert (
                self.length - self.patch
            ) % self.stride == 0, "last patch must reach current observation"
            self.npatch = (self.length - self.patch) // self.stride + 1
            self.embedding = nn.Linear(self.patch, dim)
            self.encoder = stack(dim, c["blocks"], 4, self.npatch, 4)
            # Keep all patch outputs as in the author's forecasting code, but
            # replace pred_len projection by a learned temporal feature readout.
            self.temporal_readout = nn.Linear(dim * self.npatch, c.get("read", 32))
            features = 28 * c.get("read", 32)
        elif family == "xLSTM_Mixer":
            self.time_mixing = nn.Linear(self.length, c["time_features"])
            self.embedding = nn.Linear(c["time_features"], dim)
            self.memory = nn.Parameter(torch.randn(1, 1, dim) * 0.01)
            self.encoder = stack(dim, c["blocks"], 4, 29, 0)
            self.view_readout = nn.Linear(dim * 2, c.get("read", 32))
            features = 28 * c.get("read", 32)
        else:
            raise ValueError(family)
        # Preserve the historical base initialization. AdaptedEstimator replaces
        # this head and includes the input shortcut only when config says so.
        self.head = nn.Sequential(
            nn.Linear(features + 28, head), nn.Tanh(), nn.Linear(head, 27)
        )

    def encode(self, x):
        x = x[:, -self.length :].contiguous()
        b = x.shape[0]
        if self.family == "LSTM":
            return self.encoder(x)[0][:, -1]
        if self.family == "P_sLSTM":
            patches = x.transpose(1, 2).unfold(-1, self.patch, self.stride)
            e = self.embedding(patches.reshape(b * 28, self.npatch, self.patch))
            h = self.encoder(e)
            return self.temporal_readout(h.reshape(b * 28, -1)).reshape(b, -1)
        # NLinear-style centering + residual in the input feature space, not in
        # voltage space. K is an intermediate feature width, not a future label.
        last = x[:, -1, :].detach().unsqueeze(-1)
        initial = self.time_mixing(x.transpose(1, 2) - last) + last
        e = self.embedding(initial)
        e = torch.cat([self.memory.expand(b, -1, -1), e], 1)
        forward = self.encoder(e)
        # Default downloaded Mixer code reverses latent FEATURES, not time or
        # node order. Both passes share exactly the same sLSTM parameters.
        reverse = self.encoder(torch.flip(e, [-1]))
        mixed = self.view_readout(torch.cat([forward[:, 1:], reverse[:, 1:]], -1))
        return mixed.reshape(b, -1)

    def forward(self, x, zero_encoder=False):
        encoded = self.encode(x)
        if zero_encoder:
            encoded = encoded * 0.0
        return self.head(torch.cat([encoded, x[:, -1, :]], -1))


class AdaptedEstimator(StateEstimator):
    """Preserves existing P-sLSTM/Mixer checkpoint keys and readout initialization."""

    def __init__(self, family, config):
        super().__init__(family, config)
        size = self.head[0].in_features - (0 if config["shortcut"] else 28)
        self.head = nn.Sequential(
            nn.Linear(size, config["head"]), nn.Tanh(), nn.Linear(config["head"], 27)
        )
        for p in self.head.parameters():
            if p.ndim >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, x):
        z = self.encode(x)
        if self.c["shortcut"]:
            z = torch.cat([z, x[:, -1]], -1)
        return self.head(z)


class LSTMEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.length = config["window"]
        self.lstm = nn.LSTM(
            28, config["hidden"], config["recurrent_layers"], batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(config["hidden"], config["head_width"]),
            nn.Tanh(),
            nn.Linear(config["head_width"], 27),
        )
        for p in self.parameters():
            if p.ndim >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, x):
        h = self.lstm(x[:, -self.length :].contiguous())[0][:, -1]
        return self.head(h)


def build_model(family, config):
    if family == "LSTM":
        return LSTMEstimator(config)
    if family in ("P_sLSTM", "xLSTM_Mixer"):
        return AdaptedEstimator(family, config)
    raise ValueError(f"Unknown model family: {family}")
