import torch
import torch.nn as nn
import torch.nn.functional as F

class SFMHead(nn.Module):
    """SFM head for predicting scaled mel, time, and variance.

    This module is based on Figure 2 in Appendix B of the
    "Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis" paper.
    It takes hidden states from the encoder and outputs the parameters
    needed to construct an intermediate state on the flow matching path.

    Args:
        d_hidden (int): The hidden dimension of the encoder output.
        mel_channels (int): The number of mel-spectrogram channels.
        dropout_rate (float): The dropout rate.
    """
    def __init__(self, d_hidden: int, mel_channels: int, dropout_rate: float = 0.5):
        super().__init__()
        # The architecture is derived from the duration predictor in VITS/Matcha-TTS
        # as mentioned in the paper's appendix.
        self.conv1 = nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1)
        self.layernorm1 = nn.LayerNorm(d_hidden)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1)
        self.layernorm2 = nn.LayerNorm(d_hidden)
        self.dropout2 = nn.Dropout(dropout_rate)

        # The final projection layer outputs the scaled mel (X_h), the time (t_h),
        # and the log variance (log_sigma_sq_h).
        self.proj = nn.Linear(d_hidden, mel_channels + 2)
        self.mel_channels = mel_channels  # 保存 mel_channels 参数

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SFM Head.

        Args:
            x (torch.Tensor): Hidden states from the encoder of shape [B, T, C].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - X_h (torch.Tensor): Scaled mel-spectrogram [B, mel_channels, T].
                - t_h (torch.Tensor): Predicted time [B, 1].
                - log_sigma_sq_h (torch.Tensor): Predicted log variance [B, 1].
        """
        # The input shape is [B, T, C], but Conv1d expects [B, C, T].
        x = x.transpose(1, 2)

        x = self.conv1(x)
        # LayerNorm expects [B, T, C], so we need to transpose back and forth.
        x = self.layernorm1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.layernorm2(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = self.dropout2(x)

        # Transpose back to [B, T, C] for the final Linear projection.
        x = x.transpose(1, 2)
        x = self.proj(x)

        # Split the output into X_h, t_h, and log_sigma_sq_h along the channel dimension.
        x_h, t_h, log_sigma_sq_h = torch.split(x, [self.mel_channels, 1, 1], dim=-1)

        # As per the paper, t_h and log_sigma_sq_h are single scalar values.
        # We apply sigmoid to t_h to ensure it's in (0, 1) and then average over time.
        t_h = torch.sigmoid(t_h).mean(dim=1)
        # We also average the log variance over time.
        log_sigma_sq_h = log_sigma_sq_h.mean(dim=1)

        # The output X_h needs to be in [B, C, T] format.
        return x_h.transpose(1, 2), t_h, log_sigma_sq_h
