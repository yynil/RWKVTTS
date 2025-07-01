import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice.utils.mask import make_pad_mask
# Import the new modules
from model.flow.sfm_head import SFMHead
from model.flow.flow_matching import CausalConditionalCFM


# New SFM-enabled class
class CausalMaskedDiffWithXvecSFM(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 vocab_size: int = 6561,
                 pre_lookahead_len: int = 3,
                 sfm_strength: float = 2.5, # Inference hyperparameter alpha
                 encoder: torch.nn.Module = None,
                 decoder: CausalConditionalCFM = None,
                 sfm_head: SFMHead = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.pre_lookahead_len = pre_lookahead_len
        self.sfm_strength = sfm_strength # For inference
        self.use_checkpoint = False  # Add gradient checkpointing flag

        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.encoder = encoder
        
        # This projects encoder hidden states to coarse mel-spectrograms (X_g)
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        
        # The new SFM head module
        self.sfm_head = sfm_head
        
        self.decoder = decoder

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        x1 = batch['speech_feat'].to(device) # This is X_1, the ground truth
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        streaming = True if random.random() < 0.5 else False

        embedding = F.normalize(embedding, dim=1)

        mask = (~make_pad_mask(token_len,max_len=token.shape[1])).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # 1. Generate coarse representations (H_g and X_g)
        if self.use_checkpoint and self.training:
            h_g, h_lengths = torch.utils.checkpoint.checkpoint(
                self._encoder_forward, token, token_len, streaming, use_reentrant=False
            )
        else:
            h_g, h_lengths = self.encoder(token, token_len, streaming=streaming) # h_g is H_g
        x_g = self.encoder_proj(h_g) # x_g is X_g, coarse mel

        # 2. SFM Head Prediction
        # The SFM head takes the final hidden states H_g as input
        if self.use_checkpoint and self.training:
            x_h_pred, t_h_pred, log_sigma_sq_h_pred = torch.utils.checkpoint.checkpoint(
                self._sfm_head_forward, h_g, use_reentrant=False
            )
        else:
            x_h_pred, t_h_pred, log_sigma_sq_h_pred = self.sfm_head(h_g)

        # 3. Compute Losses
        # 3.1 Coarse Mel Loss (L_coarse, Eq. 11)
        loss_mask = (~make_pad_mask(feat_len, max_len=x1.size(1))).unsqueeze(-1)
        loss_coarse = F.l1_loss(x_g * loss_mask, x1 * loss_mask)
        
        # 3.2 Orthogonal Projection to get target t_h, sigma_h (Eq. 13)
        with torch.no_grad():
            x_h_sg = x_h_pred.transpose(1, 2).detach() # [B, T, C]
            
            # Ensure correct shapes for broadcasting
            dot_product = torch.sum(x_h_sg * x1, dim=[1, 2])
            x1_norm_sq = torch.sum(x1 * x1, dim=[1, 2])
            t_h_true = (dot_product / (x1_norm_sq + 1e-8)).unsqueeze(1) # [B, 1]
            sigma_sq_h_true = torch.mean((x_h_sg - t_h_true.unsqueeze(2) * x1)**2, dim=[1, 2]).unsqueeze(1)
            
            # Numerical stability
            t_h_true = torch.clamp(t_h_true, min=0.0, max=1.0)
            sigma_sq_h_true = torch.clamp(sigma_sq_h_true, min=1e-7)

        # 3.3 SFM-specific losses (L_t, L_sigma, Eq. 16)
        loss_t = F.mse_loss(t_h_pred, t_h_true)
        loss_sigma = F.mse_loss(log_sigma_sq_h_pred, torch.log(sigma_sq_h_true))
        
        # 4. Flow Matching Loss (L_cfm + L_mu)
        cfm_mask = (~make_pad_mask(feat_len, max_len=x1.size(1))).unsqueeze(1)
        loss_cfm_and_mu, _ = self.decoder.compute_loss(
            x1=x1.transpose(1, 2).contiguous(),
            mask=cfm_mask,
            mu=x_g.transpose(1, 2).contiguous(),
            spks=embedding,
            sfm_inputs={
                'x_h': x_h_pred.contiguous(),
                't_h': t_h_true,
                'sigma_sq_h': sigma_sq_h_true
            },
            streaming=streaming,
        )

        # Total loss (Eq. 21)
        total_loss = loss_coarse + loss_t + loss_sigma + loss_cfm_and_mu
        return {'loss': total_loss, 'loss_coarse': loss_coarse, 'loss_t': loss_t, 'loss_sigma': loss_sigma, 'loss_cfm_mu': loss_cfm_and_mu}

    def _encoder_forward(self, token, token_len, streaming):
        """Helper method for gradient checkpointing"""
        return self.encoder(token, token_len, streaming=streaming)
    
    def _sfm_head_forward(self, h_g):
        """Helper method for gradient checkpointing"""
        return self.sfm_head(h_g)

    @torch.inference_mode()
    def inference(self,
                  token, token_len,
                  prompt_token, prompt_token_len,
                  prompt_feat, prompt_feat_len,
                  embedding, streaming, finalize):
        assert token.shape[0] == 1
        embedding = F.normalize(embedding, dim=1)

        # Concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # Encoder forward pass
        if finalize:
            h_g, h_lengths = self.encoder(token, token_len, streaming=streaming)
        else:
            token_main, context = token[:, :-self.pre_lookahead_len], token[:, -self.pre_lookahead_len:]
            h_g, h_lengths = self.encoder(token_main, token_len, context=context, streaming=streaming)
        
        x_g = self.encoder_proj(h_g)
        mel_len1 = prompt_feat.shape[1]
        
        # Calculate feat_len based on the concatenated token length
        # Fix: feat_len should be token_len * ratio_feat_ratio (typically 2)
        ratio_feat_ratio = 2  # This should match the training configuration
        feat_len = token_len * ratio_feat_ratio
        
        # Get SFM head predictions
        x_h, t_h, log_sigma_sq_h = self.sfm_head(h_g)
        sigma_h = torch.exp(log_sigma_sq_h * 0.5)

        # The decoder now handles the SFM logic
        feat, _ = self.decoder(
            mu=x_g.transpose(1, 2).contiguous(),
            mask=(~make_pad_mask(feat_len)).unsqueeze(1),
            spks=embedding,
            n_timesteps=10,
            streaming=streaming,
            sfm_inputs={
                'x_h': x_h,
                't_h': t_h,
                'sigma_h': sigma_h,
                'alpha': self.sfm_strength # Pass SFM strength alpha
            }
        )
        
        feat = feat[:, :, mel_len1:]
        return feat.float(), None