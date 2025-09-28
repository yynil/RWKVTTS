import torch
import torch.nn.functional as F
from matcha.models.components.flow_matching import BASECFM
from cosyvoice.utils.common import set_all_random_seed


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.sigma_min = cfm_params.get('sigma_min', 1e-6)
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=torch.zeros(1, 80, 0, 2), sfm_inputs=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
            sfm_inputs (Dict, optional): Dictionary containing SFM parameters for inference.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        # --- SFM Inference ---
        if sfm_inputs is not None:
            # 1. Get inputs from SFM head
            x_h = sfm_inputs['x_h'] # [B, C, T]
            t_h = sfm_inputs['t_h'] # [B, 1]
            sigma_h = sfm_inputs['sigma_h'] # [B, 1]
            alpha = sfm_inputs.get('alpha', 1.0) # SFM strength

            # x_h is based on token length (from encoder hidden states), mu is based on mel length
            # token_mel_ratio = 2, so x_h has token length while mu has mel length (2x token length)
            token_mel_ratio = 2
            expected_x_h_time = mu.size(2) // token_mel_ratio
            
            if x_h.size(2) != expected_x_h_time:
                # Interpolate x_h to match the expected token-based time dimension
                x_h = torch.nn.functional.interpolate(x_h, size=expected_x_h_time, mode='linear', align_corners=False)
            
            # Now expand x_h to match mu's time dimension for the computation
            x_h_expanded = torch.nn.functional.interpolate(x_h, size=mu.size(2), mode='linear', align_corners=False)

            # 2. Construct the intermediate state with SFM strength (Eq. 22)
            delta = torch.maximum(alpha * ((1 - self.sigma_min) * t_h + sigma_h), torch.tensor(1.0, device=mu.device))

            x_h_bar = (alpha / delta).unsqueeze(-1) * x_h_expanded
            t_h_bar = (alpha / delta) * t_h
            sigma_sq_h_bar = (alpha**2 / delta**2) * (sigma_h**2)

            # Add noise X_0 (Eq. 15)
            x0 = torch.randn_like(mu) * temperature
            noise_scaler_sq = torch.maximum((1 - (1 - self.sigma_min) * t_h_bar)**2 - sigma_sq_h_bar, torch.tensor(0.0, device=mu.device))
            noise_scaler = torch.sqrt(noise_scaler_sq).unsqueeze(-1)
            x_t_h = noise_scaler * x0 + x_h_bar

            # 3. Use an ODE solver to solve the integral from t_h_bar to 1
            start_time = t_h_bar[0].item()
            t_span = torch.linspace(start_time, 1, n_timesteps + 1, device=mu.device)
            
            # Fix: Use direct ODE solving without CFG processing to match compute_loss
            x = x_t_h
            for step in range(len(t_span) - 1):
                dt = t_span[step + 1] - t_span[step]
                t = t_span[step].unsqueeze(0)
                
                # Direct estimator call without CFG processing
                dphi_dt = self.estimator(x, mask, mu, t.squeeze(), spks, cond, streaming=False)
                x = x + dt * dphi_dt
            
            return x.float(), None

        # --- Standard Inference (from noise) ---
        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise or intermediate state
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t = t_span[0].unsqueeze(dim=0)
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        x_in = torch.zeros([2, x.size(1), x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2, x.size(1), x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2, self.spk_emb_dim], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2, x.size(1), x.size(2)], device=x.device, dtype=x.dtype)

        for step in range(len(t_span) - 1):
            dt = t_span[step + 1] - t_span[step]
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            if cond is not None:
                cond_in[0] = cond
            
            # The estimator time input t is now correctly handled by t_span
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                streaming
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t_span[step + 1].unsqueeze(dim=0)
            sol.append(x)

        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            # Placeholder for TRT inference logic if needed
            raise NotImplementedError("TRT inference is not implemented in this SFM version.")

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False, sfm_inputs=None):
        """Computes diffusion loss.

        If sfm_inputs is provided, computes the single-segment piecewise flow loss.
        """
        b, c, t_len = mu.shape

        # --- SFM Training (Single-segment piecewise flow) ---
        if sfm_inputs is not None:
            # 1. Get inputs from main module
            x_h_pred = sfm_inputs['x_h']       # [B, C, T]
            t_h_true = sfm_inputs['t_h'].view(b, 1, 1)
            sigma_sq_h_true = sfm_inputs['sigma_sq_h'].view(b, 1, 1)

            # 2. Construct the intermediate state X_t_h (Eq. 15)
            sigma_h_true = torch.sqrt(sigma_sq_h_true)
            delta = torch.maximum((1 - self.sigma_min) * t_h_true + sigma_h_true, torch.tensor(1.0, device=mu.device))
            x_h_bar = (1.0 / delta) * x_h_pred
            t_h_bar = (1.0 / delta) * t_h_true
            sigma_sq_h_bar = (1.0 / delta**2) * sigma_sq_h_true

            x0 = torch.randn_like(x1)
            noise_scaler_sq = torch.maximum((1 - (1 - self.sigma_min) * t_h_bar)**2 - sigma_sq_h_bar, torch.tensor(0.0, device=mu.device))
            noise_scaler = torch.sqrt(noise_scaler_sq)
            x_t_h = noise_scaler * x0 + x_h_bar

            # 3. Apply single-segment piecewise flow (Eq. 18 & 19)
            t_u = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype) * (1 - t_h_bar) + t_h_bar
            x_t = (1 - t_u) * x_t_h.detach() + t_u * (x1 + self.sigma_min * x0)
            
            # According to paper Eq. (19), the vector field U_t needs to be scaled by 1/(1-t_h)
            # This is the critical fix.
            t_h_for_scaling = t_h_true.view(b, 1, 1).detach()
            u_t = (1.0 / (1.0 - t_h_for_scaling + 1e-8)) * ((x1 + self.sigma_min * x0) - x_t_h.detach())

            # The actual time t passed to the estimator is rescaled to [t_h_bar, 1] (Eq. 17)
            t_s = (1 - t_h_bar) * t_u + t_h_bar

            # Classifier-Free Guidance during training
            if self.training_cfg_rate > 0:
                cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
                mu = mu * cfg_mask.view(-1, 1, 1)
                spks = spks * cfg_mask.view(-1, 1)
                if cond is not None:
                    cond = cond * cfg_mask.view(-1, 1, 1)

            pred = self.estimator(x_t, mask, mu, t_s.squeeze(), spks, cond, streaming=streaming)
            
            # CFM Loss (Eq. 20)
            loss_cfm = F.mse_loss(pred * mask, u_t * mask, reduction="sum") / (torch.sum(mask) * u_t.shape[1])
            
            # Mu Loss (Eq. 13) - 修复维度广播问题
            # x_h_pred: [B, C, T], t_h_true: [B, 1, 1], x1: [B, C, T]
            # 确保 t_h_true 正确广播到 x1 的维度
            loss_mu = F.mse_loss(x_h_pred, t_h_true * x1)
            
            # The total loss for SFM is L_cfm + L_mu
            return loss_cfm + loss_mu, x_t

        # --- Standard Training (from noise) ---
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            if cond is not None:
                cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y

class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        set_all_random_seed(0)
        self.rand_noise = torch.randn([1, self.n_feats, 50 * 300])

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming=False, sfm_inputs=None):
        if sfm_inputs is not None:
            # SFM inference path
            return super().forward(mu, mask, n_timesteps, temperature, spks, cond, sfm_inputs=sfm_inputs)

        # Original causal inference path
        z = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming), None
