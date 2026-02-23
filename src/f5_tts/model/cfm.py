"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
import random
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)

import math

class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):

            if cfg_strength < 1e-5:
                v = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
            else:
                pred_cfg = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    cfg_infer=True,
                    cache=True,
                )
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                v = pred + (pred - null_pred) * cfg_strength

            # DEBUG
            # print("t:", t.item() if torch.numel(t)==1 else t)
            # print("mean|x|", x.abs().mean().item())
            # print("mean|v|", v.abs().mean().item())

            return v

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0
        # use_epss=True
        print("???", use_epss, t_start)
        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        # print(trajectory.shape)
        # print("debug", trajectory)
        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred

class CFM_SDE(CFM):

    def __init__(
        self,
        *args,
        sample_method="sde",      # "sde" | "ode"
        diffusion_norm=0.03,
        diffusion_schedule="linear",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.sample_method = sample_method
        self.diffusion_norm = diffusion_norm
        self.diffusion_schedule = diffusion_schedule

    def convert_velocity_to_score(self, velocity, t, sample):

        orig_dtype = sample.dtype

        velocity = velocity.float()
        sample = sample.float()

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=sample.device, dtype=torch.float32)
        else:
            t = t.float()

        while t.ndim < sample.ndim:
            t = t.unsqueeze(-1)

        var = torch.clamp(t, min=1e-3)  # IMPORTANT luongnt29 need to check later

        reverse_alpha_ratio = -(1 - t)

        score = (reverse_alpha_ratio * velocity - sample) / var

        return score.to(orig_dtype)

    def get_diffuse(self, t):

        if self.sample_method == "ode":
            return torch.zeros_like(t)

        if self.diffusion_schedule == "linear":
            return self.diffusion_norm * (1-t)

        if self.diffusion_schedule == "square":
            return self.diffusion_norm * ((1-t) ** 2)

        if self.diffusion_schedule == "constant":
            return torch.ones_like(t) * self.diffusion_norm

        raise ValueError



    def reward(self, x0, ref_audio_len):
        import requests
        import torch

        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)

        B = x0.size(0)
        device = x0.device

        rewards = []

        for i in range(B):

            mel = x0[i].detach().cpu().float().tolist()

            response = requests.post(
                "http://localhost:8000/infer",
                json={
                    "mel": mel,
                    "ref_len": ref_audio_len 
                },
                timeout=10,
            )

            if response.status_code != 200:
                raise RuntimeError(response.text)

            mos = response.json()["mos"]
            rewards.append(mos)

        return torch.tensor(rewards, device=device, dtype=x0.dtype)


    @torch.no_grad()
    def sample(
        self,
        cond,
        text,
        duration,
        *,
        lens=None,
        steps=8,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed=None,
        max_duration=4096,
        vocoder=None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):

        self.eval()
        # steps=8

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        ref_len = cond.shape[-1] // 256
        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device

        if lens is None:
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        if isinstance(text, list):
            if self.vocab_char_map is not None:
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)

        cond_mask = lens_to_mask(lens)

        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1,
            duration,
        )

        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        cond = torch.nn.functional.pad(
            cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0
        )

        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = torch.nn.functional.pad(
            cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False
        )
        cond_mask = cond_mask.unsqueeze(-1)

        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None


        def fn(t, x):

            if cfg_strength < 1e-5:
                v = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
            else:
                pred_cfg = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    cfg_infer=True,
                    cache=True,
                )
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                v = pred + (pred - null_pred) * cfg_strength

            # print("time t:", t.item() if torch.numel(t) == 1 else t)
            # print("mean|x|", x.abs().mean().item())
            # print("mean|v|", v.abs().mean().item())

            return v

        y0 = []
        for dur in duration:
            if seed is not None:
                torch.manual_seed(seed)
            y0.append(
                torch.randn(
                    dur,
                    self.num_channels,
                    device=device,
                    dtype=step_cond.dtype,
                )
            )

        y0 = torch.nn.utils.rnn.pad_sequence(
            y0, padding_value=0, batch_first=True
        )

        t_start = 0

        if t_start == 0 and use_epss:
            t = get_epss_timesteps(
                steps, device=device, dtype=step_cond.dtype
            )
            print("EPSS", t)
        else:
            t = torch.linspace(
                t_start, 1, steps + 1,
                device=device,
                dtype=step_cond.dtype
            )

        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (
                torch.cos(torch.pi / 2 * t) - 1 + t
            )

        #  RBF 

        x_s = y0
        r_star = self.reward(x_s, ref_len)

        M = len(t) - 1
        branch_num = 4
        Q = [branch_num] * M  # quota per step

        trajectory = [x_s]

        for i in range(M):

            s = t[i]
            s_next = t[i + 1]
            dt = s_next - s
            q = Q[i]

            best_local_reward = -1
            best_local_sample = None

            for j in range(1, q + 1):

                v = fn(s, x_s)

                drift = v

                if self.sample_method == "sde":
                    score = self.convert_velocity_to_score(v, s, x_s)
                    g = self.get_diffuse(s)
                    while g.ndim < x_s.ndim:
                        g = g.unsqueeze(-1)
                    drift = v - 0.5 * (g ** 2) * score

                x_candidate = x_s + drift * dt

                if self.sample_method == "sde":
                    g = self.get_diffuse(s)
                    while g.ndim < x_s.ndim:
                        g = g.unsqueeze(-1)
                    noise = torch.randn_like(x_s)
                    x_candidate = x_candidate + g * torch.sqrt(
                        torch.clamp(dt, min=1e-8)
                    ) * noise

                r_candidate = self.reward(x_candidate, ref_len)
                print(f"Step {i} - Particle {j} - Reward:", r_candidate.item())
                if r_candidate > r_star:

                    if i + 1 < M:
                        Q[i + 1] += (Q[i] - j)

                    r_star = r_candidate
                    x_s = x_candidate
                    break

                if r_candidate > best_local_reward:
                    best_local_reward = r_candidate
                    best_local_sample = x_candidate

                if j == q:
                    x_s = best_local_sample

            trajectory.append(x_s)

        trajectory = torch.stack(trajectory, dim=0)

        self.transformer.clear_cache()


        sampled = trajectory[-1]
        print("sampled", sampled.shape)
        out = torch.where(cond_mask, cond, sampled)
        print("out", out.shape)
        if vocoder is not None:
            out = out.permute(0, 2, 1)
            out = vocoder(out)
        return out, trajectory