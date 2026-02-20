import torch
from typing import Callable, Optional


# ============================================================
# Linear Interpolant (F5-TTS)
# x_t = (1 - t) x0 + t x1
# ============================================================

def alpha(t):
    return 1.0 - t


def sigma(t):
    return t


def velocity_to_x0(x, velocity, t):
    """
    Tweedie estimator for linear interpolant:
        x0 = x_t - t * u_t
    """
    while len(t.shape) < len(x.shape):
        t = t.unsqueeze(-1)

    return x - t * velocity


def velocity_to_score(x, velocity, t, eps=1e-5):
    """
    Convert velocity → score ∇ log p_t(x)
    Linear interpolant closed-form.
    """
    while len(t.shape) < len(x.shape):
        t = t.unsqueeze(-1)

    s = torch.clamp(sigma(t), min=eps)
    a = alpha(t)

    score = (a / s) * velocity - (x / s)
    return score


# ============================================================
# Diffusion / Drift (Scheduler Logic)
# ============================================================

def get_diffuse(t,
                sample_method="ode",
                schedule="linear",
                diffusion_norm=0.03):
    if sample_method == "ode":
        return torch.zeros_like(t)

    if schedule == "linear":
        g = diffusion_norm * t
    elif schedule == "sigma":
        g = diffusion_norm * sigma(t)
    elif schedule == "square":
        g = diffusion_norm * (t ** 2)
    elif schedule == "constant":
        g = torch.ones_like(t) * diffusion_norm
    else:
        raise ValueError("Unknown diffusion schedule")

    return g


def get_drift(x,
              velocity,
              t,
              sample_method="ode",
              schedule="linear",
              diffusion_norm=0.03):

    if sample_method == "ode":
        return velocity

    # SDE case
    score = velocity_to_score(x, velocity, t)
    g = get_diffuse(t,
                    sample_method="sde",
                    schedule=schedule,
                    diffusion_norm=diffusion_norm)

    while len(g.shape) < len(x.shape):
        g = g.unsqueeze(-1)

    drift = velocity - 0.5 * (g ** 2) * score
    return drift


# ============================================================
# Single Step
# ============================================================

def step(x,
         velocity,
         t,
         dt,
         sample_method="ode",
         schedule="linear",
         diffusion_norm=0.03):

    drift = get_drift(x, velocity, t,
                      sample_method=sample_method,
                      schedule=schedule,
                      diffusion_norm=diffusion_norm)

    while len(dt.shape) < len(x.shape):
        dt = dt.unsqueeze(-1)

    x_next = x + drift * dt

    if sample_method == "sde":
        g = get_diffuse(t,
                        sample_method="sde",
                        schedule=schedule,
                        diffusion_norm=diffusion_norm)

        while len(g.shape) < len(x.shape):
            g = g.unsqueeze(-1)

        noise = torch.randn_like(x)
        x_next = x_next + g * torch.sqrt(torch.abs(dt)) * noise

    return x_next


# ============================================================
# Particle Filtering Utilities
# ============================================================

def _resample(x, reward_fn, B, N, temperature=1.0):

    rewards = reward_fn(x)  # (B*N,)
    rewards = rewards.view(B, N)

    weights = torch.softmax(rewards / temperature, dim=1)

    new_particles = []
    for b in range(B):
        idx = torch.multinomial(weights[b], N, replacement=True)
        base = b * N
        new_particles.append(x[base + idx])

    return torch.cat(new_particles, dim=0)


def _select_best(x, reward_fn, B, N):

    rewards = reward_fn(x).view(B, N)
    best_idx = rewards.argmax(dim=1)

    selected = []
    for b in range(B):
        base = b * N
        selected.append(x[base + best_idx[b]])

    return torch.stack(selected, dim=0)


# ============================================================
# Corrector (Identity Default)
# ============================================================

class IdentityCorrector:

    def pre_correct(self, x, x0, velocity, step):
        return x, velocity

    def post_correct(self, x, x0, velocity, step):
        return x, x0, velocity

    def final_correct(self, x):
        return x


# ============================================================
# Block-wise Particle Sampler
# ============================================================

@torch.no_grad()
def particle_sample(
    fn: Callable,
    y0: torch.Tensor,
    timesteps: torch.Tensor,
    reward_fn: Optional[Callable] = None,
    n_particles: int = 4,
    sample_method: str = "ode",
    schedule: str = "linear",
    diffusion_norm: float = 0.03,
    block_size: int = 4,
    temperature: float = 1.0,
    corrector: Optional[object] = None,
):

    if corrector is None:
        corrector = IdentityCorrector()

    B = y0.shape[0]
    x = y0.repeat_interleave(n_particles, dim=0)

    total_steps = len(timesteps) - 1
    step_idx = 0

    while step_idx < total_steps:

        # -------- Block --------
        for _ in range(block_size):

            if step_idx >= total_steps:
                break

            t = timesteps[step_idx]
            t_next = timesteps[step_idx + 1]
            dt = t_next - t

            velocity = fn(t, x)

            x0 = velocity_to_x0(x, velocity, t)

            x, velocity = corrector.pre_correct(
                x, x0, velocity, step_idx
            )

            x = step(x,
                     velocity,
                     t,
                     dt,
                     sample_method=sample_method,
                     schedule=schedule,
                     diffusion_norm=diffusion_norm)

            x, x0, velocity = corrector.post_correct(
                x, x0, velocity, step_idx
            )

            step_idx += 1

        # -------- Resample after block --------
        if reward_fn is not None and step_idx < total_steps:
            x = _resample(x, reward_fn, B, n_particles, temperature)

    # -------- Final Selection --------
    if reward_fn is not None:
        x = _select_best(x, reward_fn, B, n_particles)

    x = corrector.final_correct(x)

    return x
