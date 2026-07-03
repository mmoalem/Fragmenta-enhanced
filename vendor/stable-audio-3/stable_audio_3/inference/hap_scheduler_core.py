# hap_scheduler_core.py
# MD-HAP -- Hamiltonian Action-Principle Sigma Schedule
# Physics-based sigma scheduler for flow-matching diffusion models.
#
# (c) 2026 Alexander Allan (MDMAchine) | A&E Concepts
# GPL v3 -- https://github.com/MDMAchine/MD-HAP-Scheduler
#
# Version: 1.0.0
# Retrieved: 2026-07-03 from upstream

import torch

CONST_EPSILON = 1e-6


def calculate_hap_sigmas(steps, damping_friction, kinetic_energy, sigma_max, sigma_min):
    steps = max(1, steps)
    sigma_max = max(sigma_max, 0.01)
    safe_sigma_min = max(sigma_min, 0.0)

    t = torch.linspace(0.0, 1.0, steps, dtype=torch.float32)
    velocity = (1.0 + kinetic_energy * t) * torch.exp(-damping_friction * t)
    velocity = torch.clamp(velocity, min=CONST_EPSILON)
    distance = torch.cumsum(velocity, dim=0)
    distance = torch.cat([torch.tensor([0.0], dtype=torch.float32), distance])
    distance_norm = distance / distance[-1]
    sigmas = safe_sigma_min + (sigma_max - safe_sigma_min) * (1.0 - distance_norm)
    sigmas[0]  = sigma_max
    sigmas[-1] = 0.0 if safe_sigma_min == 0.0 else safe_sigma_min

    mid_idx  = steps // 2
    log_data = (
        f"HAP_POTENTIAL_WELL | "
        f"Damping: {damping_friction:.2f} | "
        f"Kinetic: {kinetic_energy:.2f} | "
        f"Mid-Sigma: {sigmas[mid_idx].item():.4f}"
    )
    return sigmas, log_data
