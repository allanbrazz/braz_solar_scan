from __future__ import annotations

"""Default distributions for a Hopwood-style fully synthetic IV generator.

Hopwood et al. generate fully synthetic IV curves by sampling *multipliers* over
single-diode parameters (Iph, I0, Rs, Rsh, n) for each class, using truncated
Gaussian distributions and (optionally) Latin Hypercube Sampling.

These defaults are intentionally conservative so you can get an end-to-end
pipeline running in this repository. Tune them to match your module technology
and your target fault taxonomy.

The structure below:
  - each class has distribution parameters for each multiplier
  - if a multiplier is omitted, it is assumed to be 1.0

Distribution spec:
  {"dist": "truncnorm", "mu": ..., "sigma": ..., "lo": ..., "hi": ...}

All multipliers are unitless.
"""

DEFAULT_N_POINTS: int = 200

# Irradiance (W/m²) and cell temperature (°C) sampling ranges for the synthetic generator.
# (Hopwood simulate multiple conditions and translate to STC; here we keep the same idea.)
DEFAULT_ENV_RANGES = {
    "g_poa_lo": 200.0,
    "g_poa_hi": 1100.0,
    "tcell_lo": 10.0,
    "tcell_hi": 70.0,
}

# Failure class definitions.
# NOTE: Values reflect the narrative you provided (baseline, partial soiling, cell cracks).
DEFAULT_FAILURE_CLASSES = {
    "normal": {
        "weight": 1.0,
        # Natural variability: small increase in Rs
        "rs_mult": {"dist": "truncnorm", "mu": 1.05, "sigma": 0.10, "lo": 0.80, "hi": 1.40},
        "rsh_mult": {"dist": "truncnorm", "mu": 1.00, "sigma": 0.10, "lo": 0.60, "hi": 1.60},
        "iph_mult": {"dist": "truncnorm", "mu": 1.00, "sigma": 0.05, "lo": 0.85, "hi": 1.10},
        "n_mult": {"dist": "truncnorm", "mu": 1.00, "sigma": 0.03, "lo": 0.90, "hi": 1.10},
        "i0_mult": {"dist": "truncnorm", "mu": 1.00, "sigma": 0.15, "lo": 0.50, "hi": 2.50},
    },
    "partial_soiling": {
        "weight": 1.0,
        # Reduced photocurrent in a portion of the module/string
        "iph_mult": {"dist": "truncnorm", "mu": 0.60, "sigma": 0.07, "lo": 0.33, "hi": 0.95},
        "rs_mult": {"dist": "truncnorm", "mu": 1.00, "sigma": 0.08, "lo": 0.70, "hi": 1.50},
        "rsh_mult": {"dist": "truncnorm", "mu": 1.00, "sigma": 0.15, "lo": 0.50, "hi": 2.50},
    },
    "cell_cracks": {
        "weight": 1.0,
        # Increased series resistance and reduced shunt
        "rs_mult": {"dist": "truncnorm", "mu": 1.30, "sigma": 0.50, "lo": 0.80, "hi": 4.00},
        "rsh_mult": {"dist": "truncnorm", "mu": 0.50, "sigma": 0.60, "lo": 0.05, "hi": 1.50},
        "iph_mult": {"dist": "truncnorm", "mu": 0.95, "sigma": 0.06, "lo": 0.70, "hi": 1.05},
    },
}
