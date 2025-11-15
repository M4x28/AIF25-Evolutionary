"""Centralized configuration for Walker QD training jobs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from .logging import debug_log
from .utils.io import ensure_dir


@dataclass(frozen=True)
class WalkerBaseConfig:
    """Shared knobs consumed by every training algorithm."""

    # --- Environment -------------------------------------------------
    env_id: str = "Walker2d-v5"  # Gymnasium environment where all agents are trained / evaluated.
    seed: int = 42  # Global RNG seed propagated to numpy, gym reset, etc., for reproducible runs.
    exclude_current_positions_from_observation: bool = True  # Remove absolute positions to match MuJoCo defaults.
    phases: int = 6  # Number of noise phases; at each phase a new Gaussian noise vector is sampled.
    iterations_per_phase: int = 100  # Inner-loop iterations (generations) executed per phase.
    max_episode_steps: int = 600  # Safety truncation when rolling out the policy inside the simulator.

    # --- Render Best Model -----------------------------------------------
    render_fps: int = 30  # FPS used when exporting rollout videos.
    video_seconds: int = 20  # Duration of exported videos (identical for all algorithms).
    allow_save_best_model_video: bool = True  # Toggle video export for best solutions.

    # --- Reward Weights ---------------------------------------------------------
    forward_reward_weight: float = 2.0  # MuJoCo reward coefficient that scales forward velocity.

    # --- Noise schedule ----------------------------------------------------------
    noise_std: float = 0.15  # Standard deviation of Gaussian observation noise (var = noise_std**2).

    # --- Parallelization ---------------------------------------------------------
    workers: int = 0  # Parallel worker count shared across Dask / multiprocessing pools (0 => means use max number of workers).

    # --- Output --------------------------------------------------------
    out_dir: str = "runs_walker2d"  # Root folder where checkpoints / videos / stats will be stored.
    best_params_filename: str = "best_solution.npy"  # Filename used when persisting the best parameters.
    phase_best_params_filename: str = "phase_best_solution.npy"  # Filename for per-phase best solution.
    reward_curve_filename: str = "reward_curve.png"  # Filename used for fitness/reward plots.
    best_video_filename: str = "best_model.mp4"  # Filename for recorded rollout videos.

    def __post_init__(self) -> None:
        debug_log(
            "WalkerBaseConfig instantiated",
            env=self.env_id,
            out_dir=self.out_dir,
            noise_std=self.noise_std,
            workers=self.workers,
            phases=self.phases,
            iterations_per_phase=self.iterations_per_phase,
        )


@dataclass(frozen=True)
class CMAMEConfig(WalkerBaseConfig):
    """Hyper-parameters specific to the CMA-ME trainer."""

    # --- CMA-ME emitters ---------------------------------------------------------
    num_emitters: int = 12  # Number of CMA-ME emitters exploring the archive concurrently.
    emitter_batch: int = 50  # Solutions proposed per emitter per iteration.
    sigma0: float = 0.5  # Initial Gaussian exploration scale for emitter search distributions.

    # --- Archive layout ----------------------------------------------------------
    archive_bins: Tuple[int, int] = (128, 128)  # MAP-Elites grid resolution along the two measures.
    forward_velocity_range: Tuple[float, float] = (0.0, 10.0)  # Bounds for velocity axis (m/s).
    torso_height_range: Tuple[float, float] = (0.8, 1.7)  # Bounds for torso height axis (meters).
    qd_score_offset: float = -500.0  # Baseline shift used when converting rewards to QD scores.
    archive_measure_labels: Tuple[str, str] = ("Forward velocity", "Torso height")  # Human-readable axis labels.

    # --- Visualization -----------------------------------------------------------
    out_dir: str = "runs_mapelites_walker2d"  # Default run directory for CMA-ME experiments.
    heatmap_vmin: float = -500.0  # Lower color scale bound for archive heatmaps.
    heatmap_vmax: float = 1500.0  # Upper color scale bound for archive heatmaps.

    def __post_init__(self) -> None:
        super().__post_init__()
        debug_log(
            "CMAMEConfig instantiated",
            env=self.env_id,
            out_dir=self.out_dir,
            emitters=self.num_emitters,
        )


@dataclass(frozen=True)
class CMAESConfig(WalkerBaseConfig):
    """Hyper-parameters for the CMA-ES baseline trainer."""

    # --- Controller topology -----------------------------------------------------
    hidden_size: int = 32  # Neurons in each hidden layer of the MLP controller.

    # --- CMA-ES settings ---------------------------------------------------------
    sigma0: float = 0.1  # Initial CMA-ES sampling stddev for policy parameters.
    pop_size: int = 200  # CMA-ES population size (evaluations per generation).

    # --- Outputs -----------------------------------------------------------------
    out_dir: str = "runs_cmaes_walker2d"  # Default output folder for CMA-ES runs.

    def __post_init__(self) -> None:
        super().__post_init__()
        debug_log(
            "CMAESConfig instantiated",
            env=self.env_id,
            out_dir=self.out_dir,
            pop_size=self.pop_size,
        )


@dataclass(frozen=True)
class NEATConfig(WalkerBaseConfig):
    """Hyper-parameters for the NEAT baseline trainer."""

    # --- Evolutionary setup ------------------------------------------------------
    pop_size: int = 500  # NEAT population size (genomes per generation).
    neat_config_path: str = "configs/neat_walker2d.cfg"  # Path to the neat-python config file maintained manually.
    fitness_criterion: str = "max"  # objective used by neat-python to rank genomes.
    fitness_threshold: int = 9_999_999  # early-stop once a genome reaches this fitness.
    reset_on_extinction: bool = False  # whether to reinitialize population if all species die out.

    # --- Genome activation / aggregation ----------------------------------------
    activation_default: str = "tanh"  # default activation for newly created nodes.
    activation_mutate_rate: float = 0.1  # probability of mutating activation function.
    activation_options: str = "tanh"  # whitelist of activation functions.
    aggregation_default: str = "sum"  # aggregates incoming signals (sum / mean / max).
    aggregation_mutate_rate: float = 0.0  # probability aggregation function mutates.
    aggregation_options: str = "sum"  # allowed aggregation operations.

    # --- Bias settings -----------------------------------------------------------
    bias_init_mean: float = 0.0
    bias_init_stdev: float = 1.0
    bias_max_value: float = 30.0
    bias_min_value: float = -30.0
    bias_mutate_power: float = 0.5
    bias_mutate_rate: float = 0.4
    bias_replace_rate: float = 0.1

    # --- Compatibility (speciation) ---------------------------------------------
    compatibility_disjoint_coefficient: float = 1.0
    compatibility_weight_coefficient: float = 0.5

    # --- Connection mutation toggles --------------------------------------------
    conn_add_prob: float = 0.1
    conn_delete_prob: float = 0.1
    enabled_default: bool = True
    enabled_mutate_rate: float = 0.01

    # --- Structural mutation ----------------------------------------------------
    feed_forward: bool = True
    initial_connection: str = "full_direct"
    node_add_prob: float = 0.1
    node_delete_prob: float = 0.1
    num_hidden: int = 0

    # --- Response parameters (rarely used) --------------------------------------
    response_init_mean: float = 1.0
    response_init_stdev: float = 0.0
    response_max_value: float = 30.0
    response_min_value: float = -30.0
    response_mutate_power: float = 0.0
    response_mutate_rate: float = 0.0
    response_replace_rate: float = 0.0

    # --- Weight initialization / mutation ---------------------------------------
    weight_init_mean: float = 0.0
    weight_init_stdev: float = 1.0
    weight_max_value: float = 30.0
    weight_min_value: float = -30.0
    weight_mutate_power: float = 0.5
    weight_mutate_rate: float = 0.8
    weight_replace_rate: float = 0.1

    # --- Species control --------------------------------------------------------
    species_compatibility_threshold: float = 3.0
    max_stagnation: int = 15
    species_elitism: int = 2

    # --- Reproduction -----------------------------------------------------------
    reproduction_elitism: int = 2
    survival_threshold: float = 0.2

    # --- Outputs -----------------------------------------------------------------
    out_dir: str = "runs_neat_walker2d"  # Default output folder for NEAT runs.

    def __post_init__(self) -> None:
        super().__post_init__()
        debug_log(
            "NEATConfig instantiated",
            env=self.env_id,
            out_dir=self.out_dir,
            pop_size=self.pop_size,
        )


def _bool_literal(value: bool) -> str:
    """Serialize booleans in the format expected by neat-python."""
    return "True" if value else "False"


def render_neat_config(cfg: NEATConfig, num_inputs: int, num_outputs: int) -> str:
    """Convert NEATConfig fields into a neat-python configuration file."""
    return f"""[NEAT]
fitness_criterion     = {cfg.fitness_criterion}
fitness_threshold     = {cfg.fitness_threshold}
pop_size              = {cfg.pop_size}
reset_on_extinction   = {_bool_literal(cfg.reset_on_extinction)}

[DefaultGenome]
activation_default      = {cfg.activation_default}
activation_mutate_rate  = {cfg.activation_mutate_rate}
activation_options      = {cfg.activation_options}
aggregation_default     = {cfg.aggregation_default}
aggregation_mutate_rate = {cfg.aggregation_mutate_rate}
aggregation_options     = {cfg.aggregation_options}

bias_init_mean          = {cfg.bias_init_mean}
bias_init_stdev         = {cfg.bias_init_stdev}
bias_max_value          = {cfg.bias_max_value}
bias_min_value          = {cfg.bias_min_value}
bias_mutate_power       = {cfg.bias_mutate_power}
bias_mutate_rate        = {cfg.bias_mutate_rate}
bias_replace_rate       = {cfg.bias_replace_rate}

compatibility_disjoint_coefficient = {cfg.compatibility_disjoint_coefficient}
compatibility_weight_coefficient   = {cfg.compatibility_weight_coefficient}

conn_add_prob           = {cfg.conn_add_prob}
conn_delete_prob        = {cfg.conn_delete_prob}
enabled_default         = {_bool_literal(cfg.enabled_default)}
enabled_mutate_rate     = {cfg.enabled_mutate_rate}

feed_forward            = {_bool_literal(cfg.feed_forward)}
initial_connection      = {cfg.initial_connection}
node_add_prob           = {cfg.node_add_prob}
node_delete_prob        = {cfg.node_delete_prob}
num_hidden              = {cfg.num_hidden}
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}

response_init_mean      = {cfg.response_init_mean}
response_init_stdev     = {cfg.response_init_stdev}
response_max_value      = {cfg.response_max_value}
response_min_value      = {cfg.response_min_value}
response_mutate_power   = {cfg.response_mutate_power}
response_mutate_rate    = {cfg.response_mutate_rate}
response_replace_rate   = {cfg.response_replace_rate}

weight_init_mean        = {cfg.weight_init_mean}
weight_init_stdev       = {cfg.weight_init_stdev}
weight_max_value        = {cfg.weight_max_value}
weight_min_value        = {cfg.weight_min_value}
weight_mutate_power     = {cfg.weight_mutate_power}
weight_mutate_rate      = {cfg.weight_mutate_rate}
weight_replace_rate     = {cfg.weight_replace_rate}

[DefaultSpeciesSet]
compatibility_threshold = {cfg.species_compatibility_threshold}

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = {cfg.max_stagnation}
species_elitism      = {cfg.species_elitism}

[DefaultReproduction]
elitism            = {cfg.reproduction_elitism}
survival_threshold = {cfg.survival_threshold}
"""


def ensure_neat_config_file(cfg: NEATConfig, num_inputs: int, num_outputs: int) -> Path:
    """Persist the neat config file (overwriting to keep in sync) and return its path."""
    cfg_path = Path(cfg.neat_config_path)
    ensure_dir(str(cfg_path.parent))
    cfg_path.write_text(render_neat_config(cfg, num_inputs, num_outputs), encoding="utf-8")
    debug_log("NEAT config file rendered", path=str(cfg_path))
    return cfg_path
