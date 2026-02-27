"""surgery_needle package

This module registers the package tasks when imported.
"""

# Expose tasks package so importing `surgery_needle` triggers registration side-effects
from . import tasks  # noqa: F401

# Register gymnasium ids so scripts that call `gym.make("SurgeryNeedleReach-v0", cfg=...)`
# can find this environment. We register the ManagerBasedRLEnv entry-point so
# Isaac Lab's `parse_env_cfg` can locate `env_cfg_entry_point` in the kwargs.
from gymnasium.envs.registration import register

try:
    register(
        id="SurgeryNeedleReach-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "surgery_needle.tasks.manager_based.surgery_needle.surgery_needle_env_cfg:SurgeryNeedleEnvCfg",
            "rl_games_cfg_entry_point": "surgery_needle.tasks.manager_based.surgery_needle.agents:rl_games_ppo_cfg.yaml",
        },
    )
    # Alias without explicit version (useful when users pass `--task=SurgeryNeedleReach`).
    register(
        id="SurgeryNeedleReach",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "surgery_needle.tasks.manager_based.surgery_needle.surgery_needle_env_cfg:SurgeryNeedleEnvCfg",
            "rl_games_cfg_entry_point": "surgery_needle.tasks.manager_based.surgery_needle.agents:rl_games_ppo_cfg.yaml",
        },
    )
except Exception:
    # registration can fail in some import-time scenarios (tests, repeated imports)
    # which is harmless — the env can still be constructed directly via the factory.
    pass
