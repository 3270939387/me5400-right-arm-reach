"""MDP helper functions for the SurgeryNeedleReach task.

These are minimal, clear implementations that you should adapt to the real
asset API (prim queries and transforms). The functions return tensors used
by the environment's observation/reward/termination terms.

TODO: replace the placeholder prim queries with the correct isaaclab asset accessors
for your robot and needle-tip prims.
"""

from __future__ import annotations

import math
import os
from typing import Tuple, List

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

# ---- Observations and helper functions ----------------------------------

def joint_positions(env, params=None):
    """Return joint positions for the robot as a tensor (shape [num_envs, n_joints]).

    NOTE: This is a template. Replace with actual access to the articulation's
    joint positions (e.g., env.scene['robot'].data.joint_pos).
    """
    asset: Articulation = env.scene["robot"]
    return asset.data.joint_pos  # assuming shape [num_envs, n_joints]


def joint_velocities(env, params=None):
    asset: Articulation = env.scene["robot"]
    return asset.data.joint_vel


# ---- Reset helpers ------------------------------------------------------

def reset_robot_state(env, env_ids=None):
    """Reset the robot root pose and joint states.

    This uses the articulation's cached default root state and zeros joints/vels.
    It is meant to be called from a reset EventTerm.
    """

    robot: Articulation = env.scene["robot"]
    device = robot.data.joint_pos.device
    num_envs = robot.data.joint_pos.shape[0]

    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device)
    else:
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)

    # Default root state (pose + velocity) if available, else identity + zeros
    if hasattr(robot.data, "default_root_state"):
        root_state = robot.data.default_root_state
        if root_state.shape[0] == 1 and num_envs > 1:
            root_state = root_state.expand(num_envs, *root_state.shape[1:]).clone()
            if hasattr(env, "scene") and hasattr(env.scene, "env_origins"):
                root_state[:, :3] += env.scene.env_origins.to(device)
        root_state = root_state.to(device)
    else:
        root_state = torch.zeros((num_envs, 13), device=device)
        root_state[:, 3] = 1.0

    robot.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)

    # Joint states
    if hasattr(robot.data, "default_joint_pos"):
        jpos = robot.data.default_joint_pos
        if jpos.shape[0] == 1 and num_envs > 1:
            jpos = jpos.expand(num_envs, *jpos.shape[1:])
        jpos = jpos.to(device)
    else:
        jpos = torch.zeros_like(robot.data.joint_pos)

    if hasattr(robot.data, "default_joint_vel"):
        jvel = robot.data.default_joint_vel
        if jvel.shape[0] == 1 and num_envs > 1:
            jvel = jvel.expand(num_envs, *jvel.shape[1:])
        jvel = jvel.to(device)
    else:
        jvel = torch.zeros_like(robot.data.joint_vel)

    robot.write_joint_state_to_sim(jpos[env_ids], jvel[env_ids], env_ids=env_ids)

    return True
# ---- Target sampling helpers ------------------------------------------

def reset_target_position(
    env,
    env_ids=None,
    x_range=None,
    y_range=None,
    z_value: float | None = None,
    **kwargs,
):
    """Sample a new target position uniformly inside the rectangle and place the visual sphere.

    Rectangle corners (world, z=1):
        (0.12492,  0.19746, 1)
        (0.12492, -0.08254, 1)
        (-0.07508, 0.19746, 1)
        (-0.07508,-0.08254, 1)

    This signature matches IsaacLab's event manager which passes `env_ids` and params as kwargs.
    """
    device = getattr(env.sim, "device", torch.device("cpu")) if hasattr(env, "sim") else torch.device("cpu")
    x_range = x_range if x_range is not None else (-0.07508, 0.12492)
    y_range = y_range if y_range is not None else (-0.08254, 0.19746)
    z_value = z_value if z_value is not None else 1.0

    # env_ids could be a list/array; normalize to tensor on device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs if hasattr(env, "num_envs") else env.scene.num_envs, device=device)
    else:
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)

    num = env_ids.numel()
    x = torch.rand((num,), device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y = torch.rand((num,), device=device) * (y_range[1] - y_range[0]) + y_range[0]
    z = torch.full((num,), z_value, device=device)
    sampled_local = torch.stack((x, y, z), dim=1)

    # env origins (world offsets for each sub-env)
    if hasattr(env, "scene") and hasattr(env.scene, "env_origins"):
        origins = env.scene.env_origins.to(device)
    else:
        origins = torch.zeros((env.num_envs if hasattr(env, "num_envs") else num, 3), device=device)

    sampled_world = sampled_local + origins[env_ids]

    # cache for later use (full tensor sized to num_envs)
    total_envs = env.num_envs if hasattr(env, "num_envs") else env.scene.num_envs
    if not hasattr(env, "_target_pos") or env._target_pos.shape[0] != total_envs:
        env._target_pos = torch.zeros((total_envs, 3), device=device)
    if env._target_pos.device != device:
        env._target_pos = env._target_pos.to(device)

    env._target_pos[env_ids] = sampled_world

    # create visualizer once and update marker poses for the env_ids subset
    if not hasattr(env, "_target_marker"):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/target_markers",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
                ),
            },
        )
        env._target_marker = VisualizationMarkers(marker_cfg)

    orientations = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device).repeat(num, 1)
    marker_indices = torch.zeros((num,), dtype=torch.long, device=device)
    env._target_marker.visualize(sampled_world, orientations, marker_indices=marker_indices)

    return env._target_pos


def get_target_position(env, device):
    """Return the current target position tensor on the given device."""
    total_envs = env.num_envs if hasattr(env, "num_envs") else env.scene.num_envs

    # If a target has already been sampled, return it (after moving to device).
    if hasattr(env, "_target_pos"):
        t = env._target_pos
        if t.device != device:
            t = t.to(device)
        return t

    # Fallback: initialize a default target above each env origin so observation
    # shape inference during manager setup does not crash before the first reset.
    if hasattr(env, "scene") and hasattr(env.scene, "env_origins"):
        origins = env.scene.env_origins.to(device)
    else:
        origins = torch.zeros((total_envs, 3), device=device)

    default_target = origins.clone()
    default_target[:, 2] = 1.0  # place at z=1.0 by default

    env._target_pos = default_target
    return env._target_pos

def needletip_error_vector(env, params=None):
    """target_pos - needletip_pos using articulation body pose for needletip.

    If no body name matches "needletip", we fall back to the TCP pose.
    """

    device = getattr(env, "device", torch.device("cpu"))
    if hasattr(env, "sim") and hasattr(env.sim, "device"):
        device = env.sim.device

    robot: Articulation = env.scene["robot"]

    # Cache the needletip body index; if missing, try TCP.
    if not hasattr(env, "_needle_tip_idx"):
        indices, _ = robot.find_bodies(".*needletip")
        if len(indices) == 0:
            raise RuntimeError("Needletip/TCP body not found; ensure panda_hand has needletip or TCP body")
        env._needle_tip_idx = int(indices[0])

    tip_pos_w = robot.data.body_pos_w[:, env._needle_tip_idx]

    if tip_pos_w.device != device:
        tip_pos_w = tip_pos_w.to(device)

    # target position from sampled target (or fallback)
    target_pos_w = get_target_position(env, device)

    return target_pos_w - tip_pos_w


# ---- Reward terms --------------------------------------------------------

def shaping_reward(env, k: float = 10.0):
    """Dense shaping: exp(-k * distance)."""
    # compute L2 distance
    err = needletip_error_vector(env)
    dist = torch.norm(err, dim=1)
    return torch.exp(-k * dist)


def success_reward(env, bonus: float = 100.0, threshold: float = 0.01):
    err = needletip_error_vector(env)
    dist = torch.norm(err, dim=1)
    return (dist < threshold).to(dtype=dist.dtype) * bonus


def action_l2_penalty(env, params=None):
    """Penalty proportional to L2 norm of actions.

    Note: depending on how actions are provided to the reward function, you may
    need to retrieve the last applied action from the env or manager. This
    placeholder assumes `env.last_action` exists with shape [num_envs, action_dim].
    """
    device = getattr(env.sim, "device", torch.device("cpu")) if hasattr(env, "sim") else torch.device("cpu")
    actions = getattr(env, "last_action", None)
    if actions is None:
        # fallback: zero penalty
        num_envs = env.num_envs if hasattr(env, "num_envs") else env.scene.num_envs
        return torch.zeros((num_envs,), dtype=torch.float32, device=device)
    l2 = torch.sum(torch.square(actions), dim=1)
    return l2


def collision_penalty(env, params=None):
    """Penalty based on contact forces from contact sensor (uses GPU tensors)."""
    robot: Articulation = env.scene["robot"]
    device = getattr(env.sim, "device", torch.device("cpu")) if hasattr(env, "sim") else torch.device("cpu")
    if not hasattr(robot.data, "net_forces_w"):
        num_envs = robot.data.body_pos_w.shape[0]
        return torch.zeros((num_envs,), device=device)

    net_forces = robot.data.net_forces_w  # [num_envs, num_bodies, 3]
    force_mag = torch.norm(net_forces, dim=(1, 2))
    is_collided = force_mag > 1.0
    return is_collided.float() * -10.0

def arm_contact_termination(env, params=None, threshold: float = 50.0):
    """Terminate if any protected links/hand/needle register contact (hard stop)."""

    device = getattr(env.sim, "device", torch.device("cpu")) if hasattr(env, "sim") else torch.device("cpu")
    num_envs = env.num_envs if hasattr(env, "num_envs") else env.scene.num_envs

    def _over_threshold(sensor_name: str) -> torch.Tensor:
        try:
            sensor = env.scene[sensor_name]
        except Exception:
            return torch.zeros((num_envs,), dtype=torch.bool, device=device)

        if not hasattr(sensor, "data") or not hasattr(sensor.data, "net_forces_w"):
            return torch.zeros((num_envs,), dtype=torch.bool, device=device)
        force_mag = torch.norm(sensor.data.net_forces_w, dim=(1, 2))
        return force_mag > threshold

    # Any of the protective sensors triggers termination
    arm_phantom = _over_threshold("contact_arm_phantom5") | _over_threshold("contact_arm_phantom6") | _over_threshold("contact_arm_phantom7")
    arm_table = _over_threshold("contact_arm_table5") | _over_threshold("contact_arm_table6") | _over_threshold("contact_arm_table7")
    hand_phantom = _over_threshold("contact_hand_phantom")
    hand_table = _over_threshold("contact_hand_table")
    # needle_hit = _over_threshold("contact_needle_protect")

    return arm_phantom | arm_table | hand_phantom | hand_table # | needle_hit

def needle_impact_penalty(env, params=None):
    """Soft penalty based on needle tip contact forces (no termination)."""
    device = getattr(env.sim, "device", torch.device("cpu")) if hasattr(env, "sim") else torch.device("cpu")
    try:
        sensor = env.scene["contact_needle_tip"]
    except Exception:
        num_envs = env.num_envs if hasattr(env, "num_envs") else env.scene.num_envs
        return torch.zeros((num_envs,), device=device)

    if not hasattr(sensor, "data") or not hasattr(sensor.data, "net_forces_w"):
        num_envs = env.num_envs if hasattr(env, "num_envs") else env.scene.num_envs
        return torch.zeros((num_envs,), device=device)

    net_force = torch.norm(sensor.data.net_forces_w, dim=(1, 2))
    penalty = (net_force > 0.05).float() * -5.0 * net_force
    return penalty


# ---- Termination helpers -------------------------------------------------

def is_success(env, threshold: float = 0.002):
    err = needletip_error_vector(env)
    dist = torch.norm(err, dim=1)
    return dist < threshold


