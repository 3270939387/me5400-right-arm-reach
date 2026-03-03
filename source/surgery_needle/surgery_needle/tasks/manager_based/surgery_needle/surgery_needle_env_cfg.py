"""Manager-based environment configuration for SurgeryNeedleReach task.

This file defines:
 - SurgeryNeedleSceneCfg (scene placement of assets)
 - ActionsCfg (7D joint velocity)
 - ObservationsCfg (joint pos, joint vel, needletip error vector)
 - RewardsCfg (shaping, success, action penalty)
 - TerminationsCfg (success, timeout)

It also exposes a create_env factory `create_env(cfg=None, render_mode=None)`
that returns a `ManagerBasedRLEnv` instance for use with `gym.make(..., cfg=...)`.

Note: some helper functions (asset/prim queries, transforms) are implemented
in the sibling `mdp.py` module. You should adapt the prim names and transforms
to match your project's assets (prim paths are noted in the user request).
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import time_out
# Extension root (used to build asset paths)
EXTENSION_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from . import mdp


@configclass
class SurgeryNeedleSceneCfg(InteractiveSceneCfg):
    """Scene configuration placing assets at fixed world poses."""

    # Assets are placed using fixed prim spawn configurations. Set spawn to
    # UsdFileCfg pointing to the extension's assets folder so Isaac Lab can load them.
    ground_plane = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Robot (Franka Panda) - absolute world pose
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/panda",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{EXTENSION_PATH}/assets/Robots/Franka/Collected_panda_assembly/panda_assembly.usda",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            # 提高此值（或保持默认）以确保穿模时有足够的排斥力
                max_depenetration_velocity=20.0, 
            # 增加阻尼可以减少机械臂在碰撞或重置时的剧烈抖动
                linear_damping=0.01,
                angular_damping=0.01,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # 开启机器人各 Link 之间的自碰撞，防止手穿过底座
                enabled_self_collisions=True, 
                # 增加解算器迭代次数，提高物理模拟的精度和硬度
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0213, -0.42073, 0.828),
            rot=(0.70710678, 0.0, 0.0, 0.70710678),  # 90 deg yaw about z
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -0.07,  # within limit [-3.072, -0.070]
                "panda_joint5": 0.0,
                "panda_joint6": 0.0,
                "panda_joint7": 0.0,
            },
        ),
        actuators={
            "panda_arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                stiffness=0.0,
                damping=40.0,
            ),
        },
    )

    # Table (absolute world pose)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{EXTENSION_PATH}/assets/Props/VentionTableWithBlackCover/table_with_cover.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.00505, 0.00954, 0.00034),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Phantom (absolute world pose, placed on table) registered as rigid object for physics + data tensors
    phantom: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/phantom",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{EXTENSION_PATH}/assets/Props/ABDPhantom/phantom.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.025, 0.057, 0.911),
            rot=(0.0, 1.0, 0.0, 0.0),  # 180 deg roll about x
        ),
    )

    # Contact sensors
    contact_arm_phantom5 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/panda/panda_link5",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/phantom"]
    )

    contact_arm_phantom6 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/panda/panda_link6",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/phantom"]
    )

    contact_arm_phantom7 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/panda/panda_link7",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/phantom"]
    )

    contact_arm_table5 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/panda/panda_link5",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/table"]
    )

    contact_arm_table6 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/panda/panda_link6",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/table"]
    )

    contact_arm_table7 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/panda/panda_link7",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/table"]
    )

    contact_hand_phantom = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/panda/panda_hand",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/phantom"]
    )

    contact_hand_table = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/panda/panda_hand",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/table"]
    )

    # contact_needle_protect = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/panda/needle",
    #     update_period=0.0,
    # )

    # contact_needle_tip = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/panda/needletip",
    #     update_period=0.0,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/phantom"],
    # )


@configclass
class ActionsCfg:
    """Action spec: 7D joint velocity applied to the robot articulation.

    The MDP action implementation expected by the manager will need to convert
    this into joint velocity commands for the robot articulation.
    """

    joint_velocity = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
        scale=1.0,
    )


@configclass
class ObservationsCfg:
    """Observation specs: joint positions, joint velocities, and needle-tip error.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_positions)
        joint_vel = ObsTerm(func=mdp.joint_velocities)
        needletip_error = ObsTerm(func=mdp.needletip_error_vector)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms: dense shaping, success bonus, action penalty, optional collision penalty."""

    shaping = RewTerm(func=mdp.shaping_reward, weight=2.0, params={"k": 15.0})
    success = RewTerm(func=mdp.success_reward, weight=1.0, params={"bonus": 100.0, "threshold": 0.01})
    action_penalty = RewTerm(func=mdp.action_l2_penalty, weight=-0.01)
    needle_contact_penalty = RewTerm(func=mdp.needle_impact_penalty, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms: success and timeout."""

    success = DoneTerm(func=mdp.is_success, params={"threshold": 0.001})
    arm_contact = DoneTerm(
        func=mdp.arm_contact_termination,
        params={"threshold": 2.0},
    )
    time_out = DoneTerm(func=time_out, time_out=True)


@configclass
class EventCfg:
    """Reset-time randomization events."""

    # Sample a new target position in the given rectangle each reset
    sample_target = EventTerm(
        func=mdp.reset_target_position,
        mode="reset",
        params={
            "x_range": (-0.07508, 0.12492),
            "y_range": (-0.08254, 0.19746),
            "z_value": 1.0,
        },
    )

    # Reset robot state (root pose + joints) each reset
    # reset_robot = EventTerm(
    #     func=mdp.reset_robot_state,
    #     mode="reset",
    # )


@configclass
class SurgeryNeedleEnvCfg(ManagerBasedRLEnvCfg):
    scene: SurgeryNeedleSceneCfg = SurgeryNeedleSceneCfg(num_envs=64, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # Basic sim settings
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (1.0, -1.5, 1.2)
        self.sim.dt = 1.0 / 120.0
        # self.sim.render_interval = self.decimation


def create_env(cfg: SurgeryNeedleEnvCfg | None = None, render_mode: str | None = None):
    """Factory to create the ManagerBasedRLEnv for the task.

    This function is referenced by the gym registration entry_point.
    If `cfg` is None, a default `SurgeryNeedleEnvCfg()` will be used.
    """

    if cfg is None:
        cfg = SurgeryNeedleEnvCfg()

    env = ManagerBasedRLEnv(cfg, render_mode=render_mode)
    return env


# alias expected by needle_reach-style registration
def create(cfg: SurgeryNeedleEnvCfg | None = None, render_mode: str | None = None):
    return create_env(cfg, render_mode)
