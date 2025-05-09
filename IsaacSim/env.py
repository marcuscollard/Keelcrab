

# launch the scene


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument("--num_envs", type=int, default=2, help="number envs to spawn")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Generic Imports
import random
import omni.usd
from pxr import Gf, Sdf

# for Isaac Sim
from source.isaaclab.isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

# for DL
import torch

# for RL
from isaaclab.envs import ManagerBasedRLEnv

# for Isaac Lab
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


# from isaaclab.sim import SimulationCfg, SimulationContext

# from isaaclab.envs import ManagerBasedRLEnv
# from isaaclab.terrains import TerrainImporterCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# from isaaclab.utils import configclass
# from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
# from omni.isaac.lab.sim import SimulationContext
# from omni.isaac.lab.assets import AssetBaseCfg



# This defines one interactive scene, imported USD and physics from Isaac Lab

@configclass
class SceneCfgOverride(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    #lights
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))

    # articulation??

    my_asset: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(usd_path="/home/ubuntu/Desktop/oc-2.usd"),
    )



def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.

    
    sim_dt = sim.get_physics_dt()

    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 250 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # object
            # root_state = rigid_object.data.default_root_state.clone()
            # root_state[:, :3] += scene.env_origins
            # rigid_object.write_root_pose_to_sim(root_state[:, :7])
            # rigid_object.write_root_velocity_to_sim(root_state[:, 7:])
            # # object collection
            # object_state = rigid_object_collection.data.default_object_state.clone()
            # object_state[..., :3] += scene.env_origins.unsqueeze(1)
            # rigid_object_collection.write_object_link_pose_to_sim(object_state[..., :7])
            # rigid_object_collection.write_object_com_velocity_to_sim(object_state[..., 7:])
            # # robot
            # # -- root state
            # root_state = robot.data.default_root_state.clone()
            # root_state[:, :3] += scene.env_origins
            # robot.write_root_pose_to_sim(root_state[:, :7])
            # robot.write_root_velocity_to_sim(root_state[:, 7:])
            # # -- joint state
            # joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            # robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting scene state...")

        # Apply action to robot
        # robot.set_joint_position_target(robot.data.default_joint_pos)
        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)







#         my_asset: AssetBaseCfg = AssetBaseCfg(
#         prim_path="/World/scene/my_asset",
#         init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
#         spawn=UsdFileCfg(usd_path="PATH_TO_MY_USD.usd"),
#     )
        
#         assert_cfg = self.cfg.my_asset
# assert_cfg.spawn.func(
#     assert_cfg.prim_path, assert_cfg.spawn, translation=assert_cfg.init_state.pos, orientation=assert_cfg.init_state.rot,
# )


def _setup_scene(self):
    pass


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    scene_cfg = SceneCfgOverride(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
