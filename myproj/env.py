
# Generic Imports
import argparse
from isaaclab.app import AppLauncher
import random
import numpy as np
import sys
import os, glob

import warnings
warnings.filterwarnings("ignore")


# for ship
import tempfile
from ShipD.HullParameterization import Hull_Parameterization as HP

# for DL
import torch

# config
import hydra
from omegaconf import DictConfig, OmegaConf


# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument("--num_envs", type=int, default=100, help="number envs to spawn")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# my imports
from myproj.shader import ShaderManager
from robot import MeshManager, HullCrawler

import carb.settings
settings = carb.settings.get_settings()

# Hide dev pop-ups & deprecation nags
settings.set("/app/enableDeveloperWarnings", False)
settings.set("/app/scripting/ignoreWarningDialog", True)

# Silence the UI console, too
for key in ("verbose", "info", "warning", "error", "fatal"):
    settings.set(f"/exts/omni.kit.window.console/logFilter/{key}", False)

# Global log thresholds
settings.set("/log/outputStreamLevel", "Error")
settings.set("/log/fileLogLevel",   "Error")
settings.set("/log/debugConsoleLevel", "Fatal")

# for Isaac Sim
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

# for RL
from isaaclab.envs import ManagerBasedRLEnv

# for Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)

# sensors
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns, RayCaster

# actuators and articulation
from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab.assets import AssetBaseCfg
# from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
JACKAL_PATH = "/home/ubuntu/Desktop/jackal_basic.usd"

# dynamic texture constants
TEXTURE_SIZE = (1024, 1024)  # (height, width) used by ShaderManager
GREEN_RGBA = np.array([0, 255, 0, 255], dtype=np.uint8)
WHITE_RGBA = np.array([255, 255, 255, 255], dtype=np.uint8)


import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdShade

from isaaclab.sim import schemas as sim_schemas, spawners as sim_spawners


# from isaaclab.sim import SimulationCfg, SimulationContext

# from isaaclab.envs import ManagerBasedRLEnv
# from isaaclab.terrains import TerrainImporterCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# from isaaclab.utils import configclass
# from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
# from omni.isaac.lab.sim import SimulationContext
# from omni.isaac.lab.assets import AssetBaseCfg
            
            
# 1) collect every USD in folder (runs at import time)
# load the parameters csv
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
usd_path = os.path.join(SCRIPT_DIR, "temp/*.usd")
HULL_BANK = glob.glob(usd_path)
assert HULL_BANK, "No hull USDs found - check the folder path"


robot_config = ArticulationCfg(
    # spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Clearpath/Jackal/jackal.usd"),
    spawn=sim_utils.UsdFileCfg(usd_path=JACKAL_PATH),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
) 


@configclass
class SceneCfgOverride(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())

    #lights
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))

    # articulation??
    # articulation - robot 1
    # robot_1 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
    # # articulation - robot 2
    # robot_2 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
    # robot_2.init_state.pos = (0.0, 1.0, 0.6)

    # sensor - ray caster attached to the base of robot 1 that scans the ground
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot_1/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )

    # robot
    # robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # robot = robot_config.replace(prim_path="/World/envs/env_.*/Robot")
    robot = robot_config.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.0, -1.0, 15.0),      # 10 m up, matching your hull's spawn height
        rot=(1.0, 0.0, 0.0, 0.0),  # identity quaternion
        )
    )


    # # ray‐caster (LIDAR) under the robot’s base, scanning straight down
    ray_caster_cfg = RayCasterCfg(
        # match each env’s Robot prim
        prim_path="{ENV_REGEX_NS}/Robot/base/lidar_cage",
        # 60 Hz update
        update_period=1/60,
        # place it 0.3 m above the base link so rays go down to the ground
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
        # only cast onto the ground plane
        mesh_prim_paths=["/World/Ground"],
        # lock pitch/roll so it always faces straight down, but follows yaw
        attach_yaw_only=True,
        # grid‐pattern: 0.05 m spacing, cover 0.8 m × 1.0 m area under the robot
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.05,
            size=(0.8, 1.0),
        ),
        # enable the debug viz in non‐headless mode
        debug_vis=not args_cli.headless,
    )

    # ----------------------------------------
    # hulls – one per parallel environment
    # ----------------------------------------
    hull = AssetBaseCfg(
        prim_path="/World/envs/env_.*/hull",
        spawn=sim_spawners.MultiUsdFileCfg(
            usd_path=HULL_BANK,          # <- *list* or wildcard
            random_choice=True,          # <- pick a new one each reset
            rigid_props=sim_schemas.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # <- immovable but still a RigidBody
            ),
            collision_props=sim_schemas.CollisionPropertiesCfg()
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 4.0],
            rot=[1.0, 0.0, 0.0, 0.0]  #[0.7071, 0.0, 0.0, 0.7071],
        ),
    )


def register_sensors(scene, sim):
    # register one downward RayCaster per environment
    sensors = {}
    for env_id, env_path in enumerate(scene.env_prim_paths):
        base_path = f"{env_path}/Robot/base/lidar_cage"
        sensor = RayCaster(
            RayCasterCfg(
                prim_path       = base_path,
                update_period   = 1.0 / 60,
                offset          = RayCasterCfg.OffsetCfg(pos=(0, 0, 0.30)),
                mesh_prim_paths = [f"{env_path}/hull"],
                attach_yaw_only = True,
                pattern_cfg     = patterns.GridPatternCfg(resolution=0.05,
                                                        size=(0.8, 1.0)),
            )
        )
        sim.add_sensor(sensor)
        sensor.initialize()
        sensors[env_id] = sensor



def run_simulator(sim: SimulationContext, scene: InteractiveScene, robots, managers):
    """Runs the simulation loop and updates dynamic textures."""
    # local texture buffers mirroring the provider contents
    # textures = [np.full((*TEXTURE_SIZE, 4), GREEN_RGBA, dtype=np.uint8)
    #             for _ in providers]
    
    
    
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.

    hull = scene['Hull']
    


    #rigid = hull.rigid_objects['hull']
    
    sim_dt = sim.get_physics_dt()
    
    sensors = register_sensors(scene, sim)
    
    half_mask = np.zeros((TEXTURE_SIZE[0], TEXTURE_SIZE[1]), dtype=bool)
    half_mask2 = np.zeros((TEXTURE_SIZE[0], TEXTURE_SIZE[1]), dtype=bool)
    half_mask[:, :TEXTURE_SIZE[1] // 2 ] = True
    half_mask2[:TEXTURE_SIZE[0] // 2 , :] = True
    
    # stripes = generate_stripe_texture(
    #     TEXTURE_SIZE[1], TEXTURE_SIZE[0], num_stripes=10, pattern_idx=0
    # )
    
    count = 0
    # Simulation loop
    while simulation_app.is_running():

        # # print information from the sensors
        # print("-------------------------------")
        # print(scene["ray_caster"])
        # print("Ray cast hit results: ", scene["ray_caster"].data) #.ray_hits_w)

        # Reset
        if count % 20000 == 0:
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
            sim.reset()
            print("[INFO]: Resetting scene state...")

        # Apply action to robot
        # robot.set_joint_position_target(robot.data.default_joint_pos)

        # if count % 20 == 0:
        #     # Update dynamic textures
        #     for tex, provider in zip(textures, providers):
        #         mask = np.random.rand(TEXTURE_SIZE[0], TEXTURE_SIZE[1]) < 0.5
        #         green_mask = np.all(tex == GREEN_RGBA, axis=2)
        #         white_mask = np.all(tex == WHITE_RGBA, axis=2)
        #         tex[np.logical_and(mask, green_mask)] = WHITE_RGBA
        #         tex[np.logical_and(mask, white_mask)] = GREEN_RGBA
        #         provider.set_data_array(tex, list(tex.shape))
                
        if count % 60 == 0:
            for tex, provider in zip(textures, providers):
                tex = generate_stripe_texture(
                    TEXTURE_SIZE[1], TEXTURE_SIZE[0], num_stripes=10, pattern_idx=(count // 60) % 4
                )

                # green_mask = np.all(tex == GREEN_RGBA, axis=2)
                # white_mask = np.all(tex == WHITE_RGBA, axis=2)
                # tex[np.logical_and(half_mask, green_mask)] = WHITE_RGBA
                # tex[np.logical_and(half_mask, white_mask)] = GREEN_RGBA
                # tex[np.logical_and(half_mask2, green_mask)] = WHITE_RGBA
                # tex[np.logical_and(half_mask2, white_mask)] = GREEN_RGBA
                provider.set_data_array(tex, list(tex.shape))


        sensor   = sensors[env.name]
        hits_w   = sensor.data["ray_hits_w"][0]         # (N,3)
        normals_w= sensor.data["ray_normals_w"][0]      # (N,3)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

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


def _setup_scene():
    pass
    

def randomize_shape_color(prim_path_expr: str):
    """Randomize the color of the geometry."""
    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            # Note: Just need to acquire the right attribute about the property you want to set
            # Here is an example on setting color randomly
            color_spec = prim_spec.GetAttributeAtPath(prim_path + "/geometry/material/Shader.inputs:diffuseColor")
            color_spec.default = Gf.Vec3f(random.random(), random.random(), random.random())

    

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main function."""

    global HULL_BANK

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    with Timer("[INFO] Time to create ships: "):
        # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
        # Note: Just need to acquire the right attribute about the property you want to set
        # Here is an example on setting color randomly
        # create_hulls(cfg)
        pass

        # from pxr import Usd
        # s = Usd.Stage.Open("/home/ubuntu/IsaacLab/Keelcrab/myproj/temp/hull_2.usd")
        # print(s.GetDefaultPrim())        # should print </Hull>
        # root_layer = s.GetRootLayer()
        # print("layer's defaultPrim token →", root_layer.defaultPrim)  # "Hull"

    with Timer("[INFO] Time to setup scene: "):
        _setup_scene()

        #args_cli.num_envs
    # 1. Build the scene (this builds & lays out the N cloned hulls)
    scene_cfg = SceneCfgOverride(num_envs=2,
                                env_spacing=4.0,
                                replicate_physics=False)

    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)      # stage now exists


    # get the components
    hull_objs = scene["hull", ...]    # returns a list of Hull components across all envs
    robot_objs = scene["robot", ...]
    
    # 2. Gather the prim paths you want to paint.
    #    If your hull asset’s root prim is called "hull", every clone
    #    will live under /World/envs/env_0/hull, /World/envs/env_1/hull, …
    hull_paths = [f"/World/envs/env_{i}/hull" for i in range(scene_cfg.num_envs)]
    
    # 3. Create the provider + material and bind it to those prims
    managers = []
    for idx in range(scene_cfg.num_envs):
        provider = ShaderManager.make_dynamic_hull(stage=scene.stage, prim_path=hull_paths[idx], idx=idx)
        
        managers.append(MeshManager(provider, scene, idx, TEXTURE_SIZE))
        # providers.append(provider)
        
        
    robots = []
    for idx, robot in enumerate(robot_objs):
        # create the robot
        robot = HullCrawler(scene, idx, ray_sensor_key="ray_caster_cfg", robot_key="Robot")
        robots.append(robot)


    with Timer("[INFO] Time to randomize scene: "):
        # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
        # Note: Just need to acquire the right attribute about the property you want to set
        # Here is an example on setting color randomly
        pass

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    run_simulator(sim, scene, robots, managers)





if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
