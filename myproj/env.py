
# Generic Imports
import argparse
from isaaclab.app import AppLauncher
import random
import numpy as np
import sys
import os, glob

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
parser.add_argument("--num_envs", type=int, default=2, help="number envs to spawn")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
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

# This defines one interactive scene, imported USD and physics from Isaac Lab

def create_hulls(cfg: DictConfig):
    #set random for numpy - do for torch too!
    np.random.seed(cfg.random.seed)

    # load the parameters csv
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(SCRIPT_DIR, cfg.ships.filepath)

    print(csv_path)

    vectors = []

    if os.path.isfile(csv_path):
        try:
            vectors = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
        except Exception as e:
            print(f"Failed to load vectors: {e}")
            vectors = np.empty((0,))
    else:
        print('not valid csv file')


    # randomly select N
    valid_indices = np.array([0,1,2,3])
    chosen_indices = np.random.choice(np.arange(valid_indices.size), 
                                      size=4, replace=False)
    boat_hulls = []
    vecs = list(vectors[chosen_indices])
    for i, vec in enumerate(vecs):
        hull = HP(vec)
        base = os.path.join("~/temp", f'hull_{i}')
        hull.gen_USD(
            NUM_WL=50,
            PointsPerWL=200,
            bit_AddTransom=1,
            bit_AddDeckLid=1,
            bit_RefineBowAndStern=0,
            namepath=base
        )
        usd_file = base + '.usd'
        if not os.path.isfile(usd_file):
            print(f"USD generation failed: {usd_file} not found")
            return



def inject_USDs(cfg: DictConfig, prim_path_expr: str):

    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
    num_envs = len(prim_paths)

    mtl_path = Sdf.Path("/World/Looks/CheckerProc")
    mtl      = UsdShade.Material.Define(stage, mtl_path)

    # ➊ MDL “3-D checker” shader
    checker  = UsdShade.Shader.Define(stage, mtl_path.AppendChild("Checker3D"))
    checker.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    checker.SetSourceAsset("core_definitions.mdl", "mdl")
    checker.SetSourceAssetSubIdentifier("3d_checker_texture", "mdl")  # procedural pattern

    # ➋ Plain UsdPreviewSurface that will read Color from the checker node
    pbs      = UsdShade.Shader.Define(stage, mtl_path.AppendChild("PreviewSurface"))
    pbs.CreateIdAttr("UsdPreviewSurface")
    pbs.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)\
       .ConnectToSource(checker.ConnectableAPI(), "color")

    # ➌ Wire the preview shader into the material
    mtl.CreateSurfaceOutput().ConnectToSource(pbs.ConnectableAPI(), "surface")
        
    # add them appropriately
    with Sdf.ChangeBlock():
        for prim_path in prim_paths:

            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            # Note: Just need to acquire the right attribute about the property you want to set
            # Here is an example on setting color randomly
            color_spec = prim_spec.GetAttributeAtPath(prim_path + "/geometry/material/Shader.inputs:diffuseColor")
            color_spec.default = Gf.Vec3f(random.random(), random.random(), random.random())
    
    # apply some augmentations


    # bind the textures + augment them
    # create_texture(prim_path_expr)


# def make_procedural_checker(mtl_path:str):
#     mtl_path = Sdf.Path(mtl_path)
#     mtl      = UsdShade.Material.Define(stage, mtl_path)

#     # ➊ MDL “3-D checker” shader
#     checker  = UsdShade.Shader.Define(stage, mtl_path.AppendChild("Checker3D"))
#     checker.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
#     checker.SetSourceAsset("core_definitions.mdl", "mdl")
#     checker.SetSourceAssetSubIdentifier("3d_checker_texture", "mdl")  # procedural pattern

#     # ➋ Plain UsdPreviewSurface that will read Color from the checker node
#     pbs      = UsdShade.Shader.Define(stage, mtl_path.AppendChild("PreviewSurface"))
#     pbs.CreateIdAttr("UsdPreviewSurface")
#     pbs.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)\
#        .ConnectToSource(checker.ConnectableAPI(), "color")

#     # ➌ Wire the preview shader into the material
#     mtl.CreateSurfaceOutput().ConnectToSource(pbs.ConnectableAPI(), "surface")
#     return mtl_path

# make_procedural_checker("/World/Looks/CheckerProc")


    

def create_texture(prim_path_expr: str):

    # 1. Create a dynamic texture provider with a unique name
    dyn_tex_name = "paintTex"
    dyn_tex = omni.ui.DynamicTextureProvider(dyn_tex_name)

    # 2. Initialize the texture data (RGBA image) – e.g. white background 1024x1024
    tex_width, tex_height = 1024, 1024
    hull_color = (255, 255, 255, 255)       # RGBA for paint (red in this example)
    algae_color    = (0, 255, 0, 255)   # RGBA background (white)
    texture_data = np.full((tex_height, tex_width, 4), algae_color, dtype=np.uint8)

    # Send the initial texture to GPU
    dyn_tex.set_data_array(texture_data, [tex_width, tex_height])

    # 3. Create an MDL material (OmniPBR) and assign the dynamic texture to it
    stage = omni.usd.get_context().get_stage()
    material_path = "/World/AlgaeMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")

    # Configure the shader to use OmniPBR (which has a diffuse texture slot)
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateIdAttr("OmniPBR")

    # Set the diffuse texture input to the dynamic texture (using dynamic:// scheme)
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)\
        .Set(f"dynamic://{dyn_tex_name}")  # Link to our DynamicTextureProvider:contentReference[oaicite:2]{index=2}

    prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
    
    # def get_file_from_index():
    #     try:
    #         self.params[idx] = float(widget.text())
    #     except ValueError:
    #     return callback

    # def get_hull_USD():
    #     hull = HP(self.params)
    #     base = os.path.join(tempfile.gettempdir(), 'hull_view')
    #     hull.gen_USD(
    #         NUM_WL=50,
    #         PointsPerWL=200,
    #         bit_AddTransom=1,
    #         bit_AddDeckLid=1,
    #         bit_RefineBowAndStern=1,
    #         namepath=base
    #     )
    #     stl_file = base + '.stl'
    #     if not os.path.isfile(stl_file):
    #         print(f"STL generation failed: {stl_file} not found")
    #         return
    #     usd_file = base + '.usd'
    #     if not os.path.isfile(usd_file):
    #         print(f"USD generation failed: {usd_file} not found")
    #         return


    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        get_hull_USD()
        for prim_path in prim_paths:
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            # Connect shader to material and bind to mesh
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            ground_prim = prim_spec  # path to the curved surface prim
            ground_prim.ApplyAPI(UsdShade.MaterialBindingAPI)           # ensure binding API is present
            UsdShade.MaterialBindingAPI(ground_prim).Bind(material)



@configclass
class SceneCfgOverride(InteractiveSceneCfg):

    # 1) collect every USD in your folder (runs once, at import time)
    # load the parameters csv
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    usd_path = os.path.join(SCRIPT_DIR, "temp/*.usd")
    HULL_BANK = glob.glob(usd_path)
    assert HULL_BANK, "No hull USDs found – check the folder path"

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    #lights
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))

    # articulation??

    hull: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/hull",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(usd_path="/home/ubuntu/Desktop/oc-2.usd"),
    )

    # ----------------------------------------
    # hulls – one per parallel environment
    # ----------------------------------------
    hull = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/hull",
        spawn=sim_spawners.MultiUsdFileCfg(
            usd_path=HULL_BANK,          # <- *list* or wildcard
            random_choice=True,          # <- pick a new one each reset
            rigid_props=sim_schemas.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # <- immovable but still a RigidBody
            ),
            collision_props=sim_schemas.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )


    #     # Set Cube as object
    # self.scene.object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         scale=(0.8, 0.8, 0.8),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #     ),
    # )

    
def make_mesh():

    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(hit_prim_path))
    # Get face vertex indices and UV coordinates
    fv_indices = mesh.GetFaceVertexIndicesAttr().Get()       # flattened list of vertex indices
    fv_counts  = mesh.GetFaceVertexCountsAttr().Get()        # should be [3,3,3,...] for triangles
    uvs_primvar = mesh.GetPrimvar("st") or mesh.GetPrimvar("uv")
    uvs = uvs_primvar.Get()                                  # UV coordinate array

    # Extract the 3 indices for the hit triangle:
    tri_idx = hit_face_index
    i0 = fv_indices[tri_idx*3 + 0]
    i1 = fv_indices[tri_idx*3 + 1]
    i2 = fv_indices[tri_idx*3 + 2]

    # Get the corresponding UV coordinates for each vertex of the triangle:
    if uvs_primvar.GetInterpolation() == UsdGeom.Tokens.faceVarying:
        # face-varying: there's a unique UV per face-vertex
        uv0 = uvs[tri_idx*3 + 0]
        uv1 = uvs[tri_idx*3 + 1]
        uv2 = uvs[tri_idx*3 + 2]
    else:
        # vertex interpolation: one UV per vertex index
        uv0 = uvs[i0];  uv1 = uvs[i1];  uv2 = uvs[i2]



def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.

    my_scene = scene['my_scene']

    rigid = my_scene.rigid_objects['Object']
    
    sim_dt = sim.get_physics_dt()

    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 1000 == 0:
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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main function."""

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

    scene_cfg = SceneCfgOverride(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)

    with Timer("[INFO] Time to randomize scene: "):
        # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
        # Note: Just need to acquire the right attribute about the property you want to set
        # Here is an example on setting color randomly
        inject_USDs(cfg, scene_cfg.my_scene.prim_path)

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
