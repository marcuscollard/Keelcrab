    
    
    
    
    
    
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