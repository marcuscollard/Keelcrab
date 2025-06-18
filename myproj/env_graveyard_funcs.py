    
    
    
    
    
    
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