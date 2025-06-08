import omni.ui as ui
import numpy as np

from pxr import UsdShade, Sdf, Gf, Usd
import omni.ui as ui, numpy as np, omni.usd


class ShaderManager:
    """
    This example shows how to create a dynamic texture and bind it to a
    material, which is then bound to a prim in the stage.
    The dynamic texture is a simple gradient from green to white.
    """
    def __init__(self, stage):
        self.stage = stage
    
    # constants
    # ----------------------------------------------------------------------
    # CONFIG — change to your mesh or subset
    PRIM_PATH = "/Hull"                       # where to bind the material
    MDL_PATH  = "file:///opt/IsaacSim/kit/mdl/core/Base/OmniPBR.mdl"
    TEX_NAME  = "live_hull"               # dynamic://live_hull
    MAT_PATH  = "/hooks/hull_dyna"
    # ----------------------------------------------------------------------
    

    @staticmethod
    def create_dynamic_hull(
        stage: Usd.Stage,
        prim_path,
        idx,
        *,
        tex_name=TEX_NAME,
        tex_rgba=None,
        size=(50, 300),                # (H, W)
        mdl_path=MDL_PATH,
        uv_set_index=0,                # 0 = primvar:st, 1 = st1, …
    ):
        """
        Parameters
        ----------
        stage : Usd.Stage              the live stage (already loaded)
        prim_paths : iterable[str]     mesh prim paths that need the material
        tex_name : str                 name of the DynamicTextureProvider
        tex_rgba : ndarray | None      RGBA image; if None a solid green is used
        size : (H, W)                  only used when tex_rgba is None
        mdl_path : str                 resolved MDL asset containing OmniPBR
        uv_set_index : int             0 = st, 1 = st1, etc.
        """
        # ------------------------------------------------------------------ #
        # 1.  Dynamic texture provider
        # ------------------------------------------------------------------ #
        provider = ui.DynamicTextureProvider(tex_name)
        if tex_rgba is None:
            H, W = size
            tex_rgba = np.full((H, W, 4), [0, 255, 0, 255], np.uint8)
        provider.set_data_array(tex_rgba, list(tex_rgba.shape))

        # ------------------------------------------------------------------ #
        # 2.  Material + Shader (re-use if they already exist)
        # ------------------------------------------------------------------ #
        mat_path = f"/World/Materials/{tex_name}"
        shd_path = f"{mat_path}/OmniPBR"

        mat = UsdShade.Material.Get(stage, mat_path)
        if True:
            # -------- (a) create fresh material & shader -------------------
            mat = UsdShade.Material.Define(stage, mat_path)
            shd = UsdShade.Shader.Define(stage, shd_path)
            shd.CreateIdAttr("OmniPBR")

            # MDL linkage (ImplementationSource must be set first on new USD)
            shd.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            shd.SetSourceAsset(Sdf.AssetPath(mdl_path), "mdl")
            shd.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

            # ---------- inputs --------------------------------------------
            # which UV set the texture will use
            shd.CreateInput("diffuse_texture_st_index",
                            Sdf.ValueTypeNames.Int).Set(uv_set_index)

            # actual texture
            shd.CreateInput("diffuse_texture",
                            Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(f"dynamic://{tex_name}")
            )

            # if you project UVW instead, uncomment:
            # shd.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)

            # ---------- required OmniPBR outputs ---------------------------
            out_surf = shd.CreateOutput("surface",      Sdf.ValueTypeNames.Token)
            out_vol  = shd.CreateOutput("volume",       Sdf.ValueTypeNames.Token)
            out_disp = shd.CreateOutput("displacement", Sdf.ValueTypeNames.Token)

            mat.CreateSurfaceOutput().ConnectToSource(out_surf)
            mat.CreateVolumeOutput().ConnectToSource(out_vol)
            mat.CreateDisplacementOutput().ConnectToSource(out_disp)

        else:
            # shader already exists → just grab it
            shd = UsdShade.Shader.Get(stage, shd_path)

        # ------------------------------------------------------------------ #
        # 3.  Bind material to every prim listed
        # ------------------------------------------------------------------ #
        for p in prim_paths:
            prim = stage.GetPrimAtPath(p)
            if not prim:
                print(f"[WARN] Prim {p} not found, skipping bind")
                continue
            UsdShade.MaterialBindingAPI(prim).Bind(mat)

        print(f"✅  Bound dynamic://{tex_name} (UV set {uv_set_index}) "
            f"to {len(prim_paths)} prims")

        return provider            # handy for updates at runtime


    
    
    def create_green_material(self, height=50, width=300):
        
        prov = ui.DynamicTextureProvider(TEX_NAME)
        
        prov.set_data_array(                       # full green
            np.full((height, width, 4), [0, 255, 0, 255], dtype=np.uint8), [height,width,4])
        # prov.set_data_array(                       # simple green→white gradient
        #     np.dstack([
        #         np.tile(np.linspace(0,255,width,dtype=np.uint8)[:,None], (1,height)),  # R
        #         np.full((width,height),255,np.uint8),                                  # G
        #         np.tile(np.linspace(0,255,width,dtype=np.uint8)[:,None], (1,height)),  # B
        #         np.full((width,height),255,np.uint8)]).astype(np.uint8), [height,width,4])

        # 2) create Material + Shader
        mat = UsdShade.Material.Define(stage, MAT_PATH)
        shd = UsdShade.Shader.Define(stage, f"{MAT_PATH}/OmniPBR")
        shd.CreateIdAttr("OmniPBR")

        # ---- MDL linkage (safe order) ----------------------------------------
        shd.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shd.SetSourceAsset(Sdf.AssetPath(MDL_PATH), "mdl")
        shd.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

        # ---- inputs ----------------------------------------------------------
        shd.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset) \
        .Set(Sdf.AssetPath(f"dynamic://{TEX_NAME}"))
        # shd.CreateInput("uv_tiling",      Sdf.ValueTypeNames.Float2) \
        #    .Set(Gf.Vec2f(1, 1))
        # shd.CreateInput("project_uvw",    Sdf.ValueTypeNames.Bool).Set(True)
        # shd.CreateInput("world_space_uv", Sdf.ValueTypeNames.Bool).Set(True)
        # shd.CreateInput("roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)

        # ---- required OmniPBR outputs ----------------------------------------
        surf = shd.CreateOutput("surface",      Sdf.ValueTypeNames.Token)
        vol  = shd.CreateOutput("volume",       Sdf.ValueTypeNames.Token)
        disp = shd.CreateOutput("displacement", Sdf.ValueTypeNames.Token)

        mat.CreateSurfaceOutput().ConnectToSource(surf)
        mat.CreateVolumeOutput().ConnectToSource(vol)
        mat.CreateDisplacementOutput().ConnectToSource(disp)

        # 3) bind to the hull (or subset)
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(PRIM_PATH)).Bind(mat)

        print(f"✅  Material bound — PRIM {PRIM_PATH} now samples dynamic://{TEX_NAME}")




def make_green_to_white_gradient(tex_name="live_hull", width=300, height=50):
    """
    Creates (or overwrites) dynamic://<tex_name> with a vertical gradient:
        row-0   = pure green   [  0,255,  0]
        row-H-1 = pure white   [255,255,255]
    """
    # ---- 1. build an (H,1,3) RGB column ---------------------------------
    ramp = np.linspace(0, 255, height, dtype=np.uint8).reshape(height, 1, 1)
    r = ramp                 # 0 → 255
    g = np.full_like(r, 255) # constant 255
    b = ramp                 # 0 → 255
    rgb_col = np.concatenate([r, g, b], axis=2)   # (H,1,3)

    # ---- 2. repeat that column across the width --------------------------
    rgb = np.repeat(rgb_col, width, axis=1)       # (H,W,3)

    # ---- 3. add opaque alpha channel -------------------------------------
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    rgba  = np.concatenate([rgb, alpha], axis=2)  # (H,W,4)

    # ---- 4. upload to the dynamic-texture provider ------------------------
    provider = ui.DynamicTextureProvider(tex_name)  # no "dynamic://" prefix
    provider.set_data_array(rgba, [height, width, 4])


