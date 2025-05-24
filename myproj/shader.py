import omni.ui as ui
import numpy as np

from pxr import UsdShade, Sdf, Gf, Usd
import omni.ui as ui, numpy as np, omni.usd

# ----------------------------------------------------------------------
# CONFIG — change to your mesh or subset
PRIM_PATH = "/Hull"                       # where to bind the material
MDL_PATH  = "file:///opt/IsaacSim/kit/mdl/core/Base/OmniPBR.mdl"
TEX_NAME  = "live_hull"               # dynamic://live_hull
MAT_PATH  = "/Hull/Looks/HullDyna"
# ----------------------------------------------------------------------

stage = omni.usd.get_context().get_stage()

# 1) make sure the dynamic texture exists
height = 50
width = 300
prov = ui.DynamicTextureProvider(TEX_NAME)
prov.set_data_array(                       # simple green→white gradient
    np.dstack([
        np.tile(np.linspace(0,255,width,dtype=np.uint8)[:,None], (1,height)),  # R
        np.full((width,height),255,np.uint8),                                  # G
        np.tile(np.linspace(0,255,width,dtype=np.uint8)[:,None], (1,height)),  # B
        np.full((width,height),255,np.uint8)]).astype(np.uint8), [height,width,4])

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

# call it once
make_green_to_white_gradient()
print("✅  dynamic://live_hull now shows a green→white gradient")
