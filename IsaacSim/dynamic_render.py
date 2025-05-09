import numpy as np
from pxr import UsdShade, Sdf

# 1. Create a dynamic texture provider with a unique name
dyn_tex_name = "paintTex"
dyn_tex = omni.ui.DynamicTextureProvider(dyn_tex_name)

# 2. Initialize the texture data (RGBA image) – e.g. white background 1024x1024
tex_width, tex_height = 1024, 1024
paint_color = (255, 0, 0, 255)       # RGBA for paint (red in this example)
bg_color    = (255, 255, 255, 255)   # RGBA background (white)
texture_data = np.full((tex_height, tex_width, 4), bg_color, dtype=np.uint8)

# Send the initial texture to GPU
dyn_tex.set_data_array(texture_data, [tex_width, tex_height])

# 3. Create an MDL material (OmniPBR) and assign the dynamic texture to it
stage = omni.usd.get_context().get_stage()
material_path = "/World/SurfaceMaterial"
material = UsdShade.Material.Define(stage, material_path)
shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")

# Configure the shader to use OmniPBR (which has a diffuse texture slot)
shader.SetSourceAsset("OmniPBR.mdl", "mdl")
shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
shader.CreateIdAttr("OmniPBR")

# Set the diffuse texture input to the dynamic texture (using dynamic:// scheme)
shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)\
      .Set(f"dynamic://{dyn_tex_name}")  # Link to our DynamicTextureProvider&#8203;:contentReference[oaicite:2]{index=2}

# Connect shader to material and bind to mesh
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
ground_prim = stage.GetPrimAtPath("/World/YourGroundMesh")  # path to the curved surface prim
ground_prim.ApplyAPI(UsdShade.MaterialBindingAPI)           # ensure binding API is present
UsdShade.MaterialBindingAPI(ground_prim).Bind(material)


from omni.physx import get_physx_scene_query_interface
import carb

# Assuming we have wheel world transform or known positions:
wheel_pos = <get wheel center World position as (x, y, z)>
origin = carb.Float3(*wheel_pos)
direction = carb.Float3(0, 0, -1)  # cast downward along -Z (assuming gravity is -Z)

hit = get_physx_scene_query_interface().raycast_closest(origin, direction, distance=10.0)
if hit["hit"]:
    hit_position = hit["position"]           # world-space (x,y,z) of contact
    hit_prim_path = hit["collision"]         # prim path of the collider (ground)
    hit_face_index = hit["faceIndex"]        # index of triangle on ground mesh&#8203;:contentReference[oaicite:9]{index=9}

from pxr import UsdGeom, Gf

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

points = mesh.GetPointsAttr().Get()
p0 = Gf.Vec3f(points[i0]);  p1 = Gf.Vec3f(points[i1]);  p2 = Gf.Vec3f(points[i2])
phit = Gf.Vec3f(*hit_position)  # PhysX gives a tuple, convert to Gf.Vec3f

# Compute barycentric coordinates of phit relative to triangle (p0,p1,p2):
v0 = p1 - p0
v1 = p2 - p0
v2 = phit - p0
d00 = Gf.Dot(v0, v0);  d01 = Gf.Dot(v0, v1);  d11 = Gf.Dot(v1, v1)
d20 = Gf.Dot(v2, v0);  d21 = Gf.Dot(v2, v1)
denom = d00 * d11 - d01 * d01
# Barycentric coordinates (u,v) for phit in triangle plane
v = (d11 * d20 - d01 * d21) / denom
w = (d00 * d21 - d01 * d20) / denom
u = 1.0 - v - w
# Now interpolate UV:
hit_uv = uv0 * u + uv1 * v + uv2 * w    # resulting UV2f

# Define brush radius in pixels (e.g., wheel radius projection)
brush_radius = 5  # pixels
for dy in range(-brush_radius, brush_radius+1):
    for dx in range(-brush_radius, brush_radius+1):
        if dx*dx + dy*dy <= brush_radius*brush_radius:  # inside circle
            x = px + dx
            y = py + dy
            if 0 <= x < tex_width and 0 <= y < tex_height:
                texture_data[y, x] = paint_color  # RGBA tuple defined earlier
                painted_mask[y, x] = 1            # mask to track coverage (see below)

# After painting into texture_data...
dyn_tex.set_data_array(texture_data, [tex_width, tex_height])

total_pixels = tex_width * tex_height
painted_pixels = np.count_nonzero(painted_mask)
coverage_percent = (painted_pixels / total_pixels) * 100.0
print(f"Coverage: {coverage_percent:.2f}%")
