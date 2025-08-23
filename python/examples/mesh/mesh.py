# %%
import bric_afm as afm

# %%
# load an image
img = afm.mfp3d.load_ibw("image.ibw")

# %%
# create mesh
# values should be one the order of 1.
scale = 1e9
z_scale = 5  # exaggerate z scaling
z = img["HeightTrace"].data * scale * z_scale
colors = img["UserIn1Trace"].data
mesh = afm.mesh.create_mesh(img.x * scale, img.y * scale, z - z.min(), colors)

# %%
# display mesh
mesh.show()

# %%
# add a conformal layer
thickness = 300
conf = afm.operations.add_conformal_layer(
    img.x, img.y, img["HeightTrace"].data, thickness, scale=scale
)

# %%
# display the surface with the conformal layer
scale = 1e9
z_scale = 5
crop = 10  # crop nan values
x = img.x[crop:-crop] * scale
y = img.y[crop:-crop] * scale
z = conf[crop:-crop, crop:-crop] * scale * z_scale
mesh = afm.mesh.create_mesh(x, y, z - z.min())
mesh.show()
# %%
