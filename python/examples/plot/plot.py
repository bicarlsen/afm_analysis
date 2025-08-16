# %%
import bric_afm as afm

# %%
# load an image
img = afm.mfp3d.load_ibw("image.ibw")

# %%
# get an image channel
height = img["HeightTrace"]

# plot raw data
fig = afm.plot.plot_interactive(height)
fig.show()

# %%
# zero and plane level the image
height.apply(afm.operations.plane_level)
height.apply(afm.operations.min_to_zero)

# plot modified data
fig = afm.plot.plot_interactive(height)
fig.show()
# %%
