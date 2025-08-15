#%%
import bric_afm as afm

#%%
# load an image
img = afm.mfp3d.load_ibw("image.ibw")

# %%
# basic properties of the image
print(img.labels)
print(img.shape)
print(img.x[:10])
print(img.y[:10])

# %%
# get an image channel
height = img["HeightTrace"]

#%%
# zero and plane level the image
height.apply(afm.operations.plane_level)
height.apply(afm.operations.min_to_zero)
# %%
