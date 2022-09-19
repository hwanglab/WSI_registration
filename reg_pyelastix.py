import pyelastix
import numpy as np
# To read the image data we use imageio
import imageio
import cv2

# Pick one lib to visualize the result, matplotlib or visvis
#import visvis as plt
import matplotlib.pyplot as plt

prefix = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/tiles/cd3_he/'

# moving_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_CD3/1664369_MR16-1693 J3_Tumor_CD3___thumbnail_tilesize_x-8-y-8.png'
fixed_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/im1.jpg'
moving_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/im2.jpg'
reg_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/reg2.jpg'
# Read image data
im1 = imageio.imread(fixed_path)
im2 = imageio.imread(moving_path)
#im2 = imageio.imread('https://dl.dropboxusercontent.com/u/1463853/images/chelsea_morph1.png')

# Select one channel (grayscale), and make float
im1 = im1[:,:,2].astype('float32')
im2 = im2[:,:,2].astype('float32')

# Get default params and adjust
params = pyelastix.get_default_params(type='BSPLINE')
params.NumberOfResolutions = 5
params.MaximumNumberOfIterations = 500
params.SampleLastDimensionRandomly = False
print(params)
print('tempdir',pyelastix.get_tempdir())
# Register!
im3, field = pyelastix.register(im1, im2, params)

# target = np.zeros((im1.shape))
# y_field=field[0]
# x_field = field[1]
# for y in range(y_field.shape[0]):
#     for x in range(y_field.shape[1]):
#         new_y=y+y_field[y][x]*target.shape[0]
#         new_x=x+x_field[y][x]*target.shape[1]

#         if new_x >=0 and new_x < target.shape[1] and new_y >=0 and new_y < target.shape[0] :
#             target[y][x]=im3[y][x]

# cv2.imwrite(prefix +'target.png', target)

imageio.imwrite(reg_path, im3)
# Visualize the result
fig = plt.figure(1)
plt.clf()
plt.subplot(231); plt.imshow(im1)
plt.subplot(232); plt.imshow(im2)
plt.subplot(234); plt.imshow(im3)
plt.subplot(235); plt.imshow(field[0])
plt.subplot(236); plt.imshow(field[1])

# Enter mainloop
if hasattr(plt, 'use'):
    plt.use().Run()  # visvis
else:
    plt.show()  # mpl