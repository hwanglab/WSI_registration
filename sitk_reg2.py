import numpy as np
import cv2 
import SimpleITK as sitk
fixed_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/im1.jpg'
moving_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/im2.jpg'
reg_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/regB.jpg'

# fixedImage = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
fixedImage = cv2.imread(fixed_path, cv2.IMREAD_GRAYSCALE)
fixedImage = fixedImage.astype(np.float32)
fixedImage =sitk.GetImageFromArray(fixedImage)


# movingImage = sitk.ReadImage(moving_path, sitk.sitkFloat32)
movingImage = cv2.imread(moving_path, cv2.IMREAD_GRAYSCALE)
movingImage = movingImage.astype(np.float32)
movingImage =sitk.GetImageFromArray(movingImage)



# fixedColor = cv2.imread(fixed_path)
# fixedColor =np.rollaxis(fixedColor,2)
# fixedB = fixedColor[:,:,0].astype(np.float32)
# fixedG = fixedColor[:,:,1].astype(np.float32)
# fixedR = fixedColor[:,:,2].astype(np.float32)

movingColor = cv2.imread(moving_path)
# movingColor =np.rollaxis(movingColor,2)
movingB = movingColor[:,:,0].astype(np.float32)
movingG = movingColor[:,:,1].astype(np.float32)
movingR = movingColor[:,:,2].astype(np.float32)

# fixedR =sitk.GetImageFromArray(fixedR)
# fixedG =sitk.GetImageFromArray(fixedG)
# fixedB =sitk.GetImageFromArray(fixedB)

movingR =sitk.GetImageFromArray(movingR)
movingG =sitk.GetImageFromArray(movingG)
movingB =sitk.GetImageFromArray(movingB)


# Define a simple callback which allows us to monitor registration progress.
def iteration_callback(filter):
    print('\r{0:.2f}'.format(filter.GetMetricValue()), end='')

registration_method = sitk.ImageRegistrationMethod()
    
# Determine the number of BSpline control points using the physical 
# spacing we want for the finest resolution control grid. 
grid_physical_spacing = [16.0,16.0, 16.0] # A control point every 50mm
image_physical_size = [size*spacing for size,spacing in zip(fixedImage.GetSize(), fixedImage.GetSpacing())]
mesh_size = [int(image_size/grid_spacing + 0.5) \
             for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
# The starting mesh size will be 1/4 of the original, it will be refined by 
# the multi-resolution framework.
mesh_size = [int(sz/4 + 0.5) for sz in mesh_size]

initial_transform = sitk.BSplineTransformInitializer(image1 = fixedImage, 
                                                     transformDomainMeshSize = mesh_size, order=3)    
# Instead of the standard SetInitialTransform we use the BSpline specific method which also
# accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with 
# the given mesh_size at the highest pyramid level then we double it in the next lower level and
# in the full resolution image we use a mesh that is four times the original size.
registration_method.SetInitialTransformAsBSpline(initial_transform,
                                                 inPlace=False,
                                                 scaleFactors=[1,2,4])


registration_method.SetMetricAsMeanSquares()
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)
# registration_method.SetMetricFixedMask(fixedImage_mask)
    
registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

registration_method.SetInterpolator(sitk.sitkLinear)
registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-2, numberOfIterations=500, deltaConvergenceTolerance=0.01)

registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(registration_method))

final_transformation = registration_method.Execute(fixedImage, movingImage)
print('\nOptimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))



resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixedImage)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(1)
resampler.SetTransform(final_transformation)

outB = resampler.Execute(movingB)
outG = resampler.Execute(movingG)
outR = resampler.Execute(movingR)

# simg1 = sitk.Cast(sitk.RescaleIntensity(fixedImage), sitk.sitkUInt8)
# simg2 = sitk.Cast(sitk.RescaleIntensity(out1), sitk.sitkUInt8)
# reg_img = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
# sitk.Show(cimg, "ImageRegistration2 Composition")
#sitk.WriteImage(simg2, reg_path)

# reg_img = sitk.Cast(sitk.RescaleIntensity(registration_method.GetResultImage()), sitk.sitkUInt8)
simgB=sitk.Cast(sitk.RescaleIntensity(outB), sitk.sitkUInt8)
simgG=sitk.Cast(sitk.RescaleIntensity(outG), sitk.sitkUInt8)
simgR=sitk.Cast(sitk.RescaleIntensity(outR), sitk.sitkUInt8)

rgb = np.dstack([sitk.GetArrayFromImage(simgB),sitk.GetArrayFromImage(simgG), sitk.GetArrayFromImage(simgR) ])
cv2.imwrite(reg_path, rgb)

