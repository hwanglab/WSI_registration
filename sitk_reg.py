import numpy as np
import cv2
import SimpleITK as sitk

prefix = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/'
fixed_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/1664369_MR16-1693 J3_Tumor_CD3___thumbnail_tilesize_x-8-y-8.png'
moving_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/1664369_MR16-1693 J3_Tumor_HE___thumbnail_tilesize_x-8-y-8.png'
reg_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/reg1.png'

# fixedImage = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
# movingImage = sitk.ReadImage(moving_path, sitk.sitkFloat32)

fixedImage = cv2.imread(fixed_path, cv2.IMREAD_GRAYSCALE)
fixedImage = fixedImage.astype(np.float32)
fixedImage =sitk.GetImageFromArray(fixedImage)


# movingImage = sitk.ReadImage(moving_path, sitk.sitkFloat32)
movingImage = cv2.imread(moving_path, cv2.IMREAD_GRAYSCALE)
movingImage = movingImage.astype(np.float32)
movingImage =sitk.GetImageFromArray(movingImage)


elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(fixedImage)
elastixImageFilter.SetMovingImage(movingImage)

# parameterMap0 = sitk.ReadParameterFile(prefix+'TransformParameters.0.txt')
# parameterMap1 = sitk.ReadParameterFile("/path/to/TransformParameters.1.txt`)
# elastixImageFilter.SetParameterMap(parameterMap0)
# elastixImageFilter.Execute()
# rim = elastixImageFilter.GetResultImage()
# reg_img = sitk.Cast(sitk.RescaleIntensity(rim), sitk.sitkUInt8)
# sitk.WriteImage(reg_img, reg_path)

# parameterMapVector = sitk.VectorOfParameterMap()

# parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
# #parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
# parameterMapVector.append(sitk.GetDefaultParameterMap('nonrigid'))

# elastixImageFilter.SetParameterMap(parameterMapVector)

# params = sitk.GetDefaultParameterMap("bspline")
# params['GridSize'] ="4 134 77"
# params['GraidSpacing'] = "16 16 16"
# params['GridOrigin'] = "-22.5 -17.0 -16.5"
elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("bspline"))
# elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap("nonrigid"))

elastixImageFilter.SetParameter("MaximumNumberOfIterations", "500")
elastixImageFilter.SetParameter("NumberOfResolutions","4")
elastixImageFilter.SetParameter("FixedImagePyramid", "FixedSmoothingImagePyramid")

elastixImageFilter.SetParameter("FixedImagePyramidSchedule", "8 8 4 4 2 2 1 1")
elastixImageFilter.SetParameter("FinalGridSpacingInPhysicalUnits", "16")
# elastixImageFilter.SetParameter("GridSize", "4 143 77")
# elastixImageFilter.SetParameter("GridOrigin", "-22.5 -17.0 -16.5")
# pyramidsamples = []
# for i in range(5):
#     pyramidsamples.extend( [0]+[2**i]*2096 )
# pyramidsamples.reverse()
# elastixImageFilter.SetParameter("ImagePyramidSchedule", str(*pyramidsamples))

# elastixImageFilter.SetParameter("HowToCombineTransforms", "Compose")
sitk.PrintParameterMap(elastixImageFilter.GetParameterMap())
elastixImageFilter.Execute()
# reg_img = sitk.Cast(sitk.RescaleIntensity(elastixImageFilter.GetResultImage()), sitk.sitkUInt8)





transformParameterMap = elastixImageFilter.GetTransformParameterMap()
transformixImageFilter = sitk.TransformixImageFilter()
transformixImageFilter.SetTransformParameterMap(transformParameterMap)
# sitk.PrintParameterMap(transformParameterMap)
sitk.WriteParameterFile(transformParameterMap[0], prefix+'tranform.txt')
transformixImageFilter.LogToConsoleOn()


movingColor=cv2.imread(moving_path,cv2.IMREAD_COLOR)
movingB = sitk.GetImageFromArray(movingColor[:,:,0])
movingG = sitk.GetImageFromArray(movingColor[:,:,1])
movingR = sitk.GetImageFromArray(movingColor[:,:,2])
im2 =sitk.ReadImage('Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/1664369_MR16-1693 J3_Tumor_HE___thumbnail_tilesize_x-8-y-8.png', sitk.sitkFloat32)
transformixImageFilter.SetMovingImage(movingB)
transformixImageFilter.ComputeDeformationFieldOn()
transformixImageFilter.Execute()

deformation_file = prefix+'deformation.nii.gz'
sitk.WriteImage(transformixImageFilter.GetDeformationField(),deformation_file)
deformationField = transformixImageFilter.GetDeformationField()

# afine_tx =sitk.AffineTransform(sitk.Cast(deformationField, sitk.sitkVectorFloat64))
dis_tx=sitk.DisplacementFieldTransform(sitk.Cast(deformationField, sitk.sitkVectorFloat64))


# deform_inv1 = np.array(sitk.GetArrayFromImage(sitk.InverseDisplacementField(deformationField, size=deformationField.GetSize(), outputOrigin=(0.0, 0.0), outputSpacing=(1.0, 1.0))))




outB = transformixImageFilter.GetResultImage()
# outB = sitk.Cast(sitk.RescaleIntensity(transformixImageFilter.GetResultImage()), sitk.sitkUInt8)
# sitk.WriteImage(reg_img, prefix+'regB.jpg')



grid_image = sitk.GridSource(outputPixelType=sitk.sitkUInt16,
                             size=outB.GetSize(), 
                             sigma=(0.1,0.1), gridSpacing=(16.0,16.0))
grid_image.CopyInformation(outB)
deformArray = sitk.GetArrayFromImage(deformationField)
fixedArray =sitk.GetArrayFromImage(fixedImage)
movingArray = sitk.GetArrayFromImage(movingImage)
new_movingArray = sitk.GetArrayFromImage(grid_image) #np.zeros_like(movingArray)

for y in range(deformArray.shape[0]):
    for x in range(deformArray.shape[1]):
        nx,ny =  dis_tx.TransformPoint((x,y)) #(x - deformArray[(y,x)][0], y -deformArray[(y,x)][1]) #dis_tx.TransformPoint((x,y))
        if nx>=0 and nx < new_movingArray.shape[1]  and ny>=0 and ny < new_movingArray.shape[0]:
            new_i, new_j = int(np.floor(nx)), int(np.floor(ny))
            new_movingArray[(new_j, new_i)] = fixedArray[(y,x)]


cv2.imwrite(prefix+'new_moving.jpg', new_movingArray)




transformixImageFilter.SetMovingImage(movingG)
transformixImageFilter.Execute()
outG = transformixImageFilter.GetResultImage()
# outG = sitk.Cast(sitk.RescaleIntensity(transformixImageFilter.GetResultImage()), sitk.sitkUInt8)
# sitk.WriteImage(outG, prefix+'regB.jpg')

transformixImageFilter.SetMovingImage(movingR)
transformixImageFilter.Execute()
outR = transformixImageFilter.GetResultImage()
# outR = sitk.Cast(sitk.RescaleIntensity(transformixImageFilter.GetResultImage()), sitk.sitkUInt8)
# sitk.WriteImage(outR prefix+'regR.jpg')

out = np.dstack([sitk.GetArrayFromImage(outB), sitk.GetArrayFromImage(outG), sitk.GetArrayFromImage(outR)])
cv2.imwrite(prefix+'out.png', out)


# resampler = sitk.ResampleImageFilter()
# resampler.SetReferenceImage(deformationField)  # Or any target geometry
# resampler.SetTransform(sitk.DisplacementFieldTransform(
#     sitk.Cast(deformationField, sitk.sitkVectorFloat64)))
# warped = resampler.Execute(movingR)
# cv2.imwrite(prefix+'warped.jpg', sitk.GetArrayFromImage(warped))

grid_image = sitk.GridSource(outputPixelType=sitk.sitkUInt16,
                             size=movingImage.GetSize(), 
                             sigma=(0.1,0.1), gridSpacing=(16.0,16.0))
grid_image.CopyInformation(movingImage)

out_grid=sitk.Resample(grid_image, fixedImage, sitk.DisplacementFieldTransform(
    sitk.Cast(deformationField, sitk.sitkVectorFloat64)))
out_grid = sitk.Cast(sitk.RescaleIntensity(out_grid), sitk.sitkUInt8)
cv2.imwrite(prefix+'out_grid.png', sitk.GetArrayFromImage(out_grid))