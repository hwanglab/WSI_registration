import numpy as np
import cv2
import SimpleITK as sitk

prefix = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/'
fixed_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_CD3/1664369_MR16-1693 J3_Tumor_CD3___thumbnail_tilesize_x-8-y-8.png'
moving_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/1664369_MR16-1693 J3_Tumor_HE___thumbnail_tilesize_x-8-y-8.png'
reg_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/reg1.png'

fixedImage = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
movingImage = sitk.ReadImage(moving_path, sitk.sitkFloat32)

# resultImage=sitk.Elastix(fixedImage, movingImage, 'nonrigid')
# reg_img = sitk.Cast(sitk.RescaleIntensity(resultImage), sitk.sitkUInt8)
# sitk.WriteImage(reg_img, reg_path)

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


transformParameterMap = elastixImageFilter.GetTransformParameterMap()
transformixImageFilter = sitk.TransformixImageFilter()
transformixImageFilter.SetTransformParameterMap(transformParameterMap)
# sitk.PrintParameterMap(transformParameterMap)

im2 =sitk.ReadImage('Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/1664369_MR16-1693 J3_Tumor_HE___thumbnail_tilesize_x-8-y-8.png', sitk.sitkFloat32)
transformixImageFilter.SetMovingImage(im2)
transformixImageFilter.Execute()
reg_img = sitk.Cast(sitk.RescaleIntensity(transformixImageFilter.GetResultImage()), sitk.sitkUInt8)

# reg_img = sitk.Cast(sitk.RescaleIntensity(elastixImageFilter.GetResultImage()), sitk.sitkUInt8)

sitk.WriteImage(reg_img, reg_path)