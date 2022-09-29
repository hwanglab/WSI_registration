from re import U
import numpy as np
import cv2
import SimpleITK as sitk

prefix = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/'
fixed_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/1664369_MR16-1693 J3_Tumor_CD3___thumbnail_tilesize_x-8-y-8.png'
moving_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/1664369_MR16-1693 J3_Tumor_HE___thumbnail_tilesize_x-8-y-8.png'
reg_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/reg1.png'


def non_rigid_registration(fixed_path, moving_path,  out_path, default_tranform='bspline', grid_size=16, NumberOfResolutions=4,  MaximumNumberOfIterations=500 ):
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
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap(default_tranform))
    # elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap("nonrigid"))

    elastixImageFilter.SetParameter("MaximumNumberOfIterations", str(MaximumNumberOfIterations))
    elastixImageFilter.SetParameter("NumberOfResolutions",str(NumberOfResolutions))
    elastixImageFilter.SetParameter("FixedImagePyramid", "FixedSmoothingImagePyramid")

    pyrsch=''
    for i in range(NumberOfResolutions):
        pyrsch+=str(pow(2,i))+' '+str(pow(2,i))+' '
    pyrsch=pyrsch[:-1]

    elastixImageFilter.SetParameter("FixedImagePyramidSchedule",pyrsch)#"8 8 4 4 2 2 1 1"
    elastixImageFilter.SetParameter("FinalGridSpacingInPhysicalUnits", str(grid_size))
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
    

    # deformation_file = prefix+'deformation.nii.gz'
    # sitk.WriteImage(transformixImageFilter.GetDeformationField(),deformation_file)
    # deformationField = transformixImageFilter.GetDeformationField()

    movingArray=cv2.imread(moving_path,cv2.IMREAD_COLOR)

    if len(movingArray.shape) == 3:
        outArray=[]
        for c in range(movingArray.shape[2]):
            movingChannelImage = sitk.GetImageFromArray(movingArray[:,:,c])
            transformixImageFilter.SetMovingImage(movingChannelImage)
            transformixImageFilter.ComputeDeformationFieldOn()
            transformixImageFilter.Execute()
            out = transformixImageFilter.GetResultImage()
            outArray.append(sitk.GetArrayFromImage(out))
        outImage = np.dstack(outArray)       
        
    else:
        outImage = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
        elastixImageFilter.Get
    
    
    cv2.imwrite(out_path, outImage)

    return  transformixImageFilter.GetDeformationField() # return deformationField





def write_deform_field(deformationField, prefix):
    # afine_tx =sitk.AffineTransform(sitk.Cast(deformationField, sitk.sitkVectorFloat64))
    dis_tx=sitk.DisplacementFieldTransform(sitk.Cast(deformationField, sitk.sitkVectorFloat64))


    grid_image = sitk.GridSource(outputPixelType=sitk.sitkUInt16,
                                size=deformationField.GetSize(), 
                                sigma=(0.1,0.1), gridSpacing=(16.0,16.0))
    # grid_image.CopyInformation(deformationField)\

   
    deformArray = sitk.GetArrayFromImage(deformationField)
    # grid_image = sitk.ReadImage('Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/1664369_MR16-1693 J3_Tumor_CD3___thumbnail_tilesize_x-8-y-8.png')

    SRCtoTRG = np.zeros((grid_image.GetSize()[1], grid_image.GetSize()[0])) #np.zeros_like(movingArray)
    

    f = open(prefix+'/'+'deformField.txt','w')
    f.write('#source(x,y)\t#target(x,y)\n')
    for y in range(deformArray.shape[0]):
        for x in range(deformArray.shape[1]):
            nx,ny =  dis_tx.TransformPoint((x,y)) #(x - deformArray[(y,x)][0], y -deformArray[(y,x)][1]) #dis_tx.TransformPoint((x,y))
            if nx>=0 and nx < SRCtoTRG.shape[1]  and ny>=0 and ny < SRCtoTRG.shape[0]:
                new_x, new_y = int(np.floor(nx)), int(np.floor(ny))
                SRCtoTRG[(new_y, new_x)] = grid_image[(x,y)]
                f.write(f'({x},{y})\t({new_x},{new_y})\n')
    f.close()
    cv2.imwrite(prefix+'/deformGrid.jpg', SRCtoTRG)


    # resampler = sitk.ResampleImageFilter()
    # resampler.SetReferenceImage(deformationField)  # Or any target geometry
    # resampler.SetTransform(sitk.DisplacementFieldTransform(
    #     sitk.Cast(deformationField, sitk.sitkVectorFloat64)))
    # warped = resampler.Execute(movingR)
    # cv2.imwrite(prefix+'warped.jpg', sitk.GetArrayFromImage(warped))




if __name__ == '__main__':
    import os

    choose = -1
    while (choose !=0 or choose !=1):
        choose=input("1 for Dir or 0 for image path:") or 0
        if choose =='0':
            fixed_path = input("Enter a ref. image path:") or 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/1664369_MR16-1693 J3_Tumor_CD3___thumbnail_tilesize_x-8-y-8.png'
            moving_path = input("Enter a moving image path:") or 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/1664369_MR16-1693 J3_Tumor_HE___thumbnail_tilesize_x-8-y-8.png'
            out_path = input("Enter an output path:") or 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/reg.png'
            deformation_field = non_rigid_registration(fixed_path,moving_path, out_path)
            write_deform_field(deformation_field,   '/'.join(out_path.split('/')[:-1]))
        elif choose =='1':
            dir_path = input("Enter a dir path including IHC and HE directories:") or 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info'
            d={}
            for root, dir, fnames in os.walk(dir_path):
                root=root.replace(os.sep,'/')
                for file in fnames:
                    if 'thumbnail_tilesize'  in file and '.png' in file:
                        parent_dir = root.split('/')[-2] 
                    
                        d[parent_dir] = d.get(parent_dir, {'fixed_path': '', 'moving_path': '', 'out_path':''})
                        if 'CD3' in root:
                            d[parent_dir]['fixed_path']=root+'/'+file
                            d[parent_dir]['out_path']='/'.join(root.split('/')[:-1])+'/'+'CD3_HE_registered.png'
                        elif 'HE' in  root:
                            d[parent_dir]['moving_path']=root+'/'+file

            for k, p in d.items():
                deformation_field = non_rigid_registration(p['fixed_path'],p['moving_path'], p['out_path'])
                write_deform_field(deformation_field,  prefix='/'.join(p['out_path'].split('/')[:-1]))



