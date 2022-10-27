from ctypes.wintypes import HACCEL
import numpy as np
import cv2
from skimage.color import rgb2hed, hed2rgb
import SimpleITK as sitk
import os
import pandas as pd
from tqdm import tqdm

# from skimage import data
print(os.getcwd())


prefix = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/'
fixed_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/1664369_MR16-1693 J3_Tumor_CD3___thumbnail_tilesize_x-8-y-8.png'
moving_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/1664369_MR16-1693 J3_Tumor_HE___thumbnail_tilesize_x-8-y-8.png'
reg_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/sample/info/1664369_MR16-1693 J3_Tumor_HE/reg1.png'

def get_rgb2hed(thumbnail, write_basepath=None)->list: 
    'rgb -> [hematoxylin, Eosin, DAB]'

    ihc_hed = rgb2hed(np.array(thumbnail))

    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)) # hematoxylin
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1)) #Eosin
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)) #DAB

    if write_basepath:
        # thumbnail.save(write_basepath+'/'+'hed.png')
        cv2.imwrite(write_basepath+'/'+'ihc_h.png',ihc_h*255)
        cv2.imwrite(write_basepath+'/'+'ihc_e.png',ihc_e*255)
        cv2.imwrite(write_basepath+'/'+'ihc_d.png',ihc_d*255)
    
    return [ihc_h, ihc_e, ihc_d]


def unmixHE(img, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(np.float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    # if saveFile is not None:
        # Image.fromarray(Inorm).save(saveFile+'.png')
        # Image.fromarray(H).save(saveFile+'_H.png')
        # Image.fromarray(E).save(saveFile+'_E.png')

    return Inorm, H, E


def initial_transform(fixed_image, moving_image):
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.AffineTransform(2), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    # moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate = 1.0, numberOfIterations = 100, convergenceMinimumValue = 1e-6, convergenceWindowSize = 10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [2, 1]) #[8, 4, 2, 1]
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [1, 0])#[3, 2, 1, 0]
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace = False)

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))
       
    return final_transform


def bspline_registration(fixed, moving):
    transformDomainMeshSize=[8]*moving.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed,
                                        transformDomainMeshSize )

    print("Initial Parameters:");
    print(tx.GetParameters())

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()

    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                        numberOfIterations=100,
                        maximumNumberOfCorrections=5,
                        maximumNumberOfFunctionEvaluations=1000,
                        costFunctionConvergenceFactor=1e+7)
    R.SetInitialTransform(tx, True)
    R.SetInterpolator(sitk.sitkLinear)
    

    # R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

    outTx = R.Execute(fixed, moving)
    # a=outTx.TransformPoint((0,0))

    print("-------")
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))


    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    # resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)
    

    out = resampler.Execute(moving)
    return out,outTx

def non_rigid_registration(fixedImage, movingImage,  default_tranform='bspline', grid_size=16, NumberOfResolutions=4,  MaximumNumberOfIterations=500 ):
    '!processed in grayscale!'
    # fixedImage = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    # movingImage = sitk.ReadImage(moving_path, sitk.sitkFloat32)



    # _,fixedH,_ = unmixHE(fixedArray)
    # fixedH = cv2.cvtColor(fixedH, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite('H.png',H)
    #H=cv2.imread('H.png',cv2.IMREAD_GRAYSCALE)
    # fixedImage =sitk.GetImageFromArray(fixedArray)


    # # Separate the stains from the IHC image
    # ihc_hed = rgb2hed(movingArray)
    # # Create an RGB image for each of the stains
    # null = np.zeros_like(ihc_hed[:, :, 0])
    # ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)) # hematoxylin
    # movingH = (ihc_h[:,:,0]*255.0).astype(np.uint8)
    # # cv2.imwrite('./ihc_h.png',movingImage)
    # # movingImage = cv2.imread('./ihc_h.png',cv2.IMREAD_GRAYSCALE)
    # movingImage =sitk.GetImageFromArray(movingArray)
    
    elastixImageFilter = sitk.ElastixImageFilter()
    
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    # sitk.WriteImage(sitk.Cast(fixedImage,sitk.sitkUInt8), './fixedImage.png')
    # sitk.WriteImage(sitk.Cast(movingImage, sitk.sitkUInt8), './movingImage.png')

    parameterMapVector = sitk.VectorOfParameterMap()
    # parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    # parameterMapVector.append(sitk.GetDefaultParameterMap('nonrigid'))
    elastixImageFilter.SetParameterMap(parameterMapVector)
    # elastixImageFilter.SetParameter("FixedImageDimension","2")
    # elastixImageFilter.SetParameter("MovingImageDimension","2")
    # elastixImageFilter.SetParameter("AutomaticParameterEstimation", "true")

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
#    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap(default_tranform))
    # elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap("nonrigid"))

#    elastixImageFilter.SetParameter("MaximumNumberOfIterations", str(MaximumNumberOfIterations))
#    elastixImageFilter.SetParameter("NumberOfResolutions",str(NumberOfResolutions))
#    elastixImageFilter.SetParameter("FixedImagePyramid", "FixedSmoothingImagePyramid")
#    elastixImageFilter.SetParameter("FixedImagePyramid", "FixedShrinkingImagePyramid")
#    elastixImageFilter.SetParameter("MovingImagePyramid", "MovingShrinkingImagePyramid")


    # pyrsch=''
    # for i in reversed(range(NumberOfResolutions)):
    #     pyrsch+=str(pow(2,i))+' '+str(pow(2,i))+' '
    # pyrsch=pyrsch[:-1]
    # elastixImageFilter.SetParameter("ImagePyramidSchedule",'16 16 16 8 8 8 4 4 4 2 2 2 1 1 1')#"8 8 4 4 2 2 1 1"
    # elastixImageFilter.SetParameter("FixedImagePyramidSchedule",'16 16 8 8 4 4 2 2 1 1')#"8 8 4 4 2 2 1 1"
    # elastixImageFilter.SetParameter("FinalGridSpacingInPhysicalUnits", str(grid_size))
    # elastixImageFilter.SetParameter("GridSize", "4 143 77")
    # elastixImageFilter.SetParameter("GridOrigin", "-22.5 -17.0 -16.5")
    # pyramidsamples = []
    # for i in range(5):
    #     pyramidsamples.extend( [0]+[2**i]*2096 )
    # pyramidsamples.reverse()
    # elastixImageFilter.SetParameter("ImagePyramidSchedule", str(*pyramidsamples))

    # elastixImageFilter.SetParameter("HowToCombineTransforms", "Compose")
    # params = elastixImageFilter.GetParameterMap()

    sitk.PrintParameterMap(elastixImageFilter.GetParameterMap())
    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.Execute()
    # reg_img = sitk.Cast(sitk.RescaleIntensity(elastixImageFilter.GetResultImage()), sitk.sitkUInt8)




    transformParameterMap = elastixImageFilter.GetTransformParameterMap()


    # sitk.PrintParameterMap(transformParameterMap)
    

    

    # deformation_file = prefix+'deformation.nii.gz'
    # sitk.WriteImage(transformixImageFilter.GetDeformationField(),deformation_file)
    # deformationField = transformixImageFilter.GetDeformationField()

    # movingArray=cv2.imread(moving_path,cv2.IMREAD_COLOR)

    outGrayArray = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())

    return outGrayArray, transformParameterMap# return deformationField




def deform_array(init_tx, transformParameterMap, movingArray:np.array, refImage:sitk.Image):
    if len(movingArray.shape) ==2:
        movingArray=np.expand_dims(movingArray,axis=2)
   
    initArray, outArray=[],[]
    for c in range(movingArray.shape[2]):
        movingChannelImage = sitk.GetImageFromArray(movingArray[:,:,c])
        movingResampled = sitk.Resample(movingChannelImage, refImage, init_tx, sitk.sitkLinear, movingChannelImage[0,0], movingChannelImage.GetPixelID()) #default pixel=moving_image[0,0]
        initArray.append(sitk.GetArrayFromImage(movingResampled))
       
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(transformParameterMap)
        
        transformixImageFilter.SetMovingImage(movingResampled)
        transformixImageFilter.ComputeDeformationFieldOn()
        transformixImageFilter.Execute()
        out = transformixImageFilter.GetResultImage()



        outArray.append(sitk.GetArrayFromImage(out))
    transformedArray = np.dstack(initArray)#intial rigid transformed image, before deformation     
    deformedArray = np.dstack(outArray)#final deformed image, after rigid, norigid transformation       
    return np.squeeze(transformedArray),np.squeeze(deformedArray), transformixImageFilter.GetDeformationField()

def deform_array1(init_tx, outTx, movingArray:np.array, refImage:sitk.Image):
    if len(movingArray.shape) ==2:
        movingArray=np.expand_dims(movingArray,axis=2)
   
    initArray, outArray=[],[]
    for c in range(movingArray.shape[2]):
        movingChannelImage = sitk.GetImageFromArray(movingArray[:,:,c])
        movingResampled = sitk.Resample(movingChannelImage, refImage, init_tx, sitk.sitkLinear, movingChannelImage[0,0], movingChannelImage.GetPixelID()) #default pixel=moving_image[0,0]
        initArray.append(sitk.GetArrayFromImage(movingResampled))
       
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(refImage)
        resampler.SetInterpolator(sitk.sitkLinear)
        # resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(outTx)
        

        out = resampler.Execute(movingResampled)
        outArray.append(sitk.GetArrayFromImage(out))
    transformedArray = np.dstack(initArray)#intial rigid transformed image, before deformation     
    deformedArray = np.dstack(outArray)#final deformed image, after rigid, norigid transformation       
    return np.squeeze(transformedArray),np.squeeze(deformedArray)
        
   
    
def write_deform_field1(init_trans, dis_tx, prefix, fixedImage):

    grid_image = sitk.GridSource(outputPixelType=sitk.sitkUInt16,
                                size=fixedImage.GetSize(), 
                                sigma=(0.1,0.1), gridSpacing=(32.0,32.0))
    # grid_image.CopyInformation(deformationField)\
    gridArr = np.zeros((grid_image.GetSize()[1], grid_image.GetSize()[0])) 


   
    deformArray = sitk.GetArrayFromImage(fixedImage)

    # grid_image = sitk.ReadImage('Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/Van_Abel_HE_thumbnails/1664369/1664369_MR16-1693 I4_LN_HE.png')
    SRCtoTRG = np.zeros_like(sitk.GetArrayFromImage(fixedImage))

    df=pd.DataFrame(index=range(deformArray.shape[0]*deformArray.shape[1]), columns=['source_x','source_y', 'target_x','target_y'])
    print(prefix)
    index=0
    for y in tqdm(range(deformArray.shape[0])):
        for x in range(deformArray.shape[1]):
            tx, ty = init_trans.TransformPoint((x,y))
            tx, ty = int(np.floor(tx)), int(np.floor(ty))
            if tx<0 or tx >= deformArray.shape[1]  or ty<0 or ty >= deformArray.shape[0]: continue

            nx,ny =  dis_tx.TransformPoint((tx,ty)) #(x - deformArray[(y,x)][0], y -deformArray[(y,x)][1]) #dis_tx.TransformPoint((x,y))
            if nx>=0 and nx < SRCtoTRG.shape[1]  and ny>=0 and ny < SRCtoTRG.shape[0]:
                new_x, new_y = int(np.floor(nx)), int(np.floor(ny))
                SRCtoTRG[(new_y, new_x)] = fixedImage[(x,y)]
                gridArr[(new_y, new_x)] = grid_image[(x,y)]
                df.loc[index, ['source_x','source_y', 'target_x','target_y']] = [x,y, new_x,new_y] #fixed -> moved
                index +=1
    

    df.to_csv(prefix+'/deformField.csv', index=False)
    cv2.imwrite(prefix+'/displaceFixedImage.jpg', SRCtoTRG)
    cv2.imwrite(prefix+'/deformGrid.jpg', gridArr.astype(np.uint8))



    # resampler = sitk.ResampleImageFilter()
    # resampler.SetReferenceImage(deformationField)  # Or any target geometry
    # resampler.SetTransform(sitk.DisplacementFieldTransform(
    #     sitk.Cast(deformationField, sitk.sitkVectorFloat64)))
    # warped = resampler.Execute(movingR)
    # cv2.imwrite(prefix+'warped.jpg', sitk.GetArrayFromImage(warped))



def write_deform_field(init_trans, deformationField, prefix, fixedImage):
    # afine_tx =sitk.AffineTransform(sitk.Cast(deformationField, sitk.sitkVectorFloat64))
    dis_tx=sitk.DisplacementFieldTransform(sitk.Cast(deformationField, sitk.sitkVectorFloat64))


    grid_image = sitk.GridSource(outputPixelType=sitk.sitkUInt16,
                                size=deformationField.GetSize(), 
                                sigma=(0.1,0.1), gridSpacing=(32.0,32.0))
    # grid_image.CopyInformation(deformationField)\
    gridArr = np.zeros((grid_image.GetSize()[1], grid_image.GetSize()[0])) 


   
    deformArray = sitk.GetArrayFromImage(deformationField)

    # grid_image = sitk.ReadImage('Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/Van_Abel_HE_thumbnails/1664369/1664369_MR16-1693 I4_LN_HE.png')
    SRCtoTRG = np.zeros_like(sitk.GetArrayFromImage(fixedImage))

    df=pd.DataFrame(index=range(deformArray.shape[0]*deformArray.shape[1]), columns=['source_x','source_y', 'target_x','target_y'])
    print(prefix)
    index=0
    for y in tqdm(range(deformArray.shape[0])):
        for x in range(deformArray.shape[1]):
            tx, ty = init_trans.TransformPoint((x,y))
            tx, ty = int(np.floor(tx)), int(np.floor(ty))
            if tx<0 or tx >= deformArray.shape[1]  or ty<0 or ty >= deformArray.shape[0]: continue

            nx,ny =  dis_tx.TransformPoint((tx,ty)) #(x - deformArray[(y,x)][0], y -deformArray[(y,x)][1]) #dis_tx.TransformPoint((x,y))
            if nx>=0 and nx < SRCtoTRG.shape[1]  and ny>=0 and ny < SRCtoTRG.shape[0]:
                new_x, new_y = int(np.floor(nx)), int(np.floor(ny))
                SRCtoTRG[(new_y, new_x)] = fixedImage[(x,y)]
                gridArr[(new_y, new_x)] = grid_image[(x,y)]
                df.loc[index, ['source_x','source_y', 'target_x','target_y']] = [x,y, new_x,new_y]
                index +=1
    

    df.to_csv(prefix+'/deformField.csv', index=False)
    cv2.imwrite(prefix+'/displaceFixedImage.jpg', SRCtoTRG)
    cv2.imwrite(prefix+'/deformGrid.jpg', gridArr.astype(np.uint8))



    # resampler = sitk.ResampleImageFilter()
    # resampler.SetReferenceImage(deformationField)  # Or any target geometry
    # resampler.SetTransform(sitk.DisplacementFieldTransform(
    #     sitk.Cast(deformationField, sitk.sitkVectorFloat64)))
    # warped = resampler.Execute(movingR)
    # cv2.imwrite(prefix+'warped.jpg', sitk.GetArrayFromImage(warped))

def inverse_deformationfield(deformationField):
    for y in range(deformationField.GetSize()[1]):
        for x in range(deformationField.GetSize()[0]):
            deformationField[x,y]=(-deformationField[x,y][0], -deformationField[x,y][1])
    return deformationField

def deform_by_deformatinfield(deformationField:sitk.Image, sourceArray:np.array, refImage:sitk.Image=None):
    resampler = sitk.ResampleImageFilter()

    
    # deformationField = sitk.GetImageFromArray(deformationArray)
    if refImage == None: refImage=deformationField
    resampler.SetReferenceImage(refImage)  # Or any target geometry
    resampler.SetTransform(sitk.DisplacementFieldTransform(
    sitk.Cast(deformationField, sitk.sitkVectorFloat64)))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    if len(sourceArray.shape)==2: #if grayimage
        sourceArray = np.expand_dims(sourceArray,axis=2)


    outChannel=[]
    for C in range(sourceArray.shape[2]):
        deformed = sitk.GetArrayFromImage(resampler.Execute(sitk.GetImageFromArray(sourceArray[:,:,C])))
        outChannel.append(deformed)
    outArray = np.dstack(outChannel).squeeze()
    # cv2.imwrite('warped1664369.jpg', RGB)
    return outArray



def load_IHC_HE(fixed_path, moving_path):

    # Example IHC image
    # ihc_rgb = data.immunohistochemistry()
    fixedArray = cv2.imread(fixed_path, cv2.IMREAD_COLOR) 
    # movingImage = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    movingArray = cv2.imread(moving_path, cv2.IMREAD_COLOR) #IHC

    if 'HE' in moving_path:
        IHC = fixedArray
        HE = movingArray
    else:
        IHC = movingArray
        HE = fixedArray

    # Separate the stains from the IHC image
    ihc_hed = rgb2hed(IHC)
    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)) # hematoxylin
    fixedIHC = (ihc_h[:,:,0]*255.0).astype(np.uint8)


    _,movingH,_ = unmixHE(HE)
    movingHE = cv2.cvtColor(movingH, cv2.COLOR_RGB2GRAY)


    return fixedIHC, movingHE



if __name__ == '__main__':
    import os

    choose = -1
    while (choose !='0' and choose !='1'):
        choose=input("1 for Dir or 0 for image path:") or 0
        if choose =='0':
            fixed_path = input("Enter a ref. image path:") or '4216530_4216530_MR14-3865 G7_Tumor_HE.png'
            moving_path = input("Enter a moving image path:") or '4216530_MR14-3865 G7_Tumor_CD3.png'
            out_path = input("Enter an output path:") or './reg1.png'

            # fixedIHC, movingHE = load_IHC_HE(fixed_path, moving_path)
            fixedArray = cv2.imread(fixed_path, cv2.IMREAD_GRAYSCALE)
            movingArray = cv2.imread(moving_path, cv2.IMREAD_GRAYSCALE)

            fixedImage = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
            movingImage = sitk.ReadImage(moving_path, sitk.sitkFloat32)

            tx = initial_transform(fixedImage, movingImage) #RGB 채널로 3번 해야함 추가하기
            movingResampled = sitk.Resample(movingImage, fixedImage, tx, sitk.sitkLinear, movingImage[0,0], movingImage.GetPixelID()) #default pixel=moving_image[0,0]
     
            outGrayArray, transformParameterMap= non_rigid_registration(fixedImage,movingResampled, out_path)#movingImage to fixedImage 
            cv2.imwrite('./elastixOut.png',outGrayArray)

            movingRGB = cv2.imread(moving_path, cv2.IMREAD_COLOR)
            deformed_movingRGB, deformation_field= deform_array(tx, transformParameterMap,movingRGB, fixedImage)    
            # deformation_field = deformationFilter.GetDeformationField()
            outpath=os.path.join(os.path.dirname(moving_path), 'deformed_'+os.path.basename(moving_path))
            cv2.imwrite(outpath,deformed_movingRGB)
            sitk.WriteTransform(tx,'/'.join(out_path.split('/')[:-1])+'/initTrans.tfm')
            write_deform_field(tx, deformation_field,   '/'.join(out_path.split('/')[:-1]), fixedImage) #forwardfield



            '''inv_deformation_field = inverse_deformationfield(deformation_field)#fixedImage to movingImage #backwardfield
            sourceRGB = cv2.imread(fixed_path, cv2.IMREAD_COLOR)
            deformed_fixedRGB = deform_by_deformatinfield(deformation_field, sourceRGB, refImage=sitk.GetImageFromArray(movingHE))
            outpath=os.path.join(os.path.dirname(fixed_path), 'deformed_'+os.path.basename(fixed_path))
            cv2.imwrite(outpath,deformed_fixedRGB)'''



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

                fixedIHC, movingHE = load_IHC_HE(fixed_path, moving_path)

                outArray, deformation_field = non_rigid_registration(fixedIHC,movingHE)
                write_deform_field(deformation_field,  prefix='/'.join(p['out_path'].split('/')[:-1]))



