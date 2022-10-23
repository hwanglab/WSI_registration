import numpy as np
import SimpleITK as sitk
import os

transformDir='Z:/PUBLIC/lab_members/inyeop_jang/data/post_process/Van_Abel_registration_thumbnails'

path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/Van_Abel_IHC_thumbnails/1664369/1664369_MR16-1693 I4_LN_CD3.png'
path1 = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/Van_Abel_HE_thumbnails/1664369/1664369_MR16-1693 I4_LN_HE.png'
for root,dirs,fnames in os.walk(transformDir):
    initTrans, transParaMap = None, None
    for fname in fnames:
        if 'initTrans.tfm' in fname:
            initTrans = sitk.ReadTransform(os.path.join(root, fname))
        if 'TransformParameterMap' in fname:
            transParaMap = sitk.ReadParameterFile(os.path.join(root, fname))
        
        if initTrans and transParaMap:
            A=sitk.ReadImage(path, sitk.sitk)
            B=sitk.ReadImage(path1)
            r = sitk.ResampleImageFilter()
            r.SetReferenceImage(B)
            r.SetInterpolator(sitk.sitkLinear)
            r.SetDefaultPixelValue(255.0)
            r.SetTransform(initTrans)
            out = r.Execute(A) #default pixel=moving_image[0,0]


            
            myfilter=sitk.TransformixImageFilter()
            
            myfilter.SetTransformParameterMap(transParaMap)
            
            myfilter.SetMovingImage(out)
            myfilter.LogToConsoleOn()
            myfilter.Execute()
            sitk.WriteImage(myfilter.GetResultImage(),'test.png')



    


