import os
from pickletools import uint8

from sqlalchemy import false, values
from IPython.display import clear_output
from registration import *
from tqdm import tqdm

all_thumbnail_path = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/Van_Abel_all_thumbnail_images'

fixStain='HE'
movingStain='IHC'
d={}
for root,dirs, fnames in os.walk(all_thumbnail_path):
    if 'registration_' in root : continue
    for fname in fnames:
        if '.tsv' in fname: continue
        token = fname.split('__')[0]
        stainID = token.split('_')[-1]
        type = token.split('_')[-2]
        key = token.replace(stainID,'')
        
        d[key]=d.get(key,{fixStain:[],movingStain:[]})
        if fixStain in stainID:
            d[key][fixStain].append({'root': root, 'stain': stainID, 'fname':fname, 'type':type})
        else:
            d[key][movingStain].append({'root': root, 'stain': stainID, 'fname':fname, 'type':type})

fixed_path, moving_path=None,None
flag = False
for key in d:
    for fixed_item in tqdm(d[key][fixStain]):
        for moving_item in  d[key][movingStain]:


            fixed_path = os.path.join(fixed_item['root'], fixed_item['fname'])
            moving_path = os.path.join(moving_item['root'], moving_item['fname'])
            if '1664369' in moving_path and 'CD20' in moving_path and 'Tumor' in moving_path: flag = True
            if not flag: continue
            if fixed_path is None or moving_path is None: continue


            outdir = fixed_item['root']+os.sep+'registration_'+fixed_item['type']+'_'+fixed_item['stain']+'_'+moving_item['stain']
            if not os.path.exists(outdir): os.mkdir(outdir)

            # fixedImage = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
            # movingImage = sitk.ReadImage(moving_path, sitk.sitkFloat32)
            fixedImage=sitk.GetImageFromArray(cv2.imread(fixed_path, cv2.IMREAD_GRAYSCALE).astype(np.float32))
            movingImage=sitk.GetImageFromArray(cv2.imread(moving_path, cv2.IMREAD_GRAYSCALE).astype(np.float32))



            tx = initial_transform(fixedImage, movingImage) 
            movingResampled = sitk.Resample(movingImage, fixedImage, tx, sitk.sitkLinear, movingImage[0,0], movingImage.GetPixelID()) #default pixel=moving_image[0,0]
            # sitk.WriteImage(sitk.Cast(movingResampled, sitk.sitkUInt8),'a.png')
            out,outTx = bspline_registration(fixedImage, movingResampled)
            # outGrayArray, transformParameterMap= non_rigid_registration(fixedImage,movingResampled)#movingImage to fixedImage 
            # cv2.imwrite('./elastixOut.png',outGrayArray)

            
            movingRGB = cv2.imread(moving_path, cv2.IMREAD_COLOR)
            initTransformed, deformed= deform_array1(tx, outTx,movingRGB, fixedImage)    

            # initTransformed, deformed, deformation_field= deform_array1(tx, transformParameterMap,movingRGB, fixedImage)    
            # deformation_field = deformationFilter.GetDeformationField()
            # outpath=os.path.join(os.path.dirname(moving_path), 'deformed_'+os.path.basename(moving_path))


            '''inv_tx=tx.GetInverse()
            inv_deformation_field = inverse_deformationfield(deformation_field)

            fixedResampled = sitk.Resample(fixedImage, movingImage, inv_tx, sitk.sitkLinear, fixedImage[0,0], fixedImage.GetPixelID()) #default pixel=moving_image[0,0]
            fixedRGB = sitk.GetArrayFromImage(fixedResampled)
            deformed= deform_by_deformatinfield(inv_deformation_field, fixedRGB, movingImage)'''

    
            outpath = os.path.join(outdir, 'initTransformed.png')
            cv2.imwrite(outpath,initTransformed)
            outpath = os.path.join(outdir, 'deformed.png')
            cv2.imwrite(outpath,deformed)
            targetRGB = cv2.imread(fixed_path,cv2.IMREAD_COLOR)              
            blend = cv2.addWeighted(targetRGB.astype(np.float32), 0.6, deformed.astype(np.float32),0.4,0.0)
            outpath = os.path.join(outdir, 'blend.png')
            cv2.imwrite(outpath, blend)


            sitk.WriteTransform(tx,   os.path.join(outdir, 'initTrans.tfm')  ) #wirte inital affine transform
            # sitk.WriteParameterFile(transformParameterMap[0], os.path.join(outdir, 'TransformParameterMap.txt')) #write non-rgid trasnfom map 
            # deformation_file = os.path.join(outdir,'deformation_field.nii.gz')
            # sitk.WriteImage(deformation_field,deformation_file)
            # write_deform_field(tx, deformation_field,  outdir, fixedImage) #forwardfield  

            write_deform_field1(tx, outTx,  outdir, fixedImage) #forwardfield  


            f = open(os.path.join(outdir, 'pair.txt'),'w')
            f.write('fixed_image: '+ fixed_path+'\n')
            f.write('moving_image: '+ moving_path+'\n')
            f.close()



