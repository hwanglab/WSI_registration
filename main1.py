import os
from pickletools import uint8
from IPython.display import clear_output
from registration import *
import shutil

fixed_dir = 'Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/Van_Abel_IHC_thumbnails'
moving_dir ='Z:/PUBLIC/lab_members/inyeop_jang/data/organized_datasets/Van_Abel_HE_thumbnails'
out_dir = 'Z:/PUBLIC/lab_members/inyeop_jang/data/post_process/Van_Abel_registration_thumbnails'
til_dir ='Z:\PUBLIC\lab_members\inyeop_jang\data\post_process\VanAbel_Oropharynx_Tumor_TIL_Stroma\Van Abel-K_req32469_OPX TILs HE\pred_tils'
fixed_dir=fixed_dir.replace('/',os.sep)
moving_dir=moving_dir.replace('/',os.sep)
out_dir=out_dir.replace('/',os.sep)

if not os.path.exists(out_dir): os.mkdir(out_dir)


for pID in (os.listdir(moving_dir)):
    # clear_output(wait=True)
    if not os.path.exists(out_dir+'/'+pID): os.mkdir(out_dir+'/'+pID)
    

    dirMoving = os.path.join(moving_dir,pID)
    dirFixed = os.path.join(fixed_dir, pID)

    for fileMoving in os.listdir(dirMoving):
        if not '.png' in fileMoving: 
            if os.path.isdir(os.path.join(dirMoving,fileMoving)): 
                shutil.rmtree(os.path.join(dirMoving,fileMoving), ignore_errors=True)
            continue

        pathMoving = os.path.join(dirMoving,fileMoving)
        
        for fileFixed in os.listdir(dirFixed):
            if not '.png' in fileFixed: continue
            pathFixed = os.path.join(dirFixed,fileFixed)
            
            tokens = fileMoving.replace('.png','').split('_')
            stainID = tokens[-1]
            LN_Tumor = tokens[-2]

            if (LN_Tumor in fileMoving) and (pID in fileFixed) and (LN_Tumor in fileFixed) and (pID in fileMoving) :
                fixed_path = pathFixed
                moving_path = pathMoving


                outPatientDir = os.path.join(out_dir, pID)
                if not os.path.exists(outPatientDir): os.mkdir(outPatientDir)


                dirField = os.path.join(outPatientDir,LN_Tumor+'_'+stainID)
              
                if not os.path.exists(dirField): os.mkdir(dirField)

                out_path = os.path.join(dirField, fileMoving+'_to_'+fileFixed )


                

                fixedImage = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
                movingImage = sitk.ReadImage(moving_path, sitk.sitkFloat32)

                tx = initial_transform(fixedImage, movingImage) 
                movingResampled = sitk.Resample(movingImage, fixedImage, tx, sitk.sitkLinear, movingImage[0,0], movingImage.GetPixelID()) #default pixel=moving_image[0,0]
        
                outGrayArray, transformParameterMap= non_rigid_registration(fixedImage,movingResampled)#movingImage to fixedImage 
                # cv2.imwrite('./elastixOut.png',outGrayArray)





                movingRGB = cv2.imread(moving_path, cv2.IMREAD_COLOR)
                deformed_movingRGB, deformation_field= deform_array(tx, transformParameterMap,movingRGB, fixedImage)    
                # deformation_field = deformationFilter.GetDeformationField()
                # outpath=os.path.join(os.path.dirname(moving_path), 'deformed_'+os.path.basename(moving_path))
                outpath = os.path.join((os.sep).join(out_path.split(os.sep)[:-1]), 'deformed.png')
                cv2.imwrite(outpath,deformed_movingRGB)


                sitk.WriteTransform(tx,   os.path.join((os.sep).join(out_path.split(os.sep)[:-1]), 'initTrans.tfm')  )
                # write_deform_field(deformation_field,   '/'.join(out_path.split('/')[:-1])) #forwardfield                
                blend = cv2.addWeighted(cv2.imread(fixed_path).astype(np.float32), 0.6, deformed_movingRGB.astype(np.float32),0.4,0.0)
                tokens=out_path.split(os.sep)
                tokens[-1]='blend.png'
                cv2.imwrite((os.sep).join(tokens), blend)
                f = open((os.sep).join(out_path.split(os.sep)[:-1])+os.sep+'pair.txt','w')
                f.write('fixed_image: '+ fixed_path+'\n')
                f.write('moving_image: '+ moving_path+'\n')
                f.close()


                '''movingTIL = cv2.imread(os.path.join(til_dir,fileMoving.replace('.png','_color.png')), cv2.IMREAD_COLOR)
                movingTIL = cv2.resize(movingTIL,(deformed_movingRGB.shape[1], deformed_movingRGB.shape[0]))
                deformed_movingTIL, deformation_field = deform_array(tx, transformParameterMap,movingTIL, fixedImage)    
                # deformation_field = deformationFilter.GetDeformationField()
                # outpath=os.path.join(os.path.dirname(moving_path), 'deformed_'+os.path.basename(moving_path))
                outpath = os.path.join((os.sep).join(out_path.split(os.sep)[:-1]), 'deformedTIL.png')
                cv2.imwrite(outpath,deformed_movingTIL)'''



