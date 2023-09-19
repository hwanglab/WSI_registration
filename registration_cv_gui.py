import sys
import os
sys.path.append(os.path.dirname(__file__))
import cv2
import numpy as np
import h5py

path_HT = '/home/m264377/hwang_group/PUBLIC/Projects/StMary_GC_MolSubtypes_05-2023/HT/processed_data/StMary_reconstructions/BF/ST_Mary_Mol_53p/230823.105512.ST_Mary_Mol_53p.001.Group1.A1R1.T001P01/230823.105512.ST_Mary_Mol_53p.001.Group1.A1R1.T001P01.h5'
path_HE = '/home/m264377/hwang_group/PUBLIC/Projects/StMary_GC_MolSubtypes_05-2023/HE_images/processed_data/registration/registrated_images/53p.png'

SCALE_DOWN_RATE_SRC = 15.0
SCALE_DOWN_RATE_TRG = 2.0

def load_hdf5_with_attributes(filepath):
    file = h5py.File(filepath, 'r')
    attributes = {}
    for key in file.attrs.keys():
        if isinstance(file.attrs[key], np.ndarray):
            attributes[key] = file.attrs[key]
        else:
            attributes[key] = str(file.attrs[key])
    data = file['data'][()]
    file.close()
    return data, attributes




# Open the image files.

img1_color, _ = load_hdf5_with_attributes(path_HT)  # Image to be aligned.
if len(img1_color.shape)==3: #for multi-channeled HT
     choose_channel= int(img1_color.shape[2]/2) #middle channel
     img1_color=img1_color[:,:,choose_channel]

img1_gray = img1_color.astype(np.uint8)
img1_color=cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)
img2_color = cv2.imread(path_HE)    # Reference image.





# Convert to grayscale.
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)

height, width, _ = img1.shape
img1 = cv2.resize(img1, (int(width/SCALE_DOWN_RATE_SRC),int(height/SCALE_DOWN_RATE_SRC)))
img1_color = cv2.resize(img1_color, (int(width/SCALE_DOWN_RATE_SRC),int(height/SCALE_DOWN_RATE_SRC)))
height, width, _ = img2.shape
img2 = cv2.resize(img2, (int(width/SCALE_DOWN_RATE_TRG),int(height/SCALE_DOWN_RATE_TRG)))
img2_color = cv2.resize(img2_color, (int(width/SCALE_DOWN_RATE_TRG),int(height/SCALE_DOWN_RATE_TRG)))



posList_source = []
def onMouse1(event, x, y, flags, param):
   global posList_source
   global img1_color
   temp = img1_color.copy()
   
   if event == cv2.EVENT_LBUTTONDOWN:
        posList_source.append((x, y))
        print('S', (x,y))
   elif event == cv2.EVENT_RBUTTONDOWN:
        posList_source.pop()

   for i, point in enumerate(posList_source):
        temp = cv2.circle(temp, (point[0],point[1]), 10, (255,0,0), -1)
        temp = cv2.putText(temp, str(i), (point[0]+1,point[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
   
   cv2.imshow("image1", temp)

posList_target = []
def onMouse2(event, x, y, flags, param):
   global posList_target
   global img2_color
   temp = img2_color.copy()

   if event == cv2.EVENT_LBUTTONDOWN:
        posList_target.append((x, y))
        print('T', (x,y))
   elif event == cv2.EVENT_RBUTTONDOWN:
        posList_target.pop()

   for i, point in enumerate(posList_target):
        temp = cv2.circle(temp, (point[0],point[1]), 10, (0,0,255), -1)
        temp = cv2.putText(temp, str(i), (point[0]+1,point[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)

   cv2.imshow("image2", temp)



cv2.imshow("image1", img1_color)
cv2.setMouseCallback('image1', onMouse1)
cv2.imshow("image2", img2_color)
cv2.setMouseCallback('image2', onMouse2)

cv2.waitKey(0)
cv2.destroyAllWindows()








posList_s = np.array(posList_source)
posList_t = np.array(posList_target)

# Find the homography matrix.
homography, mask = cv2.findHomography(posList_s, posList_t, cv2.RANSAC)

height_target, width_target,_  = img2.shape
# Use this matrix to transform the
# colored image wrt the reference image.
transformed_img = cv2.warpPerspective(img1,
                    homography, (width_target, height_target), borderValue =(243,243,243) )


# Save the output.
cv2.imwrite('initTransformed_img1.png', transformed_img)
cv2.imwrite("initTransformed_img2.png", img2)



frac =0
def on_track(value):
    global frac    
    frac = cv2.getTrackbarPos('ratio','output')/255.0

cv2.namedWindow('output')
cv2.createTrackbar('ratio','output',0,255,on_track)
cv2.setTrackbarPos('ratio','output', 128)


while(1):
        blend = cv2.addWeighted(img2, frac, transformed_img, 1-frac, 0.0)
        #print(frac)
        cv2.imshow('output', blend)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
             break
cv2.destroyAllWindows()




'''
import SimpleITK as sitk
import registration_v4 as reg
fixed_path = '181_he_0.25.png' #'initTransformed_img2.png'
moving_path = '181_nuc_0.25.png' #'initTransformed_img1.png'

# fixedIHC, movingHE = load_IHC_HE(fixed_path, moving_path)
fixedArray = cv2.imread(fixed_path, cv2.IMREAD_GRAYSCALE)
movingArray = cv2.imread(moving_path, cv2.IMREAD_GRAYSCALE)

h1,w1=fixedArray.shape
h2,w2=movingArray.shape
w = w1 if w1>=w2 else w2
h = h1 if h1>=h2 else h2
fixedBackColor = fixedArray[0][0]
movingBackColor = fixedBackColor

fixedCanvas= np.full((h,w),fixedBackColor, dtype=np.float32)
movingCanvas= np.full((h,w),movingBackColor, dtype=np.float32)

fixedCanvas[0:h1,0:w1]=fixedArray
movingCanvas[0:h2,0:w2]=movingArray

# fixedImage = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
# movingImage = sitk.ReadImage(moving_path, sitk.sitkFloat32)
fixedImage = sitk.GetImageFromArray(fixedCanvas)
movingImage = sitk.GetImageFromArray(movingCanvas)

tx = reg.initial_transform(fixedImage, movingImage) #RGB 채널로 3번 해야함 추가하기
movingImage = sitk.Resample(movingImage, fixedImage, tx, sitk.sitkLinear, movingImage[0,0], movingImage.GetPixelID()) #default pixel=moving_image[0,0]
cv2.imwrite('initTransformed_img1_transformed.png',sitk.GetArrayFromImage(movingImage))


outGrayImage, outTx= reg.bspline_registration(fixedImage, movingImage)
#outGrayArray, transformParameterMap= reg.non_rigid_registration(fixedImage,movingImage)#movingImage to fixedImage 
cv2.imwrite('initTransformed_img1_deformed.png',sitk.GetArrayFromImage(outGrayImage))


movingArray = cv2.imread(moving_path, cv2.IMREAD_COLOR)
for c in range(movingArray.shape[2]):
    tempImage = sitk.GetImageFromArray(movingArray[:,:,c])
    tempImage = sitk.Resample(tempImage, fixedImage, tx, sitk.sitkLinear, tempImage[0,0], tempImage.GetPixelID()) 
    movingArray[:,:,c]=sitk.GetArrayFromImage(tempImage)
#default pixel=moving_image[0,0]


outArray=[]
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixedImage)
resampler.SetInterpolator(sitk.sitkLinear)
# resampler.SetDefaultPixelValue(100)
resampler.SetTransform(outTx)
for c in range(movingArray.shape[2]):
        out = resampler.Execute(sitk.GetImageFromArray(movingArray[:,:,c]))
        outArray.append(sitk.GetArrayFromImage(out))
deformedArray = np.dstack(outArray)
cv2.imwrite('initTransformed_img1_deformed_color.png',deformedArray)
'''



'''
 # keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("WindowName", img1)
    key = cv2.waitKey(1) & 0xFF
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
        break

# Create ORB detector with 5000 features.
orb_detector = cv2.ORB_create(5000)
  
# Find keypoints and descriptors.
# The first arg is the image, second arg is the mask
#  (which is not required in this case).
kp1, d1 = orb_detector.detectAndCompute(img1, None)
kp2, d2 = orb_detector.detectAndCompute(img2, None)
  
# Match features between the two images.
# We create a Brute Force matcher with 
# Hamming distance as measurement mode.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
  
# Match the two sets of descriptors.
matches = matcher.match(d1, d2)
  
# Sort matches on the basis of their Hamming distance.
matches.sort(key = lambda x: x.distance)
  
# Take the top 90 % matches forward.
matches = matches[:int(len(matches)*0.9)]
no_of_matches = len(matches)
  
# Define empty matrices of shape no_of_matches * 2.
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))
  
for i in range(len(matches)):
  p1[i, :] = kp1[matches[i].queryIdx].pt
  p2[i, :] = kp2[matches[i].trainIdx].pt
  
# Find the homography matrix.
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
  
# Use this matrix to transform the
# colored image wrt the reference image.
transformed_img = cv2.warpPerspective(img1_color,
                    homography, (width, height))
  
# Save the output.
cv2.imwrite('output.jpg', transformed_img)'''