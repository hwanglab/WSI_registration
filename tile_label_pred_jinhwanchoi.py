# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import os
import openslide
from math import ceil, floor
from scipy import ndimage as ndi
import pandas as pd
from pathlib import Path
from torch.autograd import Variable

from skimage import transform, draw
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, opening,closing, square

import multiprocessing as mp
import argparse


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# GPU = "cuda:0"
nb_classes = 9
cores = 48

label_name = ["B","BD","F","H","I","M","MF","N","T"]
class_name = {"B":"Blood Cell",
              "BD":"Bile Duct",
              "F":"Fibrosis",
              "H":"Hepatocyte",
              "I":"Inflammation",
              "M":"Mucin",
              "MF":"Macrophage",
              "N":"Necrosis",
              "T":"Tumor"}

SUPPORTED_WSI_FORMATS = ["svs","ndpi","vms","vmu","scn","mrxs","tiff","svslide","tif","bif"]

def predict_wsi(model, tile) :
    except_label = ["B","BD","H","M"]
    loader = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    image = Image.open(tile)
   
    image = image.convert('RGB')
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    image = image.cuda()
    
    # image = torch.from_numpy(tile).to("cuda")
    # image = image.reshape(3,224,224)
    # image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)
    
    model.eval()  
    output = model(image)
    # index = output.data.cpu().numpy().argmax()
    percentage = torch.nn.functional.softmax(output, dim=1)[0] 
    
    # print(percentage)
    index = percentage.argmax()
    # print(percentage[index])
    
    # if label_name[index] in except_label :
    #     return -1
    
    if percentage[index] < 0.5 :
        print("except " + str(index) + ' ' + str(percentage[index]))
        return -1, 0

    # pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    
    # print(index)
    # return np.argmax(pred_probs)
    return index, int(percentage[index]*100)

    '''
    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        ax.set_title("{:.0f}% Alien, {:.0f}% Predator".format(100*pred_probs[i,0],
                                                                100*pred_probs[i,1]))
        ax.imshow(img)
    '''


# -

def prepare_tiles(wsi, mpt=128, get_chunk_id=False):
    """
    Import a WSI, calculate foreground/background, and calculate tile coordinates to output directory.

    Input:
        wsi (str): Path to WSI file to be processed.        
        mpt (int): Desire width and height of processed tiles in microns. Default: [%d].
        get_chunk_id (bool): Wether or not to identify individual tissue chunks in slide (larger than a tile). Default: False

    Output:
        Funtion exports 3 files:
            1. Reference CSV file containing coordinates and tissue ratio for each tile.
            2. RGB thumbnail of the WSI as a PNG file.
            3. Binary tissue mask of the WSI as a PNG file.
        Function returns a tuple with with the following format:
            1. Pandas dataframe containing coordinates and tissue ratio for each tile.
            2. Pixels-per-tile value for the WSI's X axis
            3. Pixels-per-tile value for the WSI's Y axis
    % (MICRONS_PER_TILE)"""

    # # Valiate output path
    # if not os.path.isdir(output):
    #     os.mkdir(output)

    # Calculate desired tile dimensions (pixels per tile)
    ppt_x = int(mpt / float(wsi.properties['openslide.mpp-x']))
    ppt_y = int(mpt / float(wsi.properties['openslide.mpp-y']))

    # Get thumbnail for tissue mask
    thumbnail_og = wsi.get_thumbnail(size=(wsi.level_dimensions[-1][0], wsi.level_dimensions[-1][1]))
    thumbnail = np.array(thumbnail_og)
    thumbnail = (rgb2gray(thumbnail) * 255).astype(np.uint8)

    # calculate mask parameters
    thumbnail_ratio = wsi.dimensions[0] / thumbnail.shape[1]  # wsi dimensions: (x,y); thumbnail dimensions: (rows,cols)
    thumbnail_mpp = float(wsi.properties['openslide.mpp-x']) * thumbnail_ratio
    noise_size_pix = round(256 / thumbnail_mpp)
    noise_size = round(noise_size_pix / thumbnail_ratio)
    thumbnail_ppt_x = ceil(ppt_x / thumbnail_ratio)
    thumbnail_ppt_y = ceil(ppt_y / thumbnail_ratio)
    tile_area = thumbnail_ppt_x*thumbnail_ppt_y

    # Create and clean tissue mask
    tissue_mask = (thumbnail[:, :] < threshold_otsu(thumbnail))
    tissue_mask = closing(tissue_mask, square(5))
    tissue_mask = opening(tissue_mask, square(5))
    tissue_mask = remove_small_objects(tissue_mask, noise_size)
    tissue_mask = ndi.binary_fill_holes(tissue_mask)

    if get_chunk_id:
        # Get labels for all chunks
        chunk_mask = ndi.label(tissue_mask)[0]

        # Filter out chunks smaller than tile size
        (chunk_label, chunk_size) = np.unique(chunk_mask,return_counts=True)
        filtered_chunks = chunk_label[ chunk_size < tile_area ]
        for l in filtered_chunks:
            chunk_mask[chunk_mask == l] = 0

    # Calculate margin according to ppt sizes
    wsi_x_tile_excess = wsi.dimensions[0] % ppt_x
    wsi_y_tile_excess = wsi.dimensions[1] % ppt_y

    # Determine WSI tile coordinates
    wsi_tiles_x = list(range(ceil(wsi_x_tile_excess / 2), wsi.dimensions[0] - floor(wsi_x_tile_excess / 2), ppt_x))
    wsi_tiles_y = list(range(ceil(wsi_y_tile_excess / 2), wsi.dimensions[1] - floor(wsi_y_tile_excess / 2), ppt_y))

    # Approximate mask tile coordinates
    mask_tiles_x = [floor(i / thumbnail_ratio) for i in wsi_tiles_x]
    mask_tiles_y = [floor(i / thumbnail_ratio) for i in wsi_tiles_y]

    # Populate tile reference table
    rowlist = []
    for x in range(len(wsi_tiles_x)):
        for y in range(len(wsi_tiles_y)):
            # Get np.array subset of image (a tile)
            aTile = tissue_mask[mask_tiles_y[y]:mask_tiles_y[y] + thumbnail_ppt_y,
                    mask_tiles_x[x]:mask_tiles_x[x] + thumbnail_ppt_x]
            
            # Determine chunk id by most prevalent ID
            if get_chunk_id:
                chunk_tile = chunk_mask[mask_tiles_y[y]:mask_tiles_y[y] + thumbnail_ppt_y,
                    mask_tiles_x[x]:mask_tiles_x[x] + thumbnail_ppt_x]
                chunk_id = np.bincount(chunk_tile.flatten()).argmax()
                

            # Calculate tissue ratio for tile
            tissue_ratio = np.sum(aTile) / aTile.size

            slide_id = len(rowlist) + 1

            new_row = {"tile_id": slide_id,
                       "index_x": x,
                       "index_y": y,
                       "wsi_x": wsi_tiles_x[x],
                       "wsi_y": wsi_tiles_y[y],
                       "mask_x": mask_tiles_x[x],
                       "mask_y": mask_tiles_y[y],
                       "filename": "__tile-n-%d_x-%d_y-%d.png" % (slide_id, x, y),
                       "tissue_ratio": tissue_ratio
                       }

            if get_chunk_id:
                new_row['chunk_id'] = chunk_id

            rowlist.append(new_row)

    # Create reference dataframe
    colnames = ["tile_id", "index_x", "index_y", "wsi_x", "wsi_y", "mask_x", "mask_y", "filename", "tissue_ratio"]
    if get_chunk_id:
                colnames.append('chunk_id')
    # print(str(len(rowlist)))
    ref_df = pd.DataFrame(data=rowlist, columns=colnames)
    
    # Remove filenames for empty tiles
    ref_df.loc[ref_df['tissue_ratio'] == 0, "filename"] = None

    # print(str(len(ref_df)))
    # print(sum(ref_df['tissue_ratio'] == 0))
    # print(sum(ref_df['tissue_ratio'] > 0))
    
    '''
    output = Path(output) 

    
    # Export Mask image
    filename_tissuemask = os.path.basename(output) + "__tissue-mask_tilesize_x-%d-y-%d.png" % (
    thumbnail_ppt_x, thumbnail_ppt_y)
    plt.figure()
    plt.imshow(tissue_mask, cmap='Greys_r', interpolation='nearest')
    plt.axis('off')
    plt.margins(0, 0)
    plt.savefig(output / filename_tissuemask, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Export Thumbnail image
    filename_thumbnail = os.path.basename(output) + "__thumbnail_tilesize_x-%d-y-%d.png" % (
        thumbnail_ppt_x, thumbnail_ppt_y)
    plt.figure()
    plt.imshow(thumbnail_og)
    plt.axis('off')
    plt.margins(0, 0)
    plt.savefig(output / filename_thumbnail, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Export CSV file
    filename_refdf = os.path.basename(output) + "__reference_wsi-tilesize_x-%d-y-%d_mask-tilesize_x-%d-y-%d.tsv" % \
                     (ppt_x, ppt_y,thumbnail_ppt_x, thumbnail_ppt_y)
    ref_df.to_csv(output / filename_refdf, sep="\t", line_terminator="\n", index=False)
    '''
    
    #tile_data_lists = np.array_split(ref_df.loc[ref_df['tissue_ratio'] > 0 ], cores)

    return (ref_df, ppt_x, ppt_y)

# +
color = [[0,0,1], [0,1,0], [1,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], [0,0,0], [0.3,0.3,0.4], [0.2,0.5,0.3]] 

def export_tiles(model, wsi_image, tile_data_lists, output_path, tile_dims, normalizer=None, final_tile_size=0):
    """
    Import a WSI, split in to tiles, normalize color if requested, and save individual tile files to output directory.

    Input:
        wsi (str): Path to WSI file to be processed.
        tile_data (Pandas dataframe): Details for tiles to be extracted, normalized and exported.
        tile_dims (dict): Python dictionary containing size of tiles in WSI. Fromat: {'x':<x-tiles>,'y':<y-tiles>}
        output (str): Path to output directory for processed tiles. Default: [./]
        normalizer (normalizer): Normalizer object that has been initialized and is ready to use. Default: [None].
        final_tile_size (int): Desire pixel width and height of final tiles. Use zero (0) for NO resizing. Default: [0].

    Output:
        Funtion exports tiles as PNG files to output directory.
    """

    labeld_list = []
    
    # Process and export each tile sequentially
    # print(len(tile_data_lists))
    
    offset_x = 0 
    offset_y = 0
    
    if 'openslide.bounds-x' in wsi_image.properties.keys() :
        offset_x = int(wsi_image.properties['openslide.bounds-x'])
    if 'openslide.bounds-y' in wsi_image.properties.keys() :
        offset_x = int(wsi_image.properties['openslide.bounds-y'])
    
    
    
#     print(offset_x)
#     print(offset_y)
        
    # pool = mp.Pool(48)
    cnt = 0
    for tile_data in tile_data_lists :
        cnt += 1
        print("list #" + str(cnt))
        for index, aTile in tile_data.iterrows():
            # Extract tile region
             
            aTile_img = wsi_image.read_region((aTile["wsi_x"], aTile["wsi_y"]), level=0,
                                    size=(tile_dims['x'], tile_dims['y']))

            #Convert to RGB array
            aTile_img = np.array( aTile_img.convert('RGB') )

            # Normalize if required
            if normalizer is not None:
                aTile_img = normalizer.transform(aTile_img)

            # Resize tile to final size
            if final_tile_size != 0:
                aTile_img = transform.resize(aTile_img, (final_tile_size,final_tile_size,3), order=1)  # 0:nearest neighbor

            png_file_path = './temp_image/' + output_path.split('.')[0]+'.png'
            plt.imsave(png_file_path, aTile_img)            
            # pred = predict_wsi(model, aTile_img)
            
            pred, probability = predict_wsi(model, png_file_path)
            
            if pred > -1 :                            
                labeld_list.append([int(aTile["wsi_x"]-offset_y), int(aTile["wsi_y"]-offset_x), int((aTile["wsi_x"]+tile_dims['x'])-offset_y), int((aTile["wsi_y"]+tile_dims['y'])-offset_x), pred, probability])                
        
 
    # pool.close()
    # pool.join()
    # pool.terminate()
        
    return labeld_list

def annotation_json(label_list, output_path) :
    import json
    
    label_color = [[255,0,0],[0,128,0],[128,0,0],[128,128,0],[0,255,0],[0,0,255],[0,255,255],[255,0,0],[255,255,0]]

    annotataion_json = {}
    annotataion_json["type"]="FeatureCollection"
    annotataion_json["features"]=[]
    
    for label in label_list :
        x1,y1,x2,y2=label[0],label[1],label[2],label[3]
                
        feature = {"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[x1,y1],[x2,y1],[x2,y2],[x1,y2],[x1,y1]]]},"properties":{"object_type":"annotation","name":label_name[label[4]]+str(label[5]),"classification":{"name":class_name[label_name[label[4]]],"color":label_color[label[4]],"isLocked":"true"}}}
                                                                                                     
        annotataion_json["features"].append(feature)
    
    
    with open(output_path, 'w') as outfile :
        json.dump(annotataion_json, outfile)
        
def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''

    parser = argparse.ArgumentParser()

    parser.add_argument(
          '-i','--input',
          required=True,
          type=str,          
          help='path of input wsi data')
    parser.add_argument(
          '-o','--output',
          required=True,
          type=str,          
          help='path for output')
    parser.add_argument(
          '-g','--gpu',          
          nargs="?",
          type=int,
          default=0,
          const='-1',
          help='Input GPU Number you want to use.')
    parser.add_argument(
          '-s','--size',          
          nargs="?",
          type=int,
          default=256,
          const='-1',
          help='tile size (um)')
                           
    args = parser.parse_args()
    return args

# +
if __name__ == '__main__':    
    args = get_args()    
    
    if args.gpu > -1 :
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # The GPU id to use, usually either "0" or "1";        
    
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(pretrained=False).to(device)
    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, nb_classes)).to(device)
    model.load_state_dict(torch.load('./resnet_9c_4p_128um_100epochs_e24.h5'))

    # wsi = '/home/ext_choi_jinhwan_mayo_edu/bucket_data/organized_datasets/colon/Yonsei-1/MSI-H/1M21.mrxs'
    
    # paths = ['~/bucket_data/organized_datasets/colon/Yonsei-remade/MSI-H/','~/bucket_data/organized_datasets/colon/Yonsei-remade/MSS/','~/bucket_data/datasets/Pathology_Slides/Colon_St_Mary_Hospital_SungHak_Lee_Whole_Slide_Image/CRC_St._Mary_hospital/','~/bucket_data/datasets/PAIP_Colon/','~/bucket_data/datasets/KI_ColonLiverMets_InvasionFrontAnnotationis/']
    
    paths = (args.input).split(',')
    
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
        
    for path in paths :                
        if os.path.isdir(Path(path)):
            print(path)
            file_list = os.listdir(path)
            for each_file in file_list :                
                for wsi_format in SUPPORTED_WSI_FORMATS :
                    if wsi_format == each_file.split('.')[-1] :
                        wsi = path + '/' + each_file
                        print(each_file)
                        json_path = args.output + '/' + each_file.split('.')[0] + '.json'
                        if os.path.exists(json_path) :
                            print('This slide is already done!!')
                            continue
                            
                        try :
                            wsi_image = openslide.open_slide(str(wsi))
                        except :
                            print('openslide error!')
                            continue

                        (ref_df, ppt_x, ppt_y) = prepare_tiles(wsi_image, mpt=args.size)

                        tile_data_lists = np.array_split(ref_df.loc[ref_df['tissue_ratio'] > 0 ], cores)
                        tile_dims = {'x':ppt_x,'y':ppt_y}

                        # output_file = args.output + '/' + each_file.split('.')[0] + '.json'
                        # print(output_file)
                        labeld_list = export_tiles(model, tile_data_lists=tile_data_lists, output_path=each_file, wsi_image=wsi_image, tile_dims=tile_dims, final_tile_size=224)
                        print(str(wsi_image.dimensions))
                                                
                        annotation_json(labeld_list, json_path)
                        
                        wsi_image.close()
                        continue
    
            
        else:
            if args.input.endswith(tuple(SUPPORTED_WSI_FORMATS)):
                all_wsi_paths.append(Path(path))

       
    
    
 