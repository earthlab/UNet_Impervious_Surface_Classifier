import rasterio
import glob
import os,sys
from skimage.transform import rescale, resize, downscale_local_mean
from matplotlib import pyplot as plt
import numpy as np
from scipy import misc
import fiona
# import geopandas as gpd
from shapely.geometry import shape
import shapely
from rasterio.mask import mask
from pyproj import Proj, transform

import torch, torchvision
import rasterio
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
from math import log10
from collections import OrderedDict

def load_pretrained_weights(model, weight_path):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    checkpoint = torch.load(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    if len(matched_layers) == 0:
        print('** ERROR: the pretrained weights "{}" cannot be loaded, please check the key names manually (ignored and continue)'.format(weight_path))
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print("* The following layers are discarded due to unmatched keys or layer size: {}".format(discarded_layers))

# convert bounding boxes into string required for DG CatalogImage
# ex. bbox = [-110.85299491882326,32.167148499672855,-110.84870338439943,32.170236308395644] WGS84
def rioBoundBoxUTM_toWGS84(bounds_obj, src_crs, wgs84='4326'):
    
    wgs = Proj(init='epsg:{}'.format(wgs84))
    p2 = Proj(init=str(src_crs['init']))
    
    try:
        min_x, min_y = transform(p2, wgs, bounds_obj.left, bounds_obj.bottom)
        max_x, max_y = transform(p2, wgs, bounds_obj.right, bounds_obj.top)
    except:
        min_x, min_y = transform(p2, wgs, bounds_obj[0], bounds_obj[1])
        max_x, max_y = transform(p2, wgs, bounds_obj[2], bounds_obj[3])
    
    return min_x, min_y, max_x, max_y

def rioBoundBoxWGS84_toUTM(bounds_obj, src_crs, wgs84='4326'):
    
    #print(src_crs['init'])
    wgs = Proj(init='epsg:{}'.format(wgs84))
    #p2 = Proj(init=str(src_crs['init']))
    p2 = Proj(init='epsg:{}'.format(str(src_crs['init'])))
    min_x, min_y = transform(wgs, p2, bounds_obj[0], bounds_obj[1])
    max_x, max_y = transform(wgs, p2, bounds_obj[2], bounds_obj[3])
    
    return min_x, min_y, max_x, max_y

# define a function to check the image bounds of the two datasets
def imageIntersectionTest(dg_bounds, planet_bounds):
    
    res = ''
    if dg_bounds[0] > planet_bounds[0]:
        res += ' DG xmin is gt PL xmin'
        
    if dg_bounds[1] > planet_bounds[1]:
        res += ' DG ymin is  gt PL ymin'
        
    if dg_bounds[2] < planet_bounds[2]:
        res += ' DG xmax is lt PL xmax'
        
    if dg_bounds[3] < planet_bounds[3]:
        res += ' DG ymax is lt PL ymax'
        
    return res

## define a function to get the chip dimensions from a larger bounding box and a chip dimension
def generateChipBoxesUTM(bbox, box_dim):
    
#     xmin, ymin, xmax, ymax = bbox.bounds
#     xmin_chips = np.arange(xmin, xmax - box_dim, box_dim)
#     xmax_chips = np.arange(xmin + box_dim, xmax, box_dim)
#     ymin_chips = np.arange(ymin, ymax - box_dim, box_dim)
#     ymax_chips = np.arange(ymin + box_dim, ymax, box_dim)
    
    
    # try a for loop
    for_loop_result = []
    for l_x in np.arange(xmin, xmax-box_dim, box_dim):
        for l_y in np.arange(ymin, ymax-box_dim, box_dim):
            res = [l_x, l_y, l_x + box_dim, l_y + box_dim ]
            for_loop_result.append(res)
    
    return  for_loop_result

def generateChipBoxesUTM_WGS84(bbox, box_dim, crs):
    
    # get bounds
    xmin, ymin, xmax, ymax = bbox.bounds
    print(xmax-xmin)
    print(ymax-ymin)
    
    # try a for loop
    utm_chips = []
    wgs84_chips = []
    
    # construct chip bounds
    for l_x in np.arange(xmin, xmax-box_dim, box_dim):
        for l_y in np.arange(ymin, ymax-box_dim, box_dim):
            
            res = [l_x, l_y, l_x + box_dim, l_y + box_dim ]
            utm_chips.append(res)
            arg = box(res[0], res[1], res[2], res[3]).bounds
            wgs84_chips.append( rioBoundBoxUTM_toWGS84(arg, crs))
            
    return utm_chips, wgs84_chips

def chip_planet_image(impath, bbox, out_file):
    
    first = (bbox[0], bbox[1])    # xmin, ymin
    second = (bbox[0], bbox[3])   # xmin, ymax
    third = (bbox[2], bbox[3])    # xmax, ymax
    fourth = (bbox[2], bbox[1])   # xmax, ymin
    
    ## construct the geometry for rasterio.mask.mask
    bbox_geom = shapely.geometry.Polygon([first, second, third, fourth, first])
    geom = [shapely.geometry.mapping(bbox_geom)]
    
    # read the image
    with rasterio.open(impath, 'r') as src:
        out_image, out_transform = mask(src, geom, crop=True)
        out_meta = src.meta.copy()
        
    # get some pixel metrics
    num_nonzero = np.count_nonzero(out_image[0,:,:])
    num_pixels = out_image[0,:,:].size
    ratio = float(num_nonzero) / float(num_pixels)
    
    ## skip this one if not enough pixels
    if ratio < 0.95:
        return False
    
    else: #continue with the writing
    
        # update the metadata
        out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

        # write the file
        with rasterio.open(out_file, "w", **out_meta) as dest:
            dest.write(out_image)

        return True

def chip_dg_image(dg_scene, bbox, dg_out_file):
    
    print('bbox {}'.format(bbox))
    
    # get the aoi
    img_aoi = dg_scene.aoi(bbox=bbox)
    
    # save the file
#     print(dg_out_file)
    img_aoi.geotiff(path=dg_out_file)
    
    return

def assignRC(rio_obj, samp_pt, window_size=64, inproj='epsg:4326', outproj='epsg:32613'):
    # project the point to source crs
    outProj = Proj(init=outproj)
    inProj = Proj(init=inproj)
    x1,y1 = samp_pt
    x2,y2 = transform(inProj,outProj,x1,y1)
    
    # get the row column
    temp = rio_obj.index(x2,y2)
    r,c = [int(c) for c in temp]
    
    return (samp_pt, r,c)

def verifyWindow(rio_obj, samp_pt, window_size=64, inproj='epsg:4326', outproj='epsg:32613'):
    # project the point to source crs
    outProj = Proj(init=outproj)
    inProj = Proj(init=inproj)
    x1,y1 = samp_pt
    x2,y2 = transform(inProj,outProj,x1,y1)
    
    # get the row column
    temp = rio_obj.index(x2,y2)
    r,c = [int(c) for c in temp]
    
    # check what the sample window would look like
    r_start = int(r - window_size/2)
    r_end = int(r_start + window_size)
    c_start = int(c - window_size/2)
    c_end = int(c_start + window_size)
    
    
    try:
        arr = rio_obj.read()
        win_arr = arr[:,r_start:r_end, c_start:c_end]
        
        
        # if it is all zero or all NaN and at least 95% data
        test_arr = win_arr[0]
        test_nan = np.isnan(np.mean(test_arr))
        test_zero = np.mean(test_arr) == 0
        
        
#         print('here')
#         plt.imshow(test_arr)
#         plt.colorbar()
#         plt.show()
        
         # get some pixel metrics
        num_nonzero = test_arr.size
        num_pixels = test_arr[test_arr == 3].size
        ratio = float(num_pixels) / float(num_nonzero)
#         print(ratio)
        
        # empirically choose no-data ratio of 10% of window
        if (test_nan or test_zero or (ratio > 0.10)) :
            # if either of the tests fail, we don't want that window
            
            pass
        else:
            return [samp_pt, r,c]
        
    except Exception as e:
        print(e)
        # should only happen if start/end coordinates are outside of image bounds. 
        # in that case, we don't want it
        pass
    
## calculate the lon/lat of the random coordinates 
def calcXYfromRC(aff, coords):
    col = coords[1]
    row = coords[0]
    
    # get the origin (why is DG storing Affine transform this way???)
    ox = aff[2]
    oy = aff[5]
    
    # calculate lon / lat
    cx = ox + aff[0]*col
    cy = oy + aff[4]*row
    
    return (cx,cy)


# lonlat_MS = [calcXYfromRC(img_2m.affine, pair) for pair in coords]
# lonlat_PAN = [calcXYfromRC(image_05m.affine, pair) for pair in coords_pan]
    
def checkWindow(rio_obj, samp_pt, window_size=64, inproj='epsg:4326', outproj='epsg:32613'):
    
    try:
        # project the point to source crs
        outProj = Proj(init=outproj)
        inProj = Proj(init=inproj)
        x1,y1 = samp_pt
        x2,y2 = transform(inProj,outProj,x1,y1)

        # get the row column
        temp = rio_obj.index(x2,y2)
        temp = rio_obj.index(x1,y1) # not sure why this needs to happen...????? the above line worked before.
        r,c = [int(c) for c in temp]

        # check what the sample window would look like
        r_start = int(r - window_size/2)
        r_end = int(r_start + window_size)
        c_start = int(c - window_size/2)
        c_end = int(c_start + window_size)


        arr = rio_obj.read()
        win_arr = arr[:,r_start:r_end, c_start:c_end]


        # if it is all zero or all NaN and at least 95% data
        test_arr = win_arr[0]
        test_nan = np.isnan(np.mean(test_arr))
        test_zero = np.mean(test_arr) == 0

         # get some pixel metrics
        num_nonzero = test_arr.size
        num_pixels = test_arr[test_arr == 3].size
        ratio = float(num_pixels) / float(num_nonzero)
    #         print(ratio)

        # empirically choose no-data ratio of 10% of window
        if (test_nan or test_zero or (ratio > 0.10)) :
            # if either of the tests fail, we don't want that window

            return True
        else:
            return False
    
    except Exception as e:
        #print(e)
        return True
    
    
class gtDatasetSampler2(Dataset):
    """DG Dataset"""
    def __init__(self, gtfile, coord_pair, window_size=64, transform=None):
        """ 
        Args:
            image_dir(string): the folder containing the DG images
            transform (callable, optional): Optional transform to  be applies
        """
        self.image_file = gtfile
        self.transform = transform
        self.coords = coord_pair
        self.window_size = window_size
        
    
    def __getitem__(self, idx):
        
        with rasterio.open(self.image_file, 'r') as src:
            temp = src.read()
        
        # get the window
        r,c = self.coords[idx]
        r_start = int(r - self.window_size/2)
        r_end = int(r_start + self.window_size)
        c_start = int(c - self.window_size/2)
        c_end = int(c_start + self.window_size)
        
        # extract the window
        img_arr = temp[0,r_start:r_end, c_start:c_end]
        img_arr = np.expand_dims(img_arr, axis=0)

        # set no data to 0
        img_arr[img_arr == 3] = 0
        
        # convert to tensor
        img_arr = torch.from_numpy(img_arr).float()
        
        return img_arr
    
    def __len__(self):
        return len(self.coords)


class DigitalGlobeSampler(Dataset):
    """DG Dataset"""
    def __init__(self, cat_img, coord_pair, window_size=64, transform=None, comb='bgr'):
        """ 
        Args:
            image_dir(string): the folder containing the DG images
            transform (callable, optional): Optional transform to  be applies
        """
        self.image = cat_img
        self.transform = transform
        self.coords = coord_pair
        self.window_size = window_size
        self.bgrn = [1,2,4,6]
        self.bgr = [1,2,4]
        
    
    def __getitem__(self, idx):
        
        # get the window
        r,c = self.coords[idx]
        r_start = int(r - self.window_size/2)
        r_end = int(r_start + self.window_size)
        c_start = int(c - self.window_size/2)
        c_end = int(c_start + self.window_size)
        
        # extract the window
        img_arr = self.image[:, r_start:r_end, c_start:c_end].compute()

        
        if self.transform:
            img_arr = self.transform(img_arr)
        
        return img_arr
    
    def __len__(self):
        return len(self.coords)
    
class DigitalGlobeSamplerTensor(Dataset):
    """DG Dataset"""
    def __init__(self, cat_img, coord_pair, window_size=64, transform=None, comb='bgr'):
        """ 
        Args:
            image_dir(string): the folder containing the DG images
            transform (callable, optional): Optional transform to  be applies
        """
        self.image = cat_img
        self.transform = transform
        self.coords = coord_pair
        self.window_size = window_size
        self.bgrn = [1,2,4,6]
        self.bgr = [1,2,4]
        self.comb = comb
        
    
    def __getitem__(self, idx):
        
        # get the window
        r,c = self.coords[idx]
        r_start = int(r - self.window_size/2)
        r_end = int(r_start + self.window_size)
        c_start = int(c - self.window_size/2)
        c_end = int(c_start + self.window_size)
        
        # extract the window
        img_arr = self.image[:, r_start:r_end, c_start:c_end].compute()
        
        if self.comb=='bgr' and self.transform and self.transform.transforms[0].mean.__len__()==3:
            
            #img_arr = np.rollaxis(img_arr, 0,3)
            #img_arr = img_arr[:,:,self.bgr]
            
            img_arr = img_arr[self.bgr,:,:]
            
            
            img_arr = self.transform(torch.from_numpy(img_arr))
            return img_arr
        
        elif self.comb=='bgrn' and self.transform and self.transform.transforms[0].mean.__len__()==4:
            
            #img_arr = np.rollaxis(img_arr, 0,3)
            #img_arr = img_arr[:,:,self.bgrn]
            
            img_arr = img_arr[self.bgrn,:,:]
            
            img_arr = self.transform(torch.from_numpy(img_arr))
            return img_arr
        
        else:
#             print('here')
            return self.transform(torch.from_numpy(img_arr))
    
    def __len__(self):
        return len(self.coords)
    


    

        