## this script takes hdr image(s) as input and gives saliency prediction
import sys
sys.path.insert(1, 'experiments')
from create_exposure_stack import create_ldrstack_from_hdr

# from experiments.create_exposure_stack import create_ldrstack_from_hdr
import cv2
import os
import shutil
import glob
import imageio as iio
import model_defination
from utils import get_split, preprocess_images_split, postprocess_predictions_split_weighted
from fuse import *

# sys.path.append('experiments')
dataset = 'imlhdr'
fusion_type = 'mink'; ## average, max_out, minkowskii fusion, UW fusion
only_fusion = True
sampling_mode = '5'
split_version = 'sliding'
model_version = '0.1_dense_cam'
spatial_size=224
model_time = '2019-07-05 15:07:46_010024'
model_dir = os.path.join('output/',model_time)
model_name = 'srsn_saliency.h5'
center_bias=True


if dataset == 'hdreye':
  hdr_dir = '/data/SaliencyDataset/Image/HDREYE/images/HDR'
  output_dir = '/data/SaliencyDataset/Image/HDREYE/saliency_map/HDR'
  # output_dir = 'test_imgs'
elif dataset == 'ethyma':
  hdr_dir = '/data/SaliencyDataset/Image/ETHyma/images'
  output_dir = '/data/SaliencyDataset/Image/ETHyma/saliency_map/exposure_stack'
elif dataset == 'imlhdr':
  hdr_dir = '/data/SaliencyDataset/Image/IMLHDR/hdr'
  output_dir = '/data/SaliencyDataset/Image/IMLHDR/saliency_map/exposure_stack'

output_dir = os.path.join(output_dir,model_time+'_split_'+sampling_mode+'_'+fusion_type)
if not os.path.isdir(output_dir):
  os.makedirs(output_dir)
temp_fusion_dir = os.path.abspath(os.path.join('test_imgs/temp_fusion',model_time+'_'+dataset))
center_map = cv2.imread('test_imgs/center.jpg',0)

if os.path.isdir(temp_fusion_dir):
  if not only_fusion:
    shutil.rmtree(temp_fusion_dir)
os.makedirs(temp_fusion_dir, exist_ok=True)

if not only_fusion:
  model = model_defination.get_model(model_version=model_version, spatial_size=spatial_size)
  weight_path = os.path.join(model_dir, model_name)
  model.load_weights(weight_path)
  hdr_path_list = glob.glob(os.path.join(hdr_dir,'*.*'))
  for hdr_path in hdr_path_list:
    # hdr_path = os.path.join(hdr_dir, 'C09.hdr')
    if len(glob.glob(os.path.join(temp_fusion_dir, os.path.basename(hdr_path).split('.')[0]+'*')))!=0:
      print(hdr_path, 'already done.')
      continue
    try:
      img = iio.imread(hdr_path)
    except:
      print(hdr_path, 'failed.')
      continue
    ldr_stack, exposure_list = create_ldrstack_from_hdr(img, sampling_mode=sampling_mode)

    processing_batch = []
    for i in range(len(ldr_stack)):
      # out_path = os.path.join(tmp_dir, out_name)
      ldr = ldr_stack[i]
      ldr = ldr[:,:,::-1] * 255
      
      # print(len(ldr_stack), ldr.shape);exit()
      ori_height, ori_width, _ = ldr.shape

      img_arr = preprocess_images_split(ldr, int(spatial_size),int(spatial_size), split_version=split_version)
      print("shape of imgarr:", img_arr.shape)
      predictions = model.predict(img_arr, batch_size=len(img_arr))

      img_name = os.path.basename(hdr_path).split('.')[0]+'_'+str(i)+'.jpg'
      # img_name = 'C01_'+str(i)+'.jpg'

      output_path = os.path.join(temp_fusion_dir, img_name)
      prediction = postprocess_predictions_split_weighted(predictions, ori_height,ori_width,split_version=split_version)

      if center_bias:
        center = cv2.resize(center_map, dsize=(ori_width, ori_height))
        prediction = prediction * center
        prediction = prediction/prediction.max() * 255
      cv2.imwrite(output_path, prediction)

image_name_list = os.listdir(hdr_dir)
sal_path_list = glob.glob(os.path.join(temp_fusion_dir, '*.*'))
print(temp_fusion_dir, sal_path_list)

if fusion_type=='uw':
  # UW fusion
  var_str = 'image_dir=\'%s\';smap_dir=\'%s\';output_dir=\'%s\';folder=\'%s\';'% (hdr_dir, os.path.dirname(temp_fusion_dir), output_dir, model_time)
  cmd = 'matlab -nodesktop -nosplash -nodisplay -r "addpath(\'fang_uncertainty_weighting\');clc;clear;%s hdr_fusion;exit()"' % var_str
  os.system(cmd)
else:
  for image_name in image_name_list:
      prefix=os.path.splitext(image_name)[0]
      cur_sal_path_list = [path for path in sal_path_list if prefix in path]

      if fusion_type=='avg':
        # raise NotImplementedError
        fusion_img = direct_combine(cur_sal_path_list)
      elif fusion_type=='max':
        fusion_img = max_out(cur_sal_path_list)
      elif fusion_type=='mink':
        fusion_img = minkowski(cur_sal_path_list)  
      else:
        raise NotImplementedError

      #normalization
      fusion_img = fusion_img - fusion_img.min()
      fusion_img = fusion_img / fusion_img.max()
      fusion_img = fusion_img * 255
      output_name = os.path.join(output_dir, prefix+'.png')

      cv2.imwrite(output_name, fusion_img)
      print(output_name, 'saved.')