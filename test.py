## this script takes hdr image(s) as input and gives saliency prediction
import sys
from create_exposure_stack import create_ldrstack_from_hdr
import cv2
import os
import shutil
import glob
import imageio as iio
import model_defination
from utils import get_split, preprocess_images_split, postprocess_predictions_split_weighted
from config import *
from fuse import *

dataset = 'imlhdr'
spatial_size=224

hdr_dir = 'test_imgs'
output_dir = 'output'

os.makedirs(output_dir, exist_ok=True)
temp_fusion_dir = os.path.abspath(os.path.join('_temp_fusion'+os.path.basename(hdr_dir)))
os.makedirs(temp_fusion_dir, exist_ok=True)

# if not only_fusion:
model = model_defination.get_model(model_version=model_version, spatial_size=spatial_size)
weight_path = os.path.join(save_base, model_name)
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
    
    ori_height, ori_width, _ = ldr.shape

    img_arr = preprocess_images_split(ldr, int(spatial_size),int(spatial_size), split_version=split_version)
    print("shape of imgarr:", img_arr.shape)
    predictions = model.predict(img_arr, batch_size=len(img_arr))

    img_name = os.path.basename(hdr_path).split('.')[0]+'_'+str(i)+'.jpg'

    output_path = os.path.join(temp_fusion_dir, img_name)
    prediction = postprocess_predictions_split_weighted(predictions, ori_height,ori_width,split_version=split_version)

    cv2.imwrite(output_path, prediction)

image_name_list = os.listdir(hdr_dir)
sal_path_list = glob.glob(os.path.join(temp_fusion_dir, '*.*'))
print(temp_fusion_dir, sal_path_list)

if fusion_type=='uw':
  # UW fusion
  var_str = 'image_dir=\'%s\';smap_dir=\'%s\';output_dir=\'%s\';'% (hdr_dir, temp_fusion_dir, output_dir)
  # cmd = 'matlab -nodesktop -nosplash -nodisplay -r "addpath(\'uncertainty_weighting\');clc;clear;%s hdr_fusion;exit()"' % var_str
  cmd = 'matlab -nodesktop -nosplash -nodisplay -r "addpath(\'uncertainty_weighting\');clc;clear;%s hdr_fusion;"' % var_str
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

shutil.rmtree(temp_fusion_dir)