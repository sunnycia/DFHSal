## copy from training_caffe
import cv2
import os
import numpy as np


def direct_combine(sal_path_list):
    blank_img = np.zeros(cv2.imread(sal_path_list[0]).shape, dtype=np.float32)
    # print blank_img
    for sal_path in sal_path_list:
        blank_img += cv2.imread(sal_path).astype(np.float32)

    blank_img = blank_img/len(sal_path_list)
    return blank_img

def max_out(sal_path_list):
    blank_img = np.zeros(cv2.imread(sal_path_list[0],0).shape, dtype=np.float32)
    for sal_path in sal_path_list:
        blank_img = np.maximum(cv2.imread(sal_path,0).astype(np.float32), blank_img)

    return blank_img

def sum_product(sal_path_list):
    sum_img = np.zeros(cv2.imread(sal_path_list[0],0).shape, dtype=np.float32)
    product_img = np.ones(cv2.imread(sal_path_list[0],0).shape, dtype=np.float32)
    for sal_path in sal_path_list:
        sal_map = cv2.imread(sal_path, 0).astype(np.float32)
        sum_img = sum_img + sal_map
        product_img = np.multiply(product_img, np.add(sal_map,1))

    sum_img = sum_img/len(sal_path_list)
    final_map = np.add(sum_img, product_img)
    
    ## normalization
    final_map = final_map-final_map.min()
    final_map = final_map/final_map.max()
    final_map = final_map*255
    return final_map

def minkowski(sal_path_list):
    m = 3.5
    sum_img = np.zeros(cv2.imread(sal_path_list[0],0).shape, dtype=np.float32)
    
    for sal_path in sal_path_list:
        sal_map = cv2.imread(sal_path, 0).astype(np.float32)
        sum_img = sum_img + np.power(sal_map,m)
    sum_img = np.power(sum_img, 1/m)

    sum_img = sum_img-sum_img.min()
    sum_img = sum_img/sum_img.max()
    sum_img = sum_img*255
    return sum_img

def mysort(img_path):
    index = int(os.path.splitext(os.path.basename(img_path))[0].split('_')[-1])
    return index

def global_contrast_weighted_combine(prefix, sal_path_list, exposion_img_path_list):
    exposion_img_path_list = exposion_img_path_list[:len(sal_path_list)]
    assert(len(sal_path_list)==len(exposion_img_path_list))
    sal_path_list.sort(key=mysort)
    exposion_img_path_list.sort(key=mysort)

    ## estimate contrast
    contrast_list = []
    for exposion_img_path in exposion_img_path_list:
        print(exposion_img_path)
        luminance_map = lum(cv2.imread(exposion_img_path)[:, :, ::-1])
        contrast_list.append(np.std(luminance_map)/np.mean(luminance_map))

    ## weighted fusion
    blank_img = np.zeros(cv2.imread(sal_path_list[0], 0).shape, dtype=np.float32)
    for (sal_path,weight) in zip(sal_path_list,contrast_list):
        sal = cv2.imread(sal_path, 0)
        blank_img += weight * sal

    return blank_img
