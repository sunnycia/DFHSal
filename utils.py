import scipy
import cv2
import numpy as np

def get_split(img, split_version='1+4'):
  height, width, _ = img.shape
  # print(img.shape)
  short_edge = (width if width < height else height)
  begin = int(height+width-(short_edge*2))
  if height > width:
    begin = int((height-width)/2)
    coarse_center_img = img[begin:begin+short_edge, ...]
  else:
    begin = int((width-height)/2)
    coarse_center_img = img[..., begin:begin+short_edge,0:3]
  half = int(short_edge/2)
  quarter = int(half/2)

  top_left = coarse_center_img[0:half, 0:half]
  top_right = coarse_center_img[0:half, half:half*2]
  bot_left = coarse_center_img[half:half*2, 0:half]
  bot_right = coarse_center_img[half:half*2, half:half*2]
    
  center = coarse_center_img[quarter:quarter+half, quarter:quarter+half]

  if height > width:
    mid_left = img[0:half, quarter:quarter+half]
    mid_right = img[-half:, quarter:quarter+half]
  else:
    mid_left = img[quarter:quarter+half, 0:half]
    mid_right = img[quarter:quarter+half, -half:]
  # print(mid_right.shape);exit()
  if split_version=='1+4':
    return [coarse_center_img, top_left,top_right, bot_left,bot_right]
  elif split_version=='1+5':
    return [coarse_center_img, center, top_left,top_right, bot_left,bot_right]
  elif split_version=='1+7':
    return [coarse_center_img, mid_left, center, mid_right, top_left, top_right, bot_left, bot_right]
  else:
    raise NotImplementedError

def sliding_window(image, stepSize, windowSize):
  # thanks to 
  # https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
  # slide a window across the image
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):
      # yield the current window
      yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def get_sliding_split(img):
  height, width, _ = img.shape
  # print(img.shape)
  if width > height:
    short_edge = height
    long_edge = width
  else:
    short_edge = width
    long_edge = height

  half = int(short_edge/2)
  quarter = int(half/2)
  residual = (long_edge%half)/2
  step = quarter

  img_list = []
  for (x, y, window) in sliding_window(img, stepSize=step, windowSize=(half, half)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != half or window.shape[1] != half:
      continue
    img_list.append(window)
  return img_list

def preprocess_images_split(img, shape_r=224, shape_c=224, split_version='sliding'):
    if split_version == 'sliding':
      img_list = get_sliding_split(img)
      total = len(img_list)+1
    
    else:
      total_list = split_version.split('+')
      total=1
      for string in total_list:
        total += int(string)
      img_list = get_split(img, split_version)
      
    ims = np.zeros((total, shape_r, shape_c, 3))

    # padded_image = padding(img, shape_r, shape_c, 3)
    ims[0] = cv2.resize(img,dsize=(shape_r,shape_c))

    for i in range(len(img_list)):
      ims[i+1] = cv2.resize(img_list[i], dsize=(shape_r,shape_c))

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68

    return ims

def postprocess_predictions_split_weighted(preds, shape_r, shape_c, split_version='sliding', save_detail=False):
    if split_version=='1+4':
      raise NotImplementedError
    elif split_version=='1+5':
      raise NotImplementedError
    elif split_version=='1+7':
      raise NotImplementedError
    if split_version=='sliding':
      prediction_list=preds
      full_map = np.zeros((shape_r, shape_c))

      if shape_c > shape_r:
        short_edge = shape_r
        long_edge = shape_c
      else:
        short_edge = shape_c
        long_edge = shape_r

      half = int(short_edge/2)
      quarter = int(half/2)
      residual = (long_edge%half)/2
      step = quarter

      coarse_prediction = prediction_list[0]
      index = 1
      for (x, y, window) in sliding_window(full_map, stepSize=step, windowSize=(half, half)):
        x = int(x);y = int(y)
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != half or window.shape[1] != half:
          continue
        prediction = cv2.resize(prediction_list[index, ..., 0], dsize=(half,half))
        output = np.maximum(prediction, window)
        full_map[y:y+half, x:x+half] = output
        index+=1

    if not save_detail:
      full_map = full_map * cv2.resize(coarse_prediction,(shape_c, shape_r))
    print("full_map sum:",np.sum(full_map))
    full_map = full_map / np.max(full_map) * 255
    img = scipy.ndimage.filters.gaussian_filter(full_map, sigma=7)
    img = img / np.max(img) * 255
    return img