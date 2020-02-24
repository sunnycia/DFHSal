import glob
import os
import cv2
import imageio as iio
import numpy as np
import math

def lum(rgb,rgb_coefficients=[0.2126,0.7152,0.0722]):
    ''' a python reinplementation of lum.m in HDR_toolbox'''
    if len(rgb.shape)==2:
        return rgb
    elif len(rgb.shape)==3:
        if rgb.shape[-1]==1:
            return rgb
        elif rgb.shape[-1]==3:
            lum = rgb_coefficients[0]*rgb[:, :, 0]+rgb_coefficients[1]*rgb[:, :, 1]+rgb_coefficients[2]*rgb[:, :, 2]
            return lum
        else:
            return -1
    else:
        return -1

def tonemapping(hdr, tmo_func='reinhard', gamma=2.2, fstop=0):
    ## tone mapping hdr
    if tmo_func=='reinhard':
        tmo = cv2.createTonemapReinhard(gamma=gamma)
    elif tmo_func =='durand':
        tmo = cv2.createTonemapDurand(gamma=gamma)
    elif tmo_func =='drago':
        tmo = cv2.createTonemapDrago(gamma=gamma)
    elif tmo_func =='mantiuk':
        tmo = cv2.createTonemapMantiuk(gamma=gamma)
    elif tmo_func =='linear':
        output = hdr - hdr.min()
        output = output/output.max()
        # return output
        return tonemapping(output,tmo_func='gamma')
    elif tmo_func =='gamma':
        inv_gamma=1.0/gamma
        exposure=np.power(2., fstop)
        output = clamp_img(np.power(exposure*hdr, inv_gamma), 0,1)
        return output
    else:
        raise NotImplementedError
    # elif tmo_func =='cut_high':
    #     output = hdr - hdr.min()
    #     output = output/output.max()
    #     return output
    output = tmo.process(hdr.astype('float32'))
    return output

def clamp_img(img, floor, ceil):
    ''' a python reinplementation of ClampImg.m in HDR_toolbox'''
    # print(img.shape, sum(~np.isnan(img)));

    img[np.where(img>ceil)]=ceil
    img[np.where(img<floor)]=floor
    return img

def apply_crf(img, lin_type='gamma', lin_fun=2.2):
    ''' reinplementation of ApplyCRF.m in HDR_toolbox'''
    if lin_type=='poly':
        raise NotImplementedError

    if lin_type =='sRGB':
        raise NotImplementedError

    if lin_type =='LUT':
        raise NotImplementedError

    if lin_type =='gamma':
        img_out = tonemapping(img, tmo_func='gamma', gamma=lin_fun)
        return img_out
    pass

def check13_color(img):
    channel = img.shape
    # print img.shape
    if len(img.shape)==2 or(len(img.shape)==3 and img.shape[-1]==3):
        pass
    else:
        raise ValueError('The image has to be an RGB or luminance image.')

def max_quart(matrix, percentile):
    total = np.size(matrix)
    # print 'total:',total
    matrix = matrix.flatten()
    matrix.sort()
    percentile_index = int(math.floor(total*percentile))
    # print percentile_index
    return matrix[percentile_index]

def histogram_HDR(img, n_zone=256, type_log='log10', b_normalized=0, b_plot=0):
    check13_color(img)
    L = lum(img);
    L = L.flatten()

    L2 = np.copy(L)

    delta=1e-6
    if type_log=='log2':
        L=np.log2(L+delta)
    if type_log=='loge':
        L=np.log(L+delta)
    if type_log=='log10':
        L=np.log10(L+delta)

    L_min = L.min()
    L_max = L.max()

    dMM = (L_max-L_min)/(n_zone-1)

    histo = np.zeros((n_zone,1))
    bound = (L_min, L_max)

    haverage = 0
    total = 0
    for i in range(1, n_zone):
        indx = np.where(np.logical_and(L>(dMM*(i-1)+L_min), L<(dMM*i+L_min)))
        count = np.size(indx)
        # print count
        if count > 0:
            histo[i] = count
            # print L2[indx], np.size(indx)
            haverage =haverage+max_quart(L2[indx],0.5)*count
            total    = total+count;

    if (b_normalized):
        norm = sum(histo)
        if (norm>0):
            histo = histo/norm

    haverage = haverage / (total)

    return histo, bound, haverage


def exposure_histogram_sampling(img, n_bit=8, eh_overlap=2.0):
    if n_bit<1:
        n_bit=8
    n_bin=np.power(2,n_bit)
    n_bit_half=round(n_bit/2.0)

    fstops=[]
    histo,bound,_ = histogram_HDR(img, n_bin, 'log2', 0,0)
    dMM = (bound[1] - bound[0])/n_bin

    if eh_overlap > n_bit_half:
        eh_overlap=0.0

    removing_bins=round((n_bit_half - eh_overlap)/dMM)

    while(sum(histo) > 1):
        # print sum(histo), histo.flatten()
        total = -1
        index = -1
        for i in range(int(removing_bins), int(n_bin) - int(removing_bins)+1):
            t_sum = sum(histo[i - int(removing_bins):i+int(removing_bins)-1])

            if t_sum > total:
                index=i
                total = t_sum
        if index > 0:
            histo[index - int(removing_bins):index + int(removing_bins)] = 0
            value = -(index * dMM+bound[0]) - 1.0
            fstops.append(value)

    return fstops

def create_ldrstack_from_hdr(img, fstops_distance=1, 
                          sampling_mode='histogram', 
                          lin_type='gamma', 
                          lin_fun=2.2):
    '''a python reinplementation of CreateLDRStackFromHDR in HDR_toolbox'''

    L = lum(img)
    # print(L.shape);exit()
    minL=min(L[np.where(L>0)])
    maxL=max(L[np.where(L>0)])

    delta = 1e-6
    min_exposure = math.floor(math.log(maxL+delta, 2))
    max_exposure = math.ceil(math.log(minL+delta, 2))
    tMin = -int(min_exposure)
    tMax = -int(max_exposure+4)
    # print(tMin, tMax,range(tMin, tMax, fstops_distance));exit()

    if tMax > 2:
        tMax = 2
    # if tMin < -10:
    #     tMin=-10
    uniform_list = np.array(range(tMin, tMax, fstops_distance), dtype=np.float32)
    if sampling_mode=='1':
        # stack_exposure = np.array(np.power(2,[-int(round(math.log((min_exposure+max_exposure)/2+delta, 2)))]))
        range_list = np.array([(tMin+tMax)/2], dtype=np.float32)
        print(range_list)
        stack_exposure = np.power(2,range_list)

    elif sampling_mode=='2':
        if len(uniform_list)<=2:
            range_list = uniform_list
        else:
            range_list = np.array([(tMin+tMax)/2, (tMin+tMax)/2+1], dtype=np.float32)
        print(range_list)
        stack_exposure = np.power(2,range_list)

    elif sampling_mode=='3':
        if len(uniform_list)<=2:
            range_list = uniform_list
        else:
            range_list = np.array([(tMin+tMax)/2, (tMin+tMax)/2+1, (tMin+tMax)/2-1], dtype=np.float32)
        print(range_list)
        stack_exposure = np.power(2,range_list)
    elif sampling_mode=='4':
        if len(uniform_list)<=2:
            range_list = uniform_list
        else:
           range_list = np.array([(tMin+tMax)/2, (tMin+tMax)/2+1, (tMin+tMax)/2-1, (tMin+tMax)/2+2], dtype=np.float32)
        print(range_list)
        stack_exposure = np.power(2,range_list)
    elif sampling_mode=='5':
        if len(uniform_list)<=2:
            range_list = uniform_list
        else:
           range_list = np.array([(tMin+tMax)/2, (tMin+tMax)/2+1, (tMin+tMax)/2-1, (tMin+tMax)/2+2, (tMin+tMax)/2-2], dtype=np.float32)
        print(range_list)
        stack_exposure = np.power(2,range_list)
    elif sampling_mode=='6':
        if len(uniform_list)<=2:
            range_list = uniform_list
        else:
           range_list = np.array([(tMin+tMax)/2, (tMin+tMax)/2+1, (tMin+tMax)/2-1, (tMin+tMax)/2+2, (tMin+tMax)/2-2,(tMin+tMax)/2+3], dtype=np.float32)
        print(range_list)
        stack_exposure = np.power(2,range_list)
    elif sampling_mode=='7':
        if len(uniform_list)<=2:
            range_list = uniform_list
        else:
           range_list = np.array([(tMin+tMax)/2, (tMin+tMax)/2+1, (tMin+tMax)/2-1, (tMin+tMax)/2+2, (tMin+tMax)/2-2, (tMin+tMax)/2+3, (tMin+tMax)/2-3], dtype=np.float32)
        print(range_list)
        stack_exposure = np.power(2,range_list)

    elif sampling_mode=='histogram':
        # raise NotImplementedError
        stack_exposure=np.power(2, exposure_histogram_sampling(img))
    elif sampling_mode=='uniform' or sampling_mode=='amap':

        if minL == maxL:
            raise Exception('create_stack_from_hdr: all pixels have the same luminance value')
        if maxL <= 256 * minL:
            # raise Exception('create_stack_from_hdr: There is no need of sampling; i.e., 8-bit dynamic range.')
            print('Warning: There is no need of sampling; i.e., 8-bit dynamic range.')
            pass

        # range_list = np.array(range(tMin, tMax, fstops_distance), dtype=np.float32)
        range_list = uniform_list

        stack_exposure=np.power(2, range_list)

    elif sampling_mode=='selected':
        raise NotImplementedError
    else:
        raise NotImplementedError

    min_val=1/256.
    image_list = []
    # print(stack_exposure);exit()
    for exposure in stack_exposure:
        # img_e = img*exposure
        img_e = apply_crf(img*exposure, lin_type, lin_fun)
        # img = img[~np.isnan(img)]
        # np.warnings.filterwarnings('ignore')

        expo = clamp_img(img_e, 0, 1)
        image_list.append(expo)
        # if expo.min() <= (1- min_val) and expo.max() >= min_val:
        #     image_list.append(expo)

    return image_list, stack_exposure ## [ldr_img_num, height, width, channel], [exposure list]

if __name__=='__main__':
    hdr_img_dir = '/data/SaliencyDataset/Image/HDREYE/images/HDR'
    output_dir = 'ldr_stack_playground'
    hdr_path_list = glob.glob(os.path.join(hdr_img_dir, '*.hdr'))
    sampling_mode = 'amap'
    for hdr_path in hdr_path_list:
        # img_path='/data/SaliencyDataset/Image/HDREYE/images/HDR/C09.hdr'
        print(hdr_path)
        img = iio.imread(hdr_path)
        name = os.path.basename(hdr_path).split('.')[0]

        # ldr_stack, exposure_list = create_ldrstack_from_hdr(img, sampling_mode='uniform')
        ldr_stack, exposure_list = create_ldrstack_from_hdr(img, sampling_mode=sampling_mode)
        
        index = 0
        for i in range(len(ldr_stack)):
            out_name = name+'_'+str(index)+'.jpg'
            out_path = os.path.join(output_dir, out_name)
            ldr = ldr_stack[i][:,:,::-1]
            cv2.imwrite(out_path, ldr*255)
            index+=1