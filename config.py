# uncompyle6 version 3.3.4
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.1 (default, Dec  5 2017, 16:33:41) 
# [GCC 4.8.5 20150623 (Red Hat 4.8.5-16)]
# Embedded file name: /data/sunnycia/hdr_works/source_code/hdr_saliency/training_keras/config.py
# Compiled at: 2019-06-16 14:17:25
# Size of source mod 2**32: 3340 bytes
import glob, os, time, datetime as dt, shutil
from loss import saliency_loss
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
# import model_defination
from loss import kl_divergence, correlation_coefficient, bhattacharyya_distance

debug_flag = False
loss_type = 'mae'
Finetune = False
learn_rate = 0.001
optimizer_type = 'sgd'
model_version = '0.1_dense'
restart = False
training_dataset_name = 'salicon'
test_dataset_name = 'hdreye'
epochs = 200
early_stop_patience_epoch = epochs // 10
lr_reduce_patience_epoch = epochs // 20

SPATIAL_SIZE = 224
WIDTH = 224
HEIGHT = 224

stream = 1 
batch_size = 20

if debug_flag:
    train_set_image_dir = 'dataset/salicon-2014-subset/train/images/1/'
    train_set_density_dir = 'dataset/salicon-2014-subset/train/density/1/'
    validation_set_image_dir = 'dataset/salicon-2014-subset/val/images/1/'
    validation_set_density_dir = 'dataset/salicon-2014-subset/val/density/1/'
else:
    if training_dataset_name == 'salicon':
        # raise NotImplementedError
        train_set_image_dir = 'dataset/salicon-2014/train/images/'
        train_set_density_dir = 'dataset/salicon-2014/train/density/'
        validation_set_image_dir = 'dataset/salicon-2014/val/images/'
        validation_set_density_dir = 'dataset/salicon-2014/val/density/'
    elif training_dataset_name == 'cat2000':
        train_set_image_dir = 'dataset/cat2000/images/'
        train_set_density_dir = 'dataset/cat2000/density/'
        validation_set_image_dir = 'dataset/salicon-2014-subset/val/images/1/'
        validation_set_density_dir = 'dataset/salicon-2014-subset/val/density/1/'
    elif training_dataset_name == 'hollywood':
        train_set_image_dir = '/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/allinone/frames/'
        train_set_density_dir = '/data/SaliencyDataset/Video/ActionInTheEye/Hollywood2/allinone/density/'
        validation_set_image_dir = 'dataset/salicon-2014-subset/val/images/1/'
        validation_set_density_dir = 'dataset/salicon-2014-subset/val/density/1/'
    elif training_dataset_name == 'ledov':
        train_set_image_dir = '/data/SaliencyDataset/Video/LEDOV/allinone/subset/frames/'
        train_set_density_dir = '/data/SaliencyDataset/Video/LEDOV/allinone/subset/density/'
        validation_set_image_dir = 'dataset/salicon-2014-subset/val/images/1/'
        validation_set_density_dir = 'dataset/salicon-2014-subset/val/density/1/'
    # train_set_image_dir = 'dataset/mit1003/images/'
    # train_set_density_dir = 'dataset/mit1003/density/'
    # validation_set_image_dir = 'dataset/salicon-2014-subset/val/images/1/'
    # validation_set_density_dir = 'dataset/salicon-2014-subset/val/density/1/'

steps_per_epoch = len(glob.glob(os.path.join(train_set_image_dir, '*.*'))) // batch_size
validation_step = len(glob.glob(os.path.join(validation_set_image_dir, '*.*'))) // batch_size

def get_input_output_dir(dataset_name):
    if dataset_name == 'mit1003':
        image_dir = '/data/SaliencyDataset/Image/MIT1003/ALLSTIMULI'
        output_dir = '/data/SaliencyDataset/Image/MIT1003/saliency'
        fixation_dir = '/data/SaliencyDataset/Image/MIT1003/fixPts'
        density_dir = '/data/SaliencyDataset/Image/MIT1003/ALLFIXATIONMAPS'
    elif dataset_name == 'cat2000':
        image_dir = '/data/SaliencyDataset/Image/CAT2000/trainSet/Combine/Stimuli'
        output_dir = '/data/SaliencyDataset/Image/CAT2000/trainSet/Combine/saliency'
        fixation_dir = '/data/SaliencyDataset/Image/CAT2000/trainSet/Combine/fixation'
        density_dir = '/data/SaliencyDataset/Image/CAT2000/trainSet/Combine/density'
    elif dataset_name == 'sice':
        image_dir = '/data/ImageDataset/SICE/All_in_one'
        output_dir = '/data/ImageDataset/SICE/Saliency_map'
        # fixation_dir = '/data/SaliencyDataset/Image/CAT2000/trainSet/Combine/fixation'
        # density_dir = '/data/SaliencyDataset/Image/CAT2000/trainSet/Combine/density'
        fixation_dir = None
        density_dir = None
    elif dataset_name == 'hdreye':
        image_dir = '/data/SaliencyDataset/Image/HDREYE/images/HDR'
        output_dir = '/data/SaliencyDataset/Image/HDREYE/saliency_map/HDR'
        fixation_dir = '/data/SaliencyDataset/Image/HDREYE/fixation_map/HDR'
        density_dir = '/data/SaliencyDataset/Image/HDREYE/density_map/HDR-sigma32'
    else:
        raise NotImplementedError
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return image_dir, output_dir, fixation_dir, density_dir

if optimizer_type == 'adadelta':
    optimizer = optimizers.Adadelta(lr=learn_rate, rho=0.95, epsilon=K.epsilon(), decay=0.0001)
elif optimizer_type == 'sgd':
    optimizer = optimizers.SGD(lr=learn_rate, decay=0.0001, momentum=0.95)
elif optimizer_type == 'rms':
    optimizer = optimizers.RMSprop(lr=learn_rate, rho=0.9, epsilon=None, decay=0.0)
else:
    raise NotImplementedError

if loss_type == 'kld':
    loss = kl_divergence
elif loss_type == 'cc':
    loss = correlation_coefficient
elif loss_type == 'bhatd':
    loss = bhattacharyya_distance
elif loss_type == 'mae':
    loss = 'mean_absolute_error'
elif loss_type == 'mse':
    loss = 'mean_squared_error'
elif loss_type == 'shing':
    loss = 'squared_hinge'
elif loss_type == 'catehing':
    loss = 'categorical_hinge'
elif loss_type == 'msle':
    loss = 'mean_squared_logarithmic_error'
elif loss_type == 'hing':
    loss = 'hinge'
elif loss_type == 'catexentro':
    loss = 'categorical_crossentropy'
elif loss_type == 'cosprox':
    loss = 'cosine_proximity'
elif loss_type == 'pois':
    loss = 'poisson'
else:
    raise NotImplementedError

model_txt = 'model_list.txt'
save_base = 'output/'
if debug_flag:
    save_base='output/debug/'
metric_base = 'metric'
model_name = 'srsn_saliency.h5'
checkpoint_name = 'checkpoint.h5'
other_num = 10
no_sauc = 1
metric_debug = 0
statistics_flag = 1
output_latex = True
latex_file = 'metric.txt'
# okay decompiling __pycache__/config.cpython-35.pyc
