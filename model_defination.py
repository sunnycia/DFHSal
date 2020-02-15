import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
# from keras_applications.densenet import DenseNet
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose,BatchNormalization, AveragePooling2D, Concatenate, Dense, Reshape, Lambda, Dropout
from tensorflow.keras.layers import Multiply,Add, UpSampling2D, multiply, GaussianNoise
from tensorflow.keras.activations import sigmoid
import config as cfg
# from tensorflow.keras.layers import merge
# from tensorflow.keras.layers.pooling import AveragePooling2D

# d_rate = 1 # dilation rate



def pam(x):
  gamma = K.variable(np.array([0]), dtype='float32',name='gamma')

  # channel = 2048
  # spatial_size = height = width = 7
  batch, height, width, channel = x.get_shape().as_list()
  assert height==width, "height and width not equal."

  proj_query = Conv2D(height, 1, padding='same', strides=1)(x)
  proj_query = Reshape((height*width, height))(proj_query)
  # print(proj_query.get_shape());exit()
  proj_query = K.permute_dimensions(proj_query, (0,2,1))

  proj_key = Conv2D(height, 1, padding='same', strides=1)(x)
  proj_key = Reshape((height*width, height))(proj_key)

  proj_value = Conv2D(channel, 1, padding='same', strides=1)(x)
  proj_value = Reshape((height*width, channel))(proj_value)
  energy = K.batch_dot(proj_key, proj_query)
  attention = K.softmax(energy)
  attention = K.permute_dimensions(attention, (0,2,1))
  out = K.batch_dot(attention, proj_value)

  out = Reshape((height, width, channel))(out)
  # out = Add()([Multiply()([gamma,out]), x])
  out = x+gamma*out

  return out

def cam(x):
  # raise NotImplementedError
  batch, height, width, channel = x.get_shape().as_list()

  gamma = K.variable(np.array([0]), dtype='float32',name='gamma')

  proj_query = Reshape((height*width, channel))(x)
  proj_key = Reshape((height*width, channel))(x)
  proj_value = Reshape((height*width, channel))(x)

  print(proj_key.get_shape(), proj_query.get_shape());
  proj_key = K.permute_dimensions(proj_key, (0,2,1))
  energy = K.batch_dot(proj_key, proj_query)
  # energy_new = 
  attention = K.softmax(energy)
  out = K.batch_dot(proj_value, attention)
  out = Reshape((height, width, channel))(out)

  out = gamma * out + x
  return out

def upsampling_module(x):
  x = Conv2DTranspose(256, 4, padding='same', strides=4, dilation_rate=1)(x)
  x = BatchNormalization()(x)
  x = Conv2DTranspose(64, 4, padding='same', strides=2, dilation_rate=1)(x)
  x = BatchNormalization()(x)
  x = Conv2DTranspose(16, 4, padding='same', strides=2, dilation_rate=1)(x)
  x = BatchNormalization()(x)
  output = Conv2DTranspose(1, 4, padding='same', strides=2, dilation_rate=1)(x)
  return output

def upsampling_module_dropout(x, dropout_ratio=0.1):
  seed = 7
  x = Conv2DTranspose(256, 4, padding='same', strides=4, dilation_rate=1)(x)
  x = BatchNormalization()(x)
  x = Dropout(dropout_ratio, seed=seed)(x)
  x = Conv2DTranspose(64, 4, padding='same', strides=2, dilation_rate=1)(x)
  x = BatchNormalization()(x)
  x = Dropout(dropout_ratio, seed=seed)(x)
  x = Conv2DTranspose(16, 4, padding='same', strides=2, dilation_rate=1)(x)
  x = BatchNormalization()(x)
  x = Dropout(dropout_ratio, seed=seed)(x)
  output = Conv2DTranspose(1, 4, padding='same', strides=2, dilation_rate=1)(x)
  return output

def model_v0_1_test(input_size=224, Finetune=False):
  base_model = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  base_model.summary()
  x = base_model.layers[-3].output

  prediction = upsampling_module(x)

  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1(input_size=448, d_rate=1, Finetune=False):
  base_model = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  base_model.summary()
  x = base_model.layers[-3].output

  x = Conv2DTranspose(256, 4, padding='same', strides=4, dilation_rate=d_rate)(x)
  x = BatchNormalization()(x)
  x = Conv2DTranspose(64, 4, padding='same', strides=2, dilation_rate=d_rate)(x)
  x = BatchNormalization()(x)
  x = Conv2DTranspose(16, 4, padding='same', strides=2, dilation_rate=d_rate)(x)
  x = BatchNormalization()(x)
  prediction = Conv2DTranspose(1, 4, padding='same', strides=2, dilation_rate=d_rate)(x)
  # prediction = BatchNormalization()(prediction)
  prediction = Lambda(sigmoid)(prediction)
  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1_trimmed(input_size=224, d_rate=1, Finetune=False):
  base_model = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  base_model.summary()
  # x = base_model.layers[-3].output
  x = base_model.get_layer('add_2').output ##(-1, 56, 56, 256)
  # exit()

  x = Conv2DTranspose(64, 4, padding='same', strides=2, dilation_rate=d_rate)(x)
  x = BatchNormalization()(x)
  prediction = Conv2DTranspose(1, 4, padding='same', strides=2, dilation_rate=d_rate)(x)
  # prediction = BatchNormalization()(prediction)
  # prediction = Lambda(sigmoid)(prediction)
  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1_trimmed_pam(input_size=224, d_rate=1, Finetune=False):
  base_model = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  base_model.summary()
  # x = base_model.layers[-3].output
  x = base_model.get_layer('add_2').output ##(-1, 56, 56, 256)
  # exit()

  x = Lambda(pam)(x)

  x = Conv2DTranspose(64, 4, padding='same', strides=2, dilation_rate=d_rate)(x)
  x = BatchNormalization()(x)
  prediction = Conv2DTranspose(1, 4, padding='same', strides=2, dilation_rate=d_rate)(x)
  # prediction = BatchNormalization()(prediction)
  # prediction = Lambda(sigmoid)(prediction)
  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1_trimmed_cam(input_size=224, d_rate=1, Finetune=False):
  base_model = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  base_model.summary()
  # x = base_model.layers[-3].output
  x = base_model.get_layer('add_2').output ##(-1, 56, 56, 256)
  # exit()
  x = Lambda(cam)(x)

  x = Conv2DTranspose(64, 4, padding='same', strides=2, dilation_rate=d_rate)(x)
  x = BatchNormalization()(x)
  prediction = Conv2DTranspose(1, 4, padding='same', strides=2, dilation_rate=d_rate)(x)
  # prediction = BatchNormalization()(prediction)
  # prediction = Lambda(sigmoid)(prediction)
  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1_cb(input_size=224, d_rate=1, Finetune=False, batch_size = cfg.batch_size):
  base_model = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  base_model.summary()
  x = base_model.layers[-3].output

  prediction = upsampling_module(x)

  # create center bias tensor
  # _, height, width, channel = prediction.get_shape().as_list()
  # cb = K.variable(cv2.resize(cv2.imread('center.jpg', 0), dsize=(input_size,input_size)))
  # cb = K.expand_dims(K.expand_dims(cb), axis=0)

  prediction = GaussianNoise(stddev=0.1)(prediction)
  # print(cb.get_shape());exit()

  # print(prediction, cb)
  # prediction = Multiply()([prediction, cb])
  # prediction = Lambda(sigmoid)(prediction)
  # print(prediction, cb)
  # prediction = prediction * cb
  # prediction = BatchNormalization()(prediction)
  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1_dropout(input_size=448, d_rate=1, Finetune=False):
  base_model = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  base_model.summary()
  x = base_model.layers[-3].output
  prediction = upsampling_module_dropout(x)
  # prediction = BatchNormalization()(x)
  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model



def model_v0_1_vgg(input_size=224,d_rate=1,Finetune=False, vgg_version='vgg16'):
  # print("Yeah")
  if vgg_version=='vgg16':
    base_model = VGG16(input_shape=(input_size,input_size,3),include_top=False)
  elif vgg_version=='vgg19':
    base_model = VGG19(input_shape=(input_size,input_size,3),include_top=False)
  else:
    raise NotImplementedError

  # base_model.summary();exit()
  x = base_model.layers[-2].output

  x = Conv2DTranspose(256, 4, padding='same', strides=2, dilation_rate=1)(x)
  x = BatchNormalization()(x)
  x = Conv2DTranspose(64, 4, padding='same', strides=2, dilation_rate=1)(x)
  x = BatchNormalization()(x)
  x = Conv2DTranspose(16, 4, padding='same', strides=2, dilation_rate=1)(x)
  x = BatchNormalization()(x)
  prediction = Conv2DTranspose(1, 4, padding='same', strides=2, dilation_rate=1)(x)

  # prediction = upsampling_module(x)

  model = Model(inputs=base_model.input, outputs=prediction)
  # model.summary()
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1_dense(input_size=448, d_rate=1, Finetune=False):
  # print("Yeah")
  base_model = DenseNet121(input_shape=(input_size,input_size,3),include_top=False)
  base_model.summary()
  x = base_model.layers[-3].output

  prediction = upsampling_module(x)

  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model


def model_v0_1_dense_pam(input_size=224, d_rate=1, Finetune=False):
  # print("Yeah")
  base_model = DenseNet121(input_shape=(input_size,input_size,3),include_top=False)
  base_model.summary()
  x = base_model.layers[-3].output

  x = Lambda(pam)(x)
  prediction = upsampling_module(x)

  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1_dense_cam(input_size=224, d_rate=1, Finetune=False):
  # print("Yeah")
  base_model = DenseNet121(input_shape=(input_size,input_size,3),include_top=False)
  base_model.summary()
  x = base_model.layers[-3].output

  x = Lambda(cam)(x)
  prediction = upsampling_module(x)

  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1_vgg_cam(input_size=224, d_rate=1, Finetune=False, vgg_version='vgg16'):
  # print("Yeah")
  if vgg_version=='vgg16':
    base_model = VGG16(input_shape=(input_size,input_size,3),include_top=False)
  elif vgg_version=='vgg19':
    base_model = VGG19(input_shape=(input_size,input_size,3),include_top=False)
  else:
    raise NotImplementedError

  x = base_model.layers[-2].output

  x = Lambda(cam)(x)
  x = Conv2DTranspose(256, 4, padding='same', strides=2, dilation_rate=1)(x)
  x = BatchNormalization()(x)
  x = Conv2DTranspose(64, 4, padding='same', strides=2, dilation_rate=1)(x)
  x = BatchNormalization()(x)
  x = Conv2DTranspose(16, 4, padding='same', strides=2, dilation_rate=1)(x)
  x = BatchNormalization()(x)
  prediction = Conv2DTranspose(1, 4, padding='same', strides=2, dilation_rate=1)(x)
  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model


def model_v0_1_pam(input_size=224, d_rate=1, Finetune=False):
  base_model = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  x = base_model.layers[-3].output

  x = Lambda(pam)(x)
  #PAM
  prediction = upsampling_module(x)
  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1_cam(input_size=224, d_rate=1, Finetune=False):
  base_model = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  x = base_model.layers[-3].output

  x = Lambda(cam)(x)
  # x = cam(x)
  #PAM
  prediction = upsampling_module(x)
  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1_p_c(input_size=224, d_rate=1, Finetune=False):
  base_model = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  x = base_model.layers[-3].output

  x = Lambda(pam)(x)
  x = Lambda(cam)(x)
  #PAM
  prediction = upsampling_module(x)
  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1_c_p(input_size=224, d_rate=1, Finetune=False):
  base_model = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  x = base_model.layers[-3].output

  x = Lambda(cam)(x)
  x = Lambda(pam)(x)

  #PAM
  prediction = upsampling_module(x)
  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_1_pc(input_size=224, d_rate=1, Finetune=False):
  base_model = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  x = base_model.layers[-3].output

  batch, height, width, in_channel = x.get_shape().as_list()
  out_channel = in_channel//4

  pam_input = Conv2D(out_channel, 1, padding='same', strides=1)(x)
  cam_input = Conv2D(out_channel, 1, padding='same', strides=1)(x)

  pam_out = Lambda(pam)(pam_input)
  cam_out = Lambda(cam)(cam_input)

  x = Add()([pam_out, cam_out])
  prediction = upsampling_module(x)

  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v0_2(input_size=384, d_rate=1, Finetune=False):
  base_model_lc = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  base_model_gb = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  pass


def model_v0_3(input_size=384, d_rate=1, Finetune=False):
  # https://stackoverflow.com/questions/49546922/keras-replacing-input-layer
  base_model_lc = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  base_model_nb = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  base_model_gb = ResNet50(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
  # base_model.get_weights()


  input_lc = Input(batch_shape=(0,input_size, input_size,3))
  input_nb = AveragePooling2D(2)(input_lc)
  input_gb = AveragePooling2D(4)(input_lc)
  base_model_lc.layers[0] = input_lc
  base_model_nb.layers[0] = input_nb
  base_model_gb.layers[0] = input_gb
  # print(base_model_nb.layers[0].get_shape())
  # print(input_nb.get_shape(), input_gb.get_shape());exit()
  feat_lc = base_model_lc.layers[-3].output
  feat_nb = base_model_nb.layers[-3].output
  feat_gb = base_model_gb.layers[-3].output

  ## concatenate three scale features
  # concat_feat = Concatenate(axis=-1)([feat_lc, feat_nb,feat_gb])
  concat_feat = K.concatenate([feat_lc,feat_nb,feat_gb], axis=-1)
  # print(concat_feat.get_shape())

  prediction = upsampling_module(concat_feat)
  # prediction = BatchNormalization()(x)

  model = Model(inputs=input_lc, outputs=prediction)

  if Finetune:
    for layer in base_model_lc.layers:
      layer.trainable=False
    for layer in base_model_nb.layers:
      layer.trainable=False
    for layer in base_model_gb.layers:
      layer.trainable=False
  return model


def model_v1_1(input_size=384, d_rate=1, Finetune=False):
  ## derive from v0_1_dense_pam

  base_model = DenseNet121(input_shape=(input_size,input_size,3),include_top=False)
  base_model.summary()
  x = base_model.layers[-3].output

  x = Lambda(pam)(x)
  weights_map = Conv2D(1, 1, padding='same', strides=1)(x)
  weights_map = UpSampling2D(size=(32,32))(weights_map)
  prediction = upsampling_module(x)
  prediction = Multiply()([weights_map, prediction])

  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def model_v1_1_cb(input_size=384, d_rate=1, Finetune=False):
  ## derive from v0_1_dense_pam

  base_model = DenseNet121(input_shape=(input_size,input_size,3),include_top=False)
  base_model.summary()
  x = base_model.layers[-3].output

  x = Lambda(pam)(x)
  weights_map = Conv2D(1, 1, padding='same', strides=1)(x)
  weights_map = UpSampling2D(size=(32,32))(weights_map)
  prediction = upsampling_module(x)
  prediction = Multiply()([weights_map, prediction])

  prediction = GaussianNoise(stddev=0.1)(prediction)
  model = Model(inputs=base_model.input, outputs=prediction)
  if Finetune:
    for layer in base_model.layers:
      layer.trainable = False
  return model

def get_model(model_version, spatial_size=224):
  # print('haha',model_version=='0.1')
  if model_version=='0.1_test':
    return model_v0_1_test(input_size=spatial_size)

  if model_version=='0.1': 
    ## basic ssrsn, 
    ## resnet50 + 4 batch norm deconvolutional layer
    return model_v0_1(input_size=spatial_size)

  if model_version=='0.1_trimmed':
    return model_v0_1_trimmed(input_size=spatial_size)

  if model_version=='0.1_trimmed_pam':
    return model_v0_1_trimmed_pam(input_size=spatial_size)

  if model_version=='0.1_trimmed_cam':
    return model_v0_1_trimmed_cam(input_size=spatial_size)

  if model_version=='0.1_cb':
    return model_v0_1_cb(input_size=spatial_size)

  if model_version=='0.1_dp':
    return model_v0_1_dropout(input_size=spatial_size)

  if model_version=='0.1_dense':
    return model_v0_1_dense(input_size=spatial_size)
  
  if model_version=='0.1_vgg':
    return model_v0_1_vgg(input_size=spatial_size)
  
  if model_version=='0.1_sa':##deprecated
    return model_v0_1_pam(input_size=spatial_size)

  if model_version=='0.1_pam': 
    ## add self attention module
    return model_v0_1_pam(input_size=spatial_size)

  if model_version=='0.1_dense_pam': 
    ## add self attention module
    return model_v0_1_dense_pam(input_size=spatial_size)

  if model_version=='0.1_dense_cam': 
    ## add self attention module
    return model_v0_1_dense_cam(input_size=spatial_size)

  if model_version=='0.1_vgg_cam':
    return model_v0_1_vgg_cam(input_size=spatial_size)

  if model_version=='0.1_cam': 
    ## add self attention module
    return model_v0_1_cam(input_size=spatial_size)

  if model_version=='0.1_p_c': 
    ## add self attention module
    return model_v0_1_p_c(input_size=spatial_size)

  if model_version=='0.1_c_p': 
    ## add self attention module
    return model_v0_1_c_p(input_size=spatial_size)

  if model_version=='0.1_pc': 
    ## add self attention module
    return model_v0_1_pc(input_size=spatial_size)

  if model_version=='0.2':
    ##  dsrsn
    raise NotImplementedError

  if model_version=='0.2.1':
    ##  dsrsn with self attention module
    raise NotImplementedError

  if model_version=='0.3':
    ## tsrsn
    return model_v0_3(input_size=spatial_size)

  if model_version=='0.3':
    ## tsrsn with self attention module  
    raise NotImplementedError

  if model_version =='1.1':
    ## densenet backbone, pam module, sigmoid, prior, mae loss
    return model_v1_1(input_size=spatial_size)

  if model_version =='1.1_cb':
    return model_v1_1_cb(input_size=spatial_size)

if __name__ =='__main__':
  model = model_v0_1_vgg()
