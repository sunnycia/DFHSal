import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose,BatchNormalization, AveragePooling2D, Concatenate, Dense, Reshape, Lambda, Dropout
from tensorflow.keras.layers import Multiply,Add, UpSampling2D, multiply, GaussianNoise
from tensorflow.keras.activations import sigmoid
import config as cfg



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

def model_v0_1_dense(input_size=448, d_rate=1, Finetune=False):
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



def get_model(model_version, spatial_size=224):
  if model_version=='0.1_dense_cam': 
    ## add self attention module
    return model_v0_1_dense_cam(input_size=spatial_size)