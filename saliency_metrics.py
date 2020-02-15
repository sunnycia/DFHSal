import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from config import *

def normalize_map(s_map):
  # normalize the salience map (as done in MIT code)
  norm_s_map = (s_map - K.min(s_map))/((K.max(s_map)-K.min(s_map))*1.0)
  return norm_s_map


def mae(y_true, y_pred):
  total_prediction = batch_size
  return K.sum(K.abs(y_true-y_pred))/total_prediction

def mse(y_true, y_pred):
  total_prediction = batch_size
  return K.sum(K.square(K.abs(y_true-y_pred)))/total_prediction

def kld(y_true, y_pred):
  total_prediction = batch_size
  total_kld = 0
  for i in range(total_prediction):
    true = y_true[i]
    pred = y_pred[i]
    total_kld += K.sum(true*K.log(K.epsilon()+true/(pred+K.epsilon())))
  return total_kld/total_prediction

def cc(y_true, y_pred):
  total_prediction = batch_size
  total_cc = 0
  # print(total_prediction);exit()
  for i in range(total_prediction):
    pred = y_pred[i]
    true = y_true[i]

    s_map_norm = (pred - K.mean(pred))/K.std(pred)
    gt_norm = (true - K.mean(true))/K.std(true)
    r = K.sum(s_map_norm*gt_norm) / K.sum(K.sqrt((s_map_norm*s_map_norm)) * K.sum(gt_norm*gt_norm))
    total_cc += r

  return total_cc / total_prediction



# def sim(y_true, y_pred):
#   # here y_true is not discretized nor normalized
#   total_prediction = batch_size
#   total_sim = 0

#   for i in range(total_prediction):
#     pred = normalize_map(y_pred[i])
#     true = normalize_map(y_true[i])

#     pred = pred/(K.sum(pred)*1.0)
#     true = true/(K.sum(true)*1.0)

#     x,y = tf.where(true>0)
#     sim = 0.0
#     for j in zip(x,y):
#       sim = sim + K.min(y_true[j[0],j[1]],y_pred[j[0],j[1]])
#     total_sim += sim

#   return total_sim / total_prediction
