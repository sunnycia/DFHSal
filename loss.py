import tensorflow.keras.backend as K
import numpy as np




def kld_loss(y_true, y_pred):
    y_pred -= K.min(y_pred)
    y_pred /= K.max(y_pred)
    y_true -= K.min(y_true)
    y_true /= K.max(y_true)

    y_true = K.clip(y_true, K.epsilon(), 1.0)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0)

    return  - K.sum((y_true*K.log(y_true/y_pred)), axis=[1, 2, 3], keepdims=True) 

def saliency_loss(y_true,y_pred):
  return 10 * kl_divergence(y_true,y_pred)-2 * correlation_coefficient(y_true,y_pred)

def bhattacharyya_distance(y_true, y_pred):
    # thanks to 
    # https://github.com/anonymauthor/gpkeras/blob/528565ba1d94d4e659ec2bed42855b86716623ad/gpkeras/losses.py
    batch, height, width, channel=y_pred.get_shape().as_list()
    eps = K.epsilon()

    # sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=1), axis=1), axis=-2), height, axis=1), axis=-2), width, axis=2)
    # sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=1), axis=1), axis=-2), height, axis=1), axis=-2), width, axis=2)
    # y_true /= (sum_y_true + K.epsilon())
    # y_pred /= (sum_y_pred + K.epsilon())

    return -K.log(K.sum(K.sum(K.sqrt(y_true * y_pred), axis=-2), axis=-2))

# KL-Divergence Loss
def kl_divergence(y_true, y_pred):
    # thanks to SAM model
    batch, height, width, channel=y_pred.get_shape().as_list()
    eps = K.epsilon()
    # y_pred = K.permute_dimensions(y_pred, (0,3,1,2))
    # y_true = K.permute_dimensions(y_pred, (0,3,1,2))
    # print(y_pred.get_shape())

    # max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1), axis=-2), height, axis=1), axis=-2), width, axis=2)
    # min_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1), axis=-2), height, axis=1), axis=-2), width, axis=2)
    # # print(max_y_pred.get_shape())
    # y_pred -= min_y_pred
    # y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=1), axis=1), axis=-2), height, axis=1), axis=-2), width, axis=2)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=1), axis=1), axis=-2), height, axis=1), axis=-2), width, axis=2)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    return K.sum(K.sum(y_true * K.log((y_true / (y_pred + eps)) + eps), axis=-2), axis=-2)

# def kullback_leibler_divergence(y_true, y_pred):
#     #thanks to https://github.com/ndrplz/dreyeve/blob/f3dd6da24578ce0ac60b5a9a1870767c2c6e5360/experiments/train/loss_functions.py
#     """
#     Kullback-Leiber divergence (sec 4.2.3 of [1]). Assumes shape (b, 1, h, w) for all tensors.
#     :param y_true: groundtruth.
#     :param y_pred: prediction.
#     :param eps: regularization epsilon.
#     :return: loss value (one symbolic value per batch element).
#     """
#     eps = K.epsilon()
#     # y_pred = y_pred/K.max(y_pred,axis=-1)
#     P = y_pred
#     P = P / (eps + K.max(P, axis=[1, 2, 3], keepdims=True))
#     Q = y_true
#     Q = Q / (eps + K.max(Q, axis=[1, 2, 3], keepdims=True))

#     kld = K.sum(Q * K.log(eps + Q/(eps + P)), axis=[1, 2, 3])

#     return kld

# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    # N = shape_r_out * shape_c_out
    batch, height, width, channel = y_pred.get_shape().as_list()
    eps = K.epsilon()

    ## not necessary when you have sigmoid activation layer
    # max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), shape_r_out, axis=-1)), shape_c_out, axis=-1)
    # y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=1), axis=1), axis=-2), height, axis=1), axis=-2), width, axis=2)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=1), axis=1), axis=-2), height, axis=1), axis=-2), width, axis=2)

    y_true /= (sum_y_true + eps)
    y_pred /= (sum_y_pred + eps)

    N = width * height

    sum_prod = K.sum(K.sum(y_true * y_pred, axis=1), axis=1)
    sum_x = K.sum(K.sum(y_true, axis=1), axis=1)
    sum_y = K.sum(K.sum(y_pred, axis=1), axis=1)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=1), axis=1)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=1), axis=1)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    return - num / den
    # return -2 * num / den

# Normalized Scanpath Saliency Loss
def nss(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred
    y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_mean)), 
                                                               shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_std)), 
                                                              shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    return -(K.sum(K.sum(y_true * y_pred, axis=2), axis=2) / K.sum(K.sum(y_true, axis=2), axis=2))
