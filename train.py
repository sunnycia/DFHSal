#import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K, optimizers
from tensorflow.keras.activations import sigmoid, relu
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import os, glob, shutil
import time
from saliency_metrics import kld, cc, mse, mae
from dataset_utils import generator, preprocess_images, postprocess_predictions
from loss import saliency_loss
import model_defination

####################
##### Basic
##### Configuration
####################
from config import *
import config as cfg

dtime = dt.datetime.now()
# time_stamp = int(time.time())
save_dir= cfg.save_base+'_'.join(str(dtime).split('.'))
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

save_script_path_list = glob.glob(os.path.join('./', '*.py'))
for save_script_path in save_script_path_list:
  shutil.copyfile(save_script_path, os.path.join(save_dir, os.path.basename(save_script_path)))

####################
##### Model
##### Defination
####################
model = model_defination.get_model(cfg.model_version, spatial_size=cfg.SPATIAL_SIZE)

# model = cfg.model

####################
##### Training
##### Configuration
####################
metric_list = [mae, mse, cc, kld]
# optimizer = optimizers.Adadelta(lr=learn_rate, rho=0.95, epsilon=K.epsilon(), decay=0.0001)
optimizer = cfg.optimizer
model.compile(optimizer=optimizer, loss=cfg.loss, metrics=metric_list)
train_gen = generator(cfg.train_set_image_dir,cfg.train_set_density_dir,shape_r=cfg.SPATIAL_SIZE, shape_c=cfg.SPATIAL_SIZE, b_s=cfg.batch_size)
val_gen = generator(cfg.validation_set_image_dir, cfg.validation_set_density_dir, shape_r=cfg.SPATIAL_SIZE, shape_c=cfg.SPATIAL_SIZE, b_s=cfg.batch_size)

## callback
# checkpointer = ModelCheckpoint(filepath=os.path.join(save_dir,cfg.checkpoint_name), verbose=1, save_best_only=True, peroid=5,mode='auto')
checkpointer = ModelCheckpoint(filepath=os.path.join(save_dir,cfg.checkpoint_name), verbose=1, save_best_only=True, mode='auto')
earlystopper = EarlyStopping(monitor='val_loss',patience=cfg.early_stop_patience_epoch, mode='auto',restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=cfg.lr_reduce_patience_epoch, verbose=0, mode='auto', min_delta=0.0001, cooldown=5, min_lr=1e-7)


####################
##### Training
####################
history = model.fit_generator(train_gen, epochs=cfg.epochs,validation_data=val_gen, steps_per_epoch=cfg.steps_per_epoch, validation_steps=cfg.validation_step, verbose=2, shuffle=True, use_multiprocessing=False,workers=1, callbacks=[checkpointer, earlystopper, lr_reducer])

####################
##### After training
##### save model
##### Plot loss
####################

model_path = os.path.join(save_dir, cfg.model_name)
# model.save(model_path)
tf.keras.models.save_model(model, model_path)
print('Saved trained model at %s ' % model_path)

# model performance skim
img_path_list = ['test_imgs/face.jpg', 'test_imgs/bulldog.jpg']
img_arr = preprocess_images(img_path_list,cfg.SPATIAL_SIZE, cfg.SPATIAL_SIZE)

predictions = model.predict(img_arr, batch_size=len(img_path_list))
# print(prediction.shape)
# for prediction in predictions:
for i in range(len(img_path_list)):
  img_name = img_path_list[i]
  img_path = os.path.join(save_dir, img_name)
  prediction = postprocess_predictions(predictions[i], cfg.SPATIAL_SIZE, cfg.SPATIAL_SIZE)

  cv2.imwrite(img_path, prediction)

plt.plot(history.history['kld'])
plt.plot(history.history['mse'])
plt.plot(history.history['mae'])
plt.title('Model accuracy')
plt.ylabel('metric')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
