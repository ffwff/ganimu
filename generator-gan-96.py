import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Lambda, Input
from keras import optimizers
import os
import cv2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K

import random
import string
def random_digits():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

#isize=64
latent_dim=128

G = load_model("models/gen.h5")
#G.load_weights('weights/gen_epoch_23.hdf5')
#G.load_weights('weights/gen_epoch_33.hdf5')
G.load_weights('weights/gen_epoch_71.hdf5')

#---
n_faces = 40
rand_vecs = np.random.normal(0., 0.6, (n_faces, latent_dim))
print(np.min(rand_vecs), np.max(rand_vecs))
y_faces = G.predict([rand_vecs])
for i in range(n_faces):
    face = y_faces[i]*127.5+127.5
    face = face.astype("uint8")
    cv2.imshow('face', face)
    cv2.waitKey(500)
    cv2.imwrite('anis/face_'+random_digits()+'.jpg', face)
