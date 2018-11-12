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

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    nd = K.shape(z_mean)[0]
    nc = K.shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(nd, nc))
    return z_mean + K.exp(z_log_var/2) * epsilon * 0.001

def build_vae():
    Vinp = Input(shape=(isize,isize,3))
    z_mean, z_log_var = E(Vinp)
    Vout = D([z_mean, z_log_var])
    vae = Model(inputs=[Vinp], outputs=[Vout])
    return vae

isize=64

E = load_model("models/encoder.h5")
z_mean, z_log_var = E.outputs
z = Lambda(sampling, name='z')([z_mean, z_log_var])
E_ = Model(inputs=E.inputs, outputs=[z])

D = load_model("models/decoder.h5")
decode = K.function([D.get_layer('post_lambda').input],
                  [D.layers[-1].output])

E_.summary()

vae = build_vae()
vae.load_weights('weights/vae_epoch_22.hdf5')

#---
n_faces = 30
rand_vecs = np.random.normal(0.000, 0.02, (n_faces, 128))
print(np.min(rand_vecs), np.max(rand_vecs))
y_faces = decode([rand_vecs, 0])[0]
for i in range(n_faces):
    face = y_faces[i]*255
    face = face.reshape((isize, isize, 3)).astype("uint8")
    print(np.min(face), np.max(face))
    cv2.imshow('face', face)
    cv2.waitKey(500)
    cv2.imwrite('7/face_'+random_digits()+'.jpg', face)
