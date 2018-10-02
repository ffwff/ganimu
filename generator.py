import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization,Reshape,GlobalAveragePooling2D, UpSampling2D
from keras import optimizers
import os
import cv2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K

import random
import string
def random_digits():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

isize=90

# data
X = []
folder='anime-faces'
for cat in ['white_background']:
#for cat in ['animal_ears', '1girl', 'akaza_akari', ':d']:
    for img in os.listdir(folder+'/'+cat):
        path='%s/%s/%s'%(folder,cat, img)
        im = cv2.imread(path)
        im = cv2.resize(im, (isize,isize))
        X.append(im)
X = np.array(X)/255.0
batch_size=12

model = load_model("model1.h5")
model.load_weights('train1_0.80_0.77.hdf5')

encoder = Model(input=model.input, output=model.get_layer('encoder').output)

func = K.function([model.get_layer('post_encoder').input],
                  [model.layers[-1].output])

n_faces = 30
x_enc = encoder.predict(X, batch_size=batch_size)
enc_shape = x_enc.shape
#x_enc = np.reshape(x_enc, (enc_shape[0], -1))
x_mean = np.mean(x_enc, axis=0)
x_stds = np.mean(x_enc, axis=0)
x_cov = np.cov((x_enc - x_mean).T)
e,v = np.linalg.eig(x_cov)
rand_vecs = np.random.normal(0.0, 1.0, (n_faces, *x_mean.shape))
x_vecs = x_mean + np.dot(v, (rand_vecs * e).T).T
#x_vecs = np.reshape(x_vecs, (n_faces, enc_shape[1], enc_shape[2], enc_shape[3]))
y_faces = func([x_vecs, 0])[0]
for i in range(n_faces):
    face = y_faces[i]
    face = face.reshape((isize, isize, 3))
    cv2.imshow('face', face)
    cv2.waitKey(500)
    cv2.imwrite('1/face_'+random_digits()+'.jpg', face*255.0)

'''
x_cov = np.cov((x_enc - x_mean).T)
e,v = np.linalg.eig(x_cov)
for i in range(n_faces):
    face = y_faces[i]
    face = face.reshape((isize, isize, 3))
    cv2.imshow('face', face)
    cv2.waitKey(100)
    cv2.imwrite('g7/face_'+random_digits()+'.jpg', face*255.0)

for i in range(n_faces):
    z = x_mean + x_stds*rand_vecs[i]
    face = func([z.reshape(1, *x_mean.shape)])[0]
    face = face.reshape((isize, isize, 3))
    cv2.imshow('face', face)
    cv2.waitKey(100)
    cv2.imwrite('face_'+random_digits()+'.jpg', face*255.0)
'''
