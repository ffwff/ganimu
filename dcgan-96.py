import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization,Reshape,GlobalAveragePooling2D, UpSampling2D, Lambda,LeakyReLU,ZeroPadding2D,Activation,Conv2DTranspose
from keras import optimizers
import os
import cv2
from keras.callbacks import ModelCheckpoint
from keras import losses
import keras.backend as K
from math import floor,ceil
import keras.initializers

# options
isize = 96
original_dim =isize*isize*3
latent_dim = 128

dis_optimizer = optimizers.Adam(1e-4,0.5)
optimizer = optimizers.Adam(4e-4,0.5)

# data
X = []
folder='extracted' # should be around 5000
p=[]
for img in os.listdir(folder):
    path='%s/%s'%(folder, img)
    im = cv2.imread(path)
    if im is None: continue
    im = cv2.resize(im, (isize, isize))
    X.append((im-127.5)/127.5)
X = np.array(X)

# Generator
G = Sequential()

G.add(Dense(5760, input_dim=latent_dim))
G.add(BatchNormalization())
G.add(LeakyReLU(0.2))
G.add(Reshape((12, 12, 40)))

G.add(Conv2DTranspose(256, 5, padding="same", strides=2,
                      kernel_initializer=keras.initializers.glorot_normal()))
G.add(BatchNormalization())
G.add(LeakyReLU(0.2))

G.add(Conv2DTranspose(128, 5, padding="same", strides=2,
                      kernel_initializer=keras.initializers.glorot_normal()))
G.add(BatchNormalization())
G.add(LeakyReLU(0.2))

G.add(Conv2DTranspose(64, 5, padding="same", strides=2,
                      kernel_initializer=keras.initializers.glorot_normal()))
G.add(BatchNormalization())
G.add(LeakyReLU(0.2))

G.add(Cropping2D())
G.add(Conv2D(3, 5, padding="same",
             kernel_initializer=keras.initializers.glorot_normal()))
G.add(Activation("tanh"))

G.summary()
G.save('models/gen.h5')

# Discriminator
D = Sequential()

D.add(Conv2D(64, 5, strides=2, padding="same", input_shape=(96,96,3)))
D.add(BatchNormalization())
D.add(LeakyReLU(0.2))

D.add(Conv2D(128, 5, strides=2, padding="same"))
D.add(BatchNormalization())
D.add(LeakyReLU(0.2))

D.add(Conv2D(256, 5, strides=2, padding="same"))
D.add(BatchNormalization())
D.add(LeakyReLU(0.2))

D.add(Flatten())

#D.add(Dropout(0.3))
D.add(Dense(1, activation='sigmoid'))

D.summary()
D.save('models/dis.h5')
D.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

# combined
x = Input(shape=(latent_dim,))
gen_out = G(x)
d_out = D(gen_out)
d_out = Activation("linear", name="d_out")(d_out)
combined = Model(inputs=[x], outputs=[d_out])
combined.layers[2].trainable = False
combined.summary()
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


def save_models(epoch):
    G.save_weights('weights/gen_epoch_%d.hdf5' % epoch)
    D.save_weights('weights/dis_epoch_%d.hdf5' % epoch)

def load_models(epoch):
    G.load_weights('weights/gen_epoch_%d.hdf5' % epoch)
    D.load_weights('weights/dis_epoch_%d.hdf5' % epoch)

cnoise = np.random.normal(0., 1., (1, latent_dim))
def show_reconstruction(epoch,i):
    d = G.predict(cnoise)
    cv2.imwrite('constructions/%d-%d.jpg' % (epoch,i),(d[0]*127.5)+127.5)

load_models(65)
initial = 66
epochs = 80
batch_count = 10
n_batches = ceil(X.shape[0] / batch_count)
#valid_test = np.ones((len(X_test), 1))
for e in range(initial, epochs+1):
    print('-'*15, 'Epoch %d/%d' % (e,epochs), '-'*15)
    
    valid = np.random.random_sample((batch_count, 1))*0.3+0.7
    fake = np.random.random_sample((batch_count, 1))*0.3
    #batch = X
    batch = np.random.permutation(X)
    noises = np.random.normal(0., 1., (n_batches*batch_count, latent_dim))
    
    for _ in range(n_batches):
        noise = noises[_*batch_count:_*batch_count+batch_count]
        
        # select random batch of images
        if _==n_batches-1:
            imgs = batch[-batch_count:]
        else:
            imgs = batch[_*batch_count:_*batch_count+batch_count]
        
        # predict from vae
        gen_imgs = G.predict(noise)
        
        # train discriminator
        d_loss_real = D.train_on_batch(imgs, valid)
        d_loss_fake = D.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        g_loss = combined.train_on_batch(noise, valid)
        
        print(
            "%d (%d/%d) [D loss: %f %f] [G loss: %f]" %
            (e, _+1, n_batches, d_loss_real, d_loss_fake, g_loss)
        )
        
    show_reconstruction(e,_+1)
    save_models(e)
