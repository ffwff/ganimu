import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization,Reshape,GlobalAveragePooling2D, UpSampling2D, Lambda,LeakyReLU
from keras import optimizers
import os
import cv2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard, LambdaCallback
from keras_contrib.losses import DSSIMObjective
from keras.losses import mse, binary_crossentropy
import keras.backend as K
from keras import metrics
import keras_contrib.backend as KC

isize = 90

#http://www.cs.toronto.edu/~jsnell/assets/perceptual_similarity_metrics_icip_2017.pdf
#Image Quality Assessment: From Error Visibility to Structural Similarity
#https://en.wikipedia.org/wiki/Structural_similarity
#https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf?gi=389deacd645c
#http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
#https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf?gi=d4c0e1939081
#https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776?gi=c9c50c11fda9
#http://kvfrans.com/variational-autoencoders-explained/

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
    return z_mean + K.exp(z_log_var/2) * epsilon * 0.01

# data
X = []
folder='anime-faces'
for cat in ['animal_ears', '1girl', 'akaza_akari', ':d']:
#folder='dataset'
#for cat in os.listdir(folder):
    if os.path.isfile(folder+'/'+cat): continue
    for img in os.listdir(folder+'/'+cat):
        path='%s/%s/%s'%(folder, cat, img)
        im = cv2.imread(path)
        if im is None: continue
        im = cv2.resize(im, (isize, isize))
        X.append(im)
X = np.array(X)/255.0
n_train = int(len(X)*0.9)
X_train, X_test = X[:n_train], X[n_train:]
        
input_img = Input(shape=(isize, isize, 3))
#3*3*3*2*2
x = Conv2D(256, 4, padding='same', strides=2)(input_img)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(128, 4, padding='same', strides=2)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(128, 4, padding='same', strides=2)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(64, 4, padding='same', strides=2)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Flatten()(x)

latent_dim = 128
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, name='encoder')([z_mean, z_log_var])

x = Dense(6*6*64, activation='relu', name='post_encoder')(z)
x = Reshape((6,6,64))(x)

x = Conv2D(64, 4, padding='same')(x)
x = LeakyReLU()(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(128, 4, padding='same')(x)
x = LeakyReLU()(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(128, 4, padding='same')(x)
x = LeakyReLU()(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(256, 4)(x)
x = LeakyReLU()(x)
x = UpSampling2D((2, 2))(x)

output_img = Conv2D(3, (2, 2), activation='sigmoid', padding='same', name='decoder')(x)

def on_epoch_end(epoch, logs):
    im = X_test[0]
    cv2.imshow('raw', im)
    d = model.predict(np.reshape(im, (1, isize, isize, 3)))
    cv2.imshow('decoded', d[0])
    cv2.imwrite('reconstruction/%d.jpg'%epoch,d[0]*255.0)
    cv2.waitKey(1000)
    
    '''
    for im in X_test:
        #im = X_test[2]
        cv2.imshow('raw', im)
        d = model.predict(np.reshape(im, (1, isize, isize, 3)))
        cv2.imshow('decoded', d[0])
        cv2.waitKey()
        #cv2.waitKey(1000)
    '''

checkpoint = ModelCheckpoint("train1_{acc:.2f}_{val_acc:.2f}.hdf5", monitor='loss', verbose=1, save_best_only=True)
lc = LambdaCallback(on_epoch_end=on_epoch_end)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=5, min_lr=1e-5, verbose=1)

model = Model(input_img, output_img)
model.summary()
model.save('model1.h5')
model.load_weights('train1_0.81_0.80.hdf5')

class VAEObjective():
    
    def __init__(self):
        self.__name__ = 'VAEObjective'
        self.dssim = DSSIMObjective()
        self.original_dim=isize*isize*3
    
    def __call__(self, y_true, y_pred):
        '''
        #reconstruction_loss = metrics.mse(y_pred, y_true)*self.original_dim
        reconstruction_loss = self.dssim(y_true, y_pred)
        return K.mean(reconstruction_loss + 0.1*kl_loss)
        '''
        
        dssim_loss = self.dssim(y_true,y_pred)
        mse_loss = metrics.mse(K.flatten(y_true), K.flatten(y_pred))*self.original_dim
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(mse_loss*0.9+dssim_loss+kl_loss)

model.compile(optimizer=optimizers.Adam(lr=1e-5), loss=VAEObjective(), metrics=['acc'])
model.fit(X_train, X_train,
                epochs=100,
                batch_size=10,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[checkpoint, lc,reduce_lr])
