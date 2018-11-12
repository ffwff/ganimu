import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization,Reshape,GlobalAveragePooling2D, UpSampling2D, Lambda,LeakyReLU,ZeroPadding2D,Activation,Conv2DTranspose
from keras import optimizers
import os
import cv2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard, LambdaCallback
from keras import losses
import keras.backend as K
from keras import metrics
import keras_contrib.backend as KC
from math import floor

# options
isize = 64
original_dim =isize*isize*3
latent_dim = 128

dis_initial_lr = 1e-4
initial_lr = 2e-4
dis_optimizer = optimizers.Adam(dis_initial_lr,0.5,clipvalue=0.5)
optimizer = optimizers.Adam(initial_lr,0.5,clipvalue=0.5)

kl_weight = 1
mse_th = 0.06
img_weight = 1e-3
nll_weight = 1e-5

# fns
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

def mean_gaussian_negative_log_likelihood(y_true, y_pred):
    nll = 0.5 * np.log(2 * np.pi) + 0.5 * K.square(y_pred - y_true)
    axis = tuple(range(1, len(K.int_shape(y_true))))
    return K.mean(K.sum(nll, axis=axis), axis=-1)

# data
X = []
folder='extracted' # should be around 5000
p=[]
for img in os.listdir(folder):
    path='%s/%s'%(folder, img)
    im = cv2.imread(path)
    if im is None: continue
    im = cv2.resize(im, (isize, isize))
    X.append(im)
    p.append(path)
X = np.array(X) / 255.0
n_train = int(len(X)*0.9)
X_train, X_test = X[:n_train], X[n_train:]
#print(p[n_train])
#raise Exception()

# encoder
def build_encoder():
    Einp = Input(shape=(isize,isize,3))
    x = Conv2D(256, 4, padding='same', strides=2)(Einp)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 4, padding='same', strides=2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(64, 4, padding='same', strides=2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    E = Model(inputs=[Einp], outputs=[z_mean, z_log_var])
    return E

E = build_encoder()
E.save('models/encoder.h5')
E.summary()

# decoder
def build_decoder():
    z_mean = Input(shape=(latent_dim,), name='z_mean')
    z_log_var = Input(shape=(latent_dim,), name='z_log_var')
    z = Lambda(sampling, name='z')([z_mean, z_log_var])
    x = Dense(8*8*64, activation='relu', name='post_lambda')(z)
    x = Reshape((8,8,64))(x)

    x = Conv2DTranspose(64, 4, padding='same', strides=2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(128, 4, padding='same', strides=2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(256, 4, padding='same', strides=2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    
    Dout = Conv2D(3, 4, activation='sigmoid', padding='same', name='decoder')(x)
    D = Model(inputs=[z_mean, z_log_var], outputs=[Dout])
    return D

D = build_decoder()
D.save('models/decoder.h5')
D.summary()

# vae
def build_vae():
    Vinp = Input(shape=(isize,isize,3))
    z_mean, z_log_var = E(Vinp)
    Vout = D([z_mean, z_log_var])
    vae = Model(inputs=[Vinp], outputs=[Vout])
    
    def loss(y_true, y_pred):
        reconstruction_loss = losses.mse(K.flatten(y_true), K.flatten(y_pred))*original_dim
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return reconstruction_loss+kl_weight*kl_loss
    vae.compile(loss=loss, optimizer=optimizer)
    
    return vae

vae = build_vae()
#vae.summary()

# discriminator
def build_discriminator():
    dinput_img = Input(shape=(isize, isize, 3))
    
    features = Conv2D(256, 4, strides=2, padding="same")(dinput_img)
    x = BatchNormalization(momentum=0.8)(features)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 4, strides=2, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)

    X = Conv2D(128, 4, strides=2, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(X)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    dout = Dense(1, activation='sigmoid')(x)
    
    discriminator = Model(inputs=[dinput_img], outputs=[dout])
    features_model = Model(inputs=[dinput_img], outputs=[features])
    return discriminator, features_model

discriminator, features_model = build_discriminator()
features_model.trainable = False
discriminator.summary()
discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimizer)
discriminator.save('models/dis.h5')

# combined
def build_combined():
    cinp = Input(shape=(isize, isize, 3))
    
    z_mean, z_log_var = E(cinp)
    cout_img = D([z_mean, z_log_var])
    cout_img = Activation('linear', name='cout_img')(cout_img)
    cout_features = features_model(cout_img)
    cout_features = Activation('linear', name='cout_features')(cout_features)
    
    combined = Model(inputs=[cinp], outputs=[cout_img, cout_features])
    combined.layers[4].trainable = False
    
    def loss(y_true, y_pred):
        img_loss = losses.mse(K.flatten(cinp), K.flatten(cout_img))*original_dim
        nll_loss = mean_gaussian_negative_log_likelihood(y_true, y_pred)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return img_weight*img_loss+nll_weight*nll_loss+kl_weight*kl_loss
    combined.compile(
        loss=loss,
        optimizer=optimizer)
    
    return combined

combined = build_combined()
combined.summary()

def save_models(epoch):
    vae.save_weights('weights/vae_epoch_%d.hdf5' % epoch)
    discriminator.save_weights('weights/dis_epoch_%d.hdf5' % epoch)

def load_models(epoch):
    vae.load_weights('weights/vae_epoch_%d.hdf5' % epoch)
    discriminator.load_weights('weights/dis_epoch_%d.hdf5' % epoch)

def show_reconstruction(epoch,i):
    im = X_train[0]
    d = vae.predict(np.reshape(im, (1, isize, isize, 3)))
    cv2.imwrite('reconstruction/%d-%d.jpg' % (epoch,i),d[0]*255.0)

print("training: %d, testing: %d" % (len(X_train), len(X_test)))

late_mode = True

load_models(20)
initial = 21
epochs = 40
batch_count = 10
n_batches = floor(X_train.shape[0] / batch_count)
valid_test = np.ones((len(X_test), 1))
for e in range(initial, epochs+1):
    print('-'*15, 'Epoch %d/%d' % (e,epochs), '-'*15)
    
    K.set_value(discriminator.optimizer.lr, dis_initial_lr*(0.9**(e-1)))
    K.set_value(combined.optimizer.lr, initial_lr*(0.9**(e-1)))
    
    valid = np.ones((batch_count, 1))
    fake = np.zeros((batch_count, 1))
    
    batch = np.random.permutation(X_train)
    
    train_G = True
    train_D = True
    
    for _ in range(n_batches):
        # select random batch of images
        imgs = batch[_*batch_count:_*batch_count+batch_count]
        
        # predict from vae
        gen_imgs = vae.predict(imgs)
        if not late_mode:
            mse_error = np.mean(np.square(imgs - gen_imgs))
        
        # train discriminator
        if train_D:
            d_loss_real = discriminator.train_on_batch(imgs, valid)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        else:
            d_loss_real = discriminator.test_on_batch(imgs, valid)
            d_loss_fake = discriminator.test_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        if late_mode or mse_error <= mse_th:
            late_mode = True
            
            # get learned features
            features = features_model.predict(imgs)
            
            # train vae with learned features
            if train_G:
                g_loss = combined.train_on_batch([imgs], [imgs, features])
            else:
                g_loss = combined.test_on_batch([imgs], [imgs, features])
            
            #ratio = d_loss_real/d_loss_fake
            #if ratio < 0.1: train_D = False
            #else: train_D = True
            
            print(
                "%d (%d/%d) [D loss: %f %f] [G loss: %.2f %.2f] %s" %
                (e, _+1, n_batches, d_loss_real, d_loss_fake, g_loss[0], g_loss[1], train_D)
            )
        else:
            # train vae with mse only
            g_loss = vae.train_on_batch(imgs, imgs)
            
            print(
                "%d (%d/%d) [mse_error: %.6f] [D loss: %f] [G loss: %.2f]" %
                (e, _+1, n_batches, mse_error, d_loss, g_loss)
            )
        
    show_reconstruction(e, 0)
    
    # evaluate
    if late_mode:
        features = features_model.predict(X_test)
        valid_loss = combined.evaluate([X_test], [X_test, features])
        print("Validation loss: %.2f, %.2f" % (valid_loss[0], valid_loss[1]))
    else:
        valid_loss = vae.evaluate(X_test, X_test)
        print("Validation loss: %.2f" % (valid_loss))
    
    save_models(e)
