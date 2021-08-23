import sys

import os, re
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.regularizers import l1
from tensorflow.keras import metrics, optimizers, backend as K
from tensorflow.keras.callbacks import Callback

import keras_tuner.tuners as kt
from keras_tuner.engine.oracle import Objective

from options import parse_command_line_arguments

def sampling(arg):
    # Function with args required for Keras Lambda function
    mu, log_variance = arg                           

    # Draw epsilon of the same shape from a standard normal distribution
    epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)

    # The latent vector is non-deterministic and differentiable
    # in respect to enc_mu and enc_log_var
    random_sample = mu + K.exp(log_variance/2) * epsilon
    return random_sample

def build_vae_encoder(data_inp, latent_dim, depth, hidden_dim):            
    # Depending on the depth of the model, the input is eventually compressed into
    # a mean and log variance vector of prespecified size. Each layer is
    # initialized with glorot uniform weights and each step (dense connections,
    # batch norm,and relu activation) are funneled separately
    #
    # Each vector of length `latent_dim` are connected to the rnaseq input tensor
    # In the case of a depth 2 architecture, input_dim -> latent_dim -> hidden_dim 

    if depth == 0:
        enc_mu_dense =      Dense(latent_dim,
                                  kernel_initializer='glorot_uniform')(data_inp)
        enc_log_var_dense = Dense(latent_dim,
                                  kernel_initializer='glorot_uniform')(data_inp)
    else:
        hidden_dense = Dense(hidden_dim,
                                 kernel_initializer='glorot_uniform')(data_inp)
        hidden_dense_batchnorm = BatchNormalization()(hidden_dense)
        hidden_enc = Activation('relu')(hidden_dense_batchnorm) 
        for i in range(1,depth-1):
            hidden_dense = Dense(hidden_dim,
                                 kernel_initializer='glorot_uniform')(hidden_enc)
            hidden_dense_batchnorm = BatchNormalization()(hidden_dense)
            hidden_enc = Activation('relu')(hidden_dense_batchnorm)

        enc_mu_dense = Dense(latent_dim ,
                             kernel_initializer='glorot_uniform')(hidden_enc)
        enc_log_var_dense = Dense(latent_dim,
                                kernel_initializer='glorot_uniform')(hidden_enc)
    z_shape = (latent_dim, 1)
    enc_mu_dense_batchnorm = BatchNormalization()(enc_mu_dense)
    enc_mu = Activation('relu')(enc_mu_dense_batchnorm)

    enc_log_var_dense_batchnorm = BatchNormalization()(enc_log_var_dense)
    enc_log_var   = Activation('relu')(enc_log_var_dense_batchnorm)

    # return the encoded and randomly sampled z vector
    # Takes two keras layers as input to the custom sampling function layer with a
    # latent_dim` output
    z = Lambda(sampling,output_shape=z_shape)([enc_mu, enc_log_var])
    encoder_output = [z, enc_mu, enc_log_var]

    encoder = Model(data_inp, encoder_output , name='encoder')
    return encoder

def build_vae_decoder(latent_input, input_dim, latent_dim, hidden_dim, depth):
    # The layers are different depending on the prespecified depth.
    #
    # Single layer: glorot uniform initialized and sigmoid activation.
    # Double layer: relu activated hidden layer followed by sigmoid reconstruction
    if depth == 0:
        dense = Dense(input_dim, kernel_initializer='glorot_uniform',
                      activation='sigmoid')(latent_input)
    else:          
        dense1 = Dense(hidden_dim, kernel_initializer='glorot_uniform',
                          activation='relu', input_dim=latent_dim)(latent_input)
        for i in range(1,depth-1):
            dense1 = Dense(hidden_dim,activation='relu')(dense1)
        dense  = Dense(input_dim, kernel_initializer='glorot_uniform',
                      activation='sigmoid')(dense1)
    decoder = Model(inputs = latent_input, outputs = dense)
    return decoder

def vae_custom_loss(encoder_mu, encoder_log_variance, kappa):

    def vae_reconstruction_loss(y_true, y_predict):
#       reconstruction_loss = metrics.binary_crossentropy(y_true, y_predict)  
        reconstruction_loss = metrics.mean_squared_error(y_true, y_predict) 
        return reconstruction_loss

    def vae_kl_loss(y_true, y_predict, encoder_mu=1., encoder_log_variance=1.):
        #kl_loss = -0.5 * K.mean(1.0 + encoder_log_variance 
        kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance \
                                   - K.square(encoder_mu) \
                                   - K.exp(encoder_log_variance))
        return kl_loss

    def vae_loss(y_true, y_predict, encoder_mu=1., encoder_log_variance=1., 
                 kappa=1.):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict, encoder_mu=encoder_mu, \
                              encoder_log_variance=encoder_log_variance)
        loss = reconstruction_loss + kappa*kl_loss
        return loss

    return vae_loss

def build_vae_model(input_dim, latent_dim, depth, hidden_dim, kappa, lr):
    # Encoder
    encoder_input = Input(shape=(input_dim, ))
    encoder = build_vae_encoder(encoder_input, latent_dim, depth, hidden_dim)
    encoder_output = encoder(encoder_input)
    enc_z, enc_mu, enc_log_var = encoder_output 

    # Decoder
    decoder_input = Input(shape=(latent_dim,), name='z_sampling')
    decoder = build_vae_decoder(decoder_input, input_dim, latent_dim, \
                          hidden_dim, depth)
    decoder_output = decoder(enc_z)

    # Combined (vae_model)
    vae_model = Model(encoder_input, decoder_output)
    vae_model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                      loss=vae_custom_loss(enc_mu, enc_log_var, kappa))

    # Visualize the connections of the custom VAE model
    if depth == 1:
        output_model_file = os.path.join('figures', 'vae1_architecture.png')
    else:
        output_model_file = os.path.join('figures', 'vae3_architecture.png')

    return (vae_model, encoder, decoder)

def build_adage_encoder(data_input, latent_dim, sparsity, noise,
                        depth, hidden_dim):
    encoded_rnaseq = Dropout(noise)(data_input)
    encoded_rnaseq_2 = Dense(latent_dim,
                             activity_regularizer=l1(sparsity))(encoded_rnaseq)
    activation = Activation('relu')(encoded_rnaseq_2)
    for i in range(1,depth):
        encoded_rnaseq_2 = Dense(latent_dim,
                             activity_regularizer=l1(sparsity))(activation)            
        activation = Activation('relu')(encoded_rnaseq_2)
    encoder = Model(data_input, activation, name='encoder')

    return encoder

def build_adage_decoder(latent_input, input_dim, latent_dim, sparsity, noise,
                        depth, hidden_dim):
    decoded = latent_input                           
    for i in range(1,depth):
        activated = Activation('relu')(decoded)
        decoded = Dense(hidden_dim,
                         activity_regularizer=l1(sparsity))(activated)   
    decoded = Dense(input_dim)(decoded)     
    activated  = Activation('sigmoid')(decoded)            
    decoder = Model(inputs = latent_input, outputs = activated)

    return decoder

def build_adage_model(input_dim, latent_dim, sparsity, noise, 
                      depth, hidden_dim, lr):
    # Encoder
    data_input = Input(shape=(input_dim, ))
    encoder = build_adage_encoder(data_input, latent_dim, sparsity, noise,
                                  depth, hidden_dim)

    # Decoder
    latent_input = Input(shape=(latent_dim,))
    decoder = build_adage_decoder(latent_input, input_dim, latent_dim, 
                                  sparsity, noise, depth, hidden_dim)

    adage_model = Model(inputs = data_input,\
                        outputs = decoder(encoder(data_input)))
    adage_model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                        loss='mse')
    return (adage_model, encoder, decoder)

def build_combined_model(model_name, input_dim, latent_dim,
                         depth, hidden_dim, kappa, lr,
                         noise, sparsity):
    if model_name == "vae":
        combined_model, encoder, decoder = \
            build_vae_model(input_dim, latent_dim, depth, hidden_dim, kappa, lr)
    elif model_name == "adage":
        combined_model, encoder, decoder = \
            build_adage_model(input_dim, latent_dim, sparsity, noise, 
                              depth, hidden_dim, lr)
    else:
        sys.exit("Unsupported model name: " + model_name)
    return (combined_model, encoder, decoder)

def hypermodel_wrapper(opt):
    def hypermodel(hp):
        hidden_dim = hp.Choice('hidden_dim', 
                     [int(h) for h in opt.hidden_dim.split(",")])
        depth      = hp.Choice('depth',      
                     [int(d) for d in opt.depth.split(",")])             
        kappa      = hp.Choice('kappa',      
                     [float(k) for k in opt.kappa.split(",")]) 
        learning_rate = hp.Choice('learning_rate', 
                     [float(r) for r in opt.lr.split(",")])
        combined_model, _, _ = \
            build_combined_model(opt.model_name, opt.input_dim, opt.latent_dim,
                                 depth, hidden_dim, kappa,
                                 opt.noise, opt.sparsity, learning_rate) 
        return combined_model
    return hypermodel

class RandomSearch(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        opt, _, _, _ = parse_command_line_arguments("train")
        kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size',
                               [int(s) for s in opt.batch_size.split(",")])  
        kwargs['epochs']     = trial.hyperparameters.Choice('num_epochs',
                               [int(e) for e in opt.num_epochs.split(",")])
        super(RandomSearch, self).run_trial(trial, *args, **kwargs)

class BayesianOptimization(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        opt, _, _, _ = parse_command_line_arguments("train")
        kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size',
                               [int(s) for s in opt.batch_size.split(",")])
        kwargs['epochs']     = trial.hyperparameters.Choice('num_epochs',
                               [int(e) for e in opt.num_epochs.split(",")])
        super(BayesianOptimization, self).run_trial(trial, *args, **kwargs)

class Hyperband(kt.Hyperband):    
    def run_trial(self, trial, *args, **kwargs):
        opt, _, _, _ = parse_command_line_arguments("train")
        kwargs['batch_size']   = trial.hyperparameters.Choice('batch_size',
                                 [int(s) for s in opt.batch_size.split(",")])
        kwargs['epochs']       = trial.hyperparameters.Choice('num_epochs',
                                 [int(e) for e in opt.num_epochs.split(",")])
        super(Hyperband, self).run_trial(trial, *args, **kwargs)  



