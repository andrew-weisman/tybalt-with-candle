#!/usr/bin/env python

# Tasks
# 1) observe reconstruction fidelity
# 2) encode
# 3) perform tSNE

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
import keras

from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot

from options import parse_command_line_arguments
from models import build_combined_model

sns.set(style="white", color_codes=True)
sns.set_context("paper", rc={"font.size":14,"axes.titlesize":15,"axes.labelsize":20,
                             'xtick.labelsize':14, 'ytick.labelsize':14})

def get_data():
    # Load Zero-One transformed (min-max scaled) Gene Expression/RNAseq data
    rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv.gz')
    # NOTE: rnaseq_file is produced by process_data.py
    rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
    rnaseq_df.head(2)

    # Split 10% test set randomly
    test_set_percent = 0.1
    rnaseq_test_df = rnaseq_df.sample(frac=test_set_percent)
    rnaseq_train_df = rnaseq_df.drop(rnaseq_test_df.index)

    # Set hyperparameters
    input_dim   = rnaseq_df.shape[1]
    latent_dim  = 100
    hidden_dim  = int(opt.hidden_dim) 
    kappa       = 1

    return (rnaseq_df, rnaseq_train_df, rnaseq_test_df, 
            input_dim, latent_dim, hidden_dim, kappa)

def evaluate_reconstruction_fidelity(encoded_rnaseq_df):
    # How well does the model reconstruct the input RNAseq data?
    input_rnaseq_reconstruct = decoder.predict(np.array(encoded_rnaseq_df))
    input_rnaseq_reconstruct = pd.DataFrame(input_rnaseq_reconstruct, \
        index=rnaseq_df.index, columns=rnaseq_df.columns)
    input_rnaseq_reconstruct.head(2)

    reconstruction_fidelity = rnaseq_df - input_rnaseq_reconstruct

    gene_mean = reconstruction_fidelity.mean(axis=0)
    gene_abssum = reconstruction_fidelity.abs().sum(axis=0).divide(rnaseq_df.shape[0])
    gene_summary = pd.DataFrame([gene_mean, gene_abssum], index=['gene mean', 'gene abs(sum)']).T
    gene_summary.sort_values(by='gene abs(sum)', ascending=False).head()
    g = sns.jointplot('gene mean', 'gene abs(sum)', data=gene_summary, stat_func=None);

def save_encoded_data(encoded_rnaseq_df, model):
    encoded_rnaseq_df.columns.name = 'sample_id'
    encoded_rnaseq_df.columns = encoded_rnaseq_df.columns + 1
    if model == 'vae':
        encoded_file = os.path.join('data', 'encoded_rnaseq_onehidden_warmup_batchnorm.tsv')
    elif model == 'adage':
        encoded_file = os.path.join('data', 'encoded_adage_features.tsv')
    encoded_rnaseq_df.to_csv(encoded_file, sep='\t')
    return

def perform_tSNE(df, model):
    df.head(2)

    if model == 'vae':
        tsne_out_file = os.path.join('results', 'vae_tsne_features.tsv')
    elif model == 'adage':
        tsne_out_file = os.path.join('results', 'adage_tsne_features.tsv')
    else: # use non-encoded data
        tsne_out_file = os.path.join('results', 'rnaseq_tsne_features.tsv')

    N = df.shape[0]       # total number of data noints
    lr = max(200, int(N/12))
    print("learning_rate=", lr)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, \
                         perplexity=20, learning_rate=lr, n_iter=400)
    tsne_out = tsne.fit_transform(df)
    tsne_out = pd.DataFrame(tsne_out, columns=['1', '2'])
    tsne_out.index = df.index
    tsne_out.index.name = 'tcga_id'
    if not os.path.isdir('results'):
        os.mkdir('results')
    tsne_out.to_csv(tsne_out_file, sep='\t')
    tsne_out.head(2)

if __name__ == '__main__':
    # Parse command line options
    opt, checkpoint_combined = parse_command_line_arguments("predict")

    # Load data
    rnaseq_df, rnaseq_train_df, rnaseq_test_df, input_dim, latent_dim,\
    hidden_dim, kappa = get_data()
    input_dim = rnaseq_df.shape[1]

    # Define a model
    combined_model, encoder, decoder = build_combined_model(opt.model_name,
            input_dim, opt.latent_dim, int(opt.depth), int(opt.hidden_dim), 
            float(opt.kappa), float(opt.lr), opt.noise, opt.sparsity)
    
    # Run the model
    if not opt.model_name in ["vae", "adage"]:
        perform_tSNE(rnaseq_df, "rnaseq")
    else:
        print("Encoding...")
        print("\ncheckpoint_combined=", checkpoint_combined)
        combined_model.load_weights(checkpoint_combined)
        encoded_df = encoder.predict_on_batch(rnaseq_df)
        if opt.model_name == "vae":
            encoded_df = encoded_df[2]
        encoded_df = pd.DataFrame(encoded_df, index=rnaseq_df.index)
        save_encoded_data(encoded_df, opt.model_name)
#       evaluate_reconstruction_fidelity(encoded_df)

    print("Performing tSNE...")
    perform_tSNE(encoded_df, opt.model_name)

