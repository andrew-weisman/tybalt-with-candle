#!/usr/bin/env python

import os, re
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

from keras_tuner.engine.oracle import Objective

from options import parse_command_line_arguments
from models import build_combined_model, hypermodel_wrapper 
from models import RandomSearch, BayesianOptimization, Hyperband

def get_data(opt):
    # Load Gene Expression Data
    rnaseq_file = os.path.join("data", 'pancan_scaled_zeroone_rnaseq.tsv.gz')
    rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
    rnaseq_df.head(2)

    # Split 10% test set randomly
    test_set_percent = opt.val_split
    test_df   = rnaseq_df.sample(frac=test_set_percent)
    train_df  = np.array(rnaseq_df.drop(test_df.index))
    test_df   = np.array(test_df)

    opt.input_dim = rnaseq_df.shape[1]
    return (opt, rnaseq_df, train_df, test_df)

def save_training_performance(pd, args):
    history_df = pd.DataFrame(hist.history)
    history_df = history_df.assign(num_components=int(args.latent_dim))
    history_df = history_df.assign(learning_rate=float(args.learning_rate))
    history_df = history_df.assign(batch_size=int(args.batch_size))
    history_df = history_df.assign(epochs=int(args.num_epochs))
    history_df = history_df.assign(kappa=float(args.kappa))
    seed = int(np.random.randint(low=0, high=10000, size=1))
    np.random.seed(seed)
    history_df = history_df.assign(seed=seed)
    history_df = history_df.assign(depth=int(args.depth))
    history_df = history_df.assign(first_layer=int(args.first_layer))
    history_df.to_csv(args.output_filename, sep='\t')
    return history_df

if __name__ == '__main__':
    opt, checkpoint_combined, checkpoint_encoder, checkpoint_decoder = \
        parse_command_line_arguments("train", candle_params['tybalt_model_name'])

    opt, rnaseq_df, train_df, test_df = get_data(opt)

    if not opt.hpo:
        combined_model,encoder,decoder = build_combined_model(opt.model_name, 
            opt.input_dim, opt.latent_dim, int(opt.depth),int(opt.hidden_dim),
            float(opt.kappa), float(opt.lr), opt.noise, opt.sparsity) 
        encoder.summary(); decoder.summary(); combined_model.summary()
    else:
        pr_name = 'ktuner_' + opt.model_name + "_" + opt.hpo
        if re.search("random", opt.hpo):
            ktuner = RandomSearch(hypermodel_wrapper(opt), overwrite=True,          
                objective='val_loss', seed=1, project_name=pr_name, 
                max_trials=opt.max_trials, executions_per_trial=opt.ex_per_trial)
        elif re.search("bayesian", opt.hpo):
            ktuner = BayesianOptimization(hypermodel_wrapper(opt),overwrite=True,
                objective='val_loss', seed=1, project_name=pr_name,
                max_trials=opt.max_trials, executions_per_trial=opt.ex_per_trial)
        elif re.search("hyperband", opt.hpo):
            ktuner = Hyperband(hypermodel_wrapper(opt), overwrite=True,
                objective = Objective("val_loss",direction="min"),
                project_name=pr_name, max_epochs=int(opt.num_epochs))

    if not opt.hpo:
        checkpointer = ModelCheckpoint(filepath=checkpoint_combined, 
            verbose=opt.verbose, save_weights_only=True)
        combined_model.fit(train_df, train_df, shuffle=True,
            epochs=int(opt.num_epochs), batch_size=int(opt.batch_size),
            validation_data=(test_df, test_df), callbacks=[checkpointer])
        combined_model.save_weights(checkpoint_combined)
        encoder.save_weights(checkpoint_encoder)
        decoder.save_weights(checkpoint_decoder)
    else:
        ktuner.search(train_df, train_df, validation_data=(test_df,test_df),
                      use_multiprocessing=True)                                  
