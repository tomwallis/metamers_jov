# coding: utf-8

import os
import pandas as pd
import numpy as np
import pystan
import data_helpers as helpers
import patsy
import psyutils as pu
import pickle
from numpy.random import RandomState

"""
Fit the multilevel model to all experiment data.

For demos of what this script is doing,
see `expt_9_stan_model_development.ipynb`.
"""

"""
Set up parameters
"""

experiment_num = 13

# model parameter upper bounds:
alpha_upper = 5.
beta_upper = 10.
scale_upper = 16.

# Stan params:
# assumes that stan model file is in the current working directory:
model_file = 'oddity_3AFC_tanh3_full.stan'
model_name = 'tanh3_full'

chains = 4
# sampling params for *each* chain:
iterations = 10000
warmup = iterations // 2
thin = 5
seed = 4242

# plot dataframe params:
n_subset = 100  # the number of samples to use for curve plotting.
rng = RandomState(1020931)

# set up the figure path:
top_dir = helpers.project_directory()
results_dir = os.path.join(top_dir, 'results',
                           ('experiment-' + str(experiment_num)))

out_file = os.path.join(results_dir,
                        ('expt_' + str(experiment_num) +
                         '_' + model_name + '_model_fit.pkl'))
print(out_file)
# whether to apply rule of succession correction:
rule_of_succession = True

"""
helper functions
"""


def subj_to_numeric(dat):
    # add a numeric code for each subject
    subj_codes = np.unique(dat.subj)
    for i, subj in enumerate(subj_codes):
        dat.loc[dat['subj'] == subj, 'subj_num'] = i+1
    dat['subj_num'] = dat['subj_num'].astype(np.int)

    return dat

"""
Load and munge data
"""
dat = pd.read_csv(os.path.join(results_dir, 'all_data.csv'), index_col=False)

if experiment_num == 13:
    dat = helpers.experiment_13_data_ready_for_analysis(dat)
else:
    raise ValueError('specify a data munging function!')

# create binomial groupings
dat = helpers.binomial_binning(dat,
                               ['subj', 'patch_size', 'surround', 'blur_sigma'],
                               rule_of_succession=rule_of_succession)

# design matrix ignores subject:
X = patsy.dmatrix('~ C(patch_size, Sum) * C(surround, Sum)', dat)

dat = subj_to_numeric(dat)

# grid for model predictions:
s_pred = np.linspace(0, 16, num=99)
pred_dat = pu.misc.expand_grid({'s_pred': s_pred,
                                'patch_size': np.unique(dat['patch_size']),
                                'surround': np.unique(dat['surround']),
                                'subj': np.unique(dat['subj'])})
pred_dat = subj_to_numeric(pred_dat)
pred_X = patsy.dmatrix('~ C(patch_size, Sum) * C(surround, Sum)', pred_dat)
pred_ss = pred_dat['subj_num']

# grid for calculating model params:
param_dat = pu.misc.expand_grid({'patch_size': np.unique(dat['patch_size']),
                                 'surround': np.unique(dat['surround']),
                                 'subj': np.unique(dat['subj'])})
param_dat = subj_to_numeric(param_dat)
param_X = patsy.dmatrix('~ C(patch_size, Sum) * C(surround, Sum)', param_dat)
param_ss = param_dat['subj_num']

# grid for calculating population params:
pop_param_dat = pu.misc.expand_grid({'patch_size': np.unique(dat['patch_size']),
                                     'surround': np.unique(dat['surround'])})
pop_param_X = patsy.dmatrix('~ C(patch_size, Sum) * C(surround, Sum)',
                            pop_param_dat)

# set up data for Stan:
# remove intercept from design matrix:
design_mat = X[:, 1:]
pred_mat = pred_X[:, 1:]
param_mat = param_X[:, 1:]
pop_param_mat = pop_param_X[:, 1:]

# dict of stan data:
stan_dat = {'N': len(dat),
            'n': dat['n_trials'].values.astype(np.int),
            'r': dat['n_successes'].values.astype(np.int),
            's': dat['blur_sigma'].values,
            'S': len(np.unique(dat['subj'])),
            'ss': dat['subj_num'].values,
            'D': design_mat.shape[1],
            'X': design_mat,
            'alpha_upper': alpha_upper,
            'beta_upper': beta_upper,
            'scale_upper': scale_upper,
            'N_pred': len(pred_dat),
            's_pred': pred_dat['s_pred'].values,
            'X_pred': pred_mat,
            'ss_pred': pred_ss.values,
            'N_param_pred': len(param_mat),
            'X_params_pred': param_mat,
            'ss_params_pred': param_ss.values,
            'N_pop_param_pred': len(pop_param_mat),
            'X_pop_params_pred': pop_param_mat}

"""
Compile and fit Stan model
"""
# Compile Stan model (necessary separate step for pickling):
stan_model = pystan.StanModel(file=model_file,
                              model_name=model_name)


# sample from model:
print('SAMPLING FROM MODEL')

fit = pystan.stan(file=model_file,
                  model_name=model_name,
                  data=stan_dat,
                  iter=iterations,
                  warmup=warmup,
                  thin=thin,
                  chains=chains,
                  seed=seed)

"""
Extract parameters, join with other dataframes for easier plotting
"""

print('EXTRACTING PREDICTIONS')

""" Yhat (curve) values"""

preds = fit.extract('yhat')
n_samples = preds['yhat'].shape[0]

# subsample the preds yhat matrix to reduce overplotting:
idx = rng.randint(preds['yhat'].shape[0], size=n_subset)
yhat = preds['yhat'][idx, :]

pred_frame = pd.DataFrame()
for i in range(n_subset):
    # duplicate pred_dat for each sample:
    pred_dat['yhat'] = yhat[i, :]
    pred_dat['sample'] = i
    pred_frame = pred_frame.append(pred_dat)

pred_frame.rename(columns={'s_pred': 'blur_sigma'}, inplace=True)

plot_df = dat.append(pred_frame, ignore_index=True)

""" Parameter distributions in model scales for each subject, condition """
param_preds = fit.extract(['alpha_pred',
                           'beta_pred',
                           'scale_pred',
                           'prior_alpha_pred',
                           'prior_beta_pred',
                           'prior_scale_pred'])
n_samples = param_preds['alpha_pred'].shape[0]

param_dat.drop('subj_num', axis=1, inplace=True)

# arrange predictions into a large dataframe for plotting:
param_frame = pd.DataFrame()

for i in range(n_samples):
    param_dat['sample'] = i

    param_dat['alpha'] = param_preds['alpha_pred'][i, :]
    param_dat['beta'] = param_preds['beta_pred'][i, :]
    param_dat['critical_scale'] = param_preds['scale_pred'][i, :]

    param_dat['prior_alpha'] = param_preds['prior_alpha_pred'][i, :]
    param_dat['prior_beta'] = param_preds['prior_beta_pred'][i, :]
    param_dat['prior_critical_scale'] = param_preds['prior_scale_pred'][i, :]

    param_frame = param_frame.append(param_dat)


""" Population parameter distributions for each condition """
param_preds = fit.extract(['mu_alpha_pred',
                           'mu_beta_pred',
                           'mu_scale_pred',
                           'prior_mu_alpha_pred',
                           'prior_mu_beta_pred',
                           'prior_mu_scale_pred'])
n_samples = param_preds['mu_alpha_pred'].shape[0]

# arrange predictions into a large dataframe for plotting:
pop_param_frame = pd.DataFrame()

for i in range(n_samples):
    pop_param_dat['sample'] = i

    pop_param_dat['mu_alpha'] = param_preds['mu_alpha_pred'][i, :]
    pop_param_dat['mu_beta'] = param_preds['mu_beta_pred'][i, :]
    pop_param_dat['mu_critical_scale'] = param_preds['mu_scale_pred'][i, :]

    pop_param_dat['prior_mu_alpha'] = param_preds['prior_mu_alpha_pred'][i, :]
    pop_param_dat['prior_mu_beta'] = param_preds['prior_mu_beta_pred'][i, :]
    pop_param_dat['prior_mu_critical_scale'] = \
        param_preds['prior_mu_scale_pred'][i, :]

    pop_param_frame = pop_param_frame.append(pop_param_dat)


"""
Pickle important objects.
"""

print('Saving output...')

with open(out_file, 'wb') as f:
    pickle.dump([stan_model,
                 fit,
                 plot_df,
                 param_frame,
                 pop_param_frame], f)

pu.files.session_info()
