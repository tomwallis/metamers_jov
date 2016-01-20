# coding: utf-8

import os as _os
import glob as _glob
import pandas as _pd
import numpy as _np
from scipy.stats import beta as _beta


"""This file contains a number of common helper functions that will be called
by other stuff.


Tom Wallis wrote it.

"""


def raw_data_list(experiment_num):
    """ Return a list of filenames for the raw data files for this experiment.

    """

    top_dir = project_directory()
    data_dir = _os.path.join(top_dir,
                             'raw-data',
                             ('experiment-' + str(experiment_num)))
    # get a list of all the scenes in the directory:
    file_list = []
    wildcard = '*.csv'
    for file in _glob.glob(_os.path.join(data_dir, wildcard)):
        file_list.append(file)
    return(file_list)


def eye_data_list(experiment_num):
    """ Return a list of filenames for the eye data (.asc)
    files for this experiment.

    """

    top_dir = project_directory()
    data_dir = _os.path.join(top_dir,
                             'raw-data',
                             ('experiment-' + str(experiment_num)),
                             'eye_data_files')
    # get a list of all the scenes in the directory:
    file_list = []
    wildcard = '*.asc'
    for file in _glob.glob(_os.path.join(data_dir, wildcard)):
        file_list.append(file)
    return(file_list)


def project_directory():
    """Returns the full path of the top level directory for this project.

    Assuming that this function is being called from somewhere in the project.

    """

    orig_dir = _os.getcwd()
    parent = _os.path.split(orig_dir)[0]
    target = "metamers-natural-scenes"

    try:
        # go up directories, searching for target:
        while parent[-len(target):] != "metamers-natural-scenes":
            _os.chdir('..')
            parent = _os.path.split(_os.getcwd())[0]

        top_path = parent
        # set working directory back to original:
        _os.chdir(orig_dir)
    except:
        _os.chdir(orig_dir)
    return(top_path)


def munge_data(experiment_num):
    """Function to munge together the separate subject .csv files into
    one large file located in /results/. Also returns the data as a pandas
    dataframe.

    """
    top_dir = project_directory()
    out_dir = _os.path.join(top_dir,
                            'results',
                            ('experiment-' + str(experiment_num)))

    if not _os.path.exists(out_dir):
        _os.makedirs(out_dir)

    files = raw_data_list(experiment_num)

    dat = _pd.DataFrame()

    for file in files:
        this_dat = _pd.read_csv(file)
        dat = dat.append(this_dat, ignore_index=True)

    # create a "correct" column by comparing target and response:
    dat.target_loc = dat.target_loc.apply(str)
    dat.response = dat.response.apply(str)
    dat['correct'] = dat.target_loc == dat.response
    dat.correct = dat.correct.astype(_np.int)
    dat.loc[dat['response'] == 'na', 'correct'] = _np.nan

    if experiment_num < 4:
        # realised that the surround distance (width of donut) is *half*
        # of what I intended. Corrected here:
        def correct_surround(x):
            if x == 'blank':
                return(x)
            else:
                x_num = float(x)
                x_num /= 2.
                return(str(x_num))

        dat.surround = dat.surround.apply(correct_surround)

    # integrate parsed eye data, if it exists. Eye parsing
    # created by /analysis/eyetracker_data_parsing...
    eye_file = _os.path.join(out_dir, 'eye_data_parsed.csv')
    if _os.path.isfile(eye_file):
        eye_data = _pd.read_csv(eye_file, index_col=False)
        dat = dat.merge(eye_data,
                        how='left',
                        on=['subj', 'session', 'trial'])
    else:
        print('\nNo eye data found for this experiment\n')

    # integrate synthesis distances, if exists
    # created by /analysis/expt_X_synth_error...
    synth_file = _os.path.join(out_dir, 'synth_error.csv')
    if _os.path.isfile(synth_file):
        synth_data = _pd.read_csv(synth_file, index_col=False)
        melted = _pd.melt(synth_data,
                          id_vars='im_code',
                          var_name='cond',
                          value_name='synth_error')
        melted['synth_error'] = _np.log(melted['synth_error'])  # use log?
        dat = dat.merge(melted,
                        how='outer',
                        on=('im_code', 'cond'))

        # re-sort:
        dat = dat.sort(['subj', 'session', 'trial'])
    else:
        print('\nNo synth distance data found for this experiment\n')

    # save data:
    fname = _os.path.join(out_dir, 'all_data.csv')
    dat.to_csv(fname, index=False)
    return(dat)


def discard_invalid_trials(dat):
    """Function to discard trials with invalid eye data
    (blinks or missed fixations). Assumes data has been
    compiled with the munge data function, above.

    Note: I *include* any trials for which no eye data
    is available (e.g. where eyetracker broke). Given the
    low number of fixation breaks this seems reasonable.

    Parameters
    -----------

    dat:    a Pandas dataframe object containing munged data,
            including eye data.

    Returns
    ----------
    dat:    a Pandas dataframe object with invalid trials removed.
    """

    if 'time_invalid' in dat.columns:
        mask = ((dat['eye_invalid'] != 1) | (_np.isnan(dat['eye_invalid'])) &
                (dat['time_invalid'] != 1))
    else:
        mask = (dat['eye_invalid'] != 1) | (_np.isnan(dat['eye_invalid']))

    dat = dat.loc[mask, :]
    return(dat)


def discard_error_blocks(dat):
    """Function that discards the first two sessions of observer S6.
    At the start of the third testing session, S6 reported misunderstanding
    the instructions. She mistakenly believed that the target patch occurred
    in the same place for each unique image. Thus, she was trying to remember
    her last response to a given unique natural scene.

    Therefore I discard her first two sessions. You can see the effect of doing
    this on the fitted parameters by running `code/analysis/check_sg.ipynb`.

    :param dat:
        the dataframe of all observations from experiment 1.
    :returns:
        the dataframe with S6's first two sessions removed.

    """
    mask = (dat['subj'] == 'S6') & (dat['session'] < 3)
    dat = dat.loc[~mask, :]
    return dat


def experiment_1_data_ready_for_analysis(dat):
    """ Do all the things needed to the raw data from experiment 1
    for it to be ready for analysis.

    A wrapper function for other functions in data_helpers.

    :param dat:
        the dataframe of all observations from experiment 1.
    :returns:
        the dataframe with invalid eye trials and S6 blocks 1 & 2 removed.

    """
    dat = discard_invalid_trials(dat)
    dat = discard_error_blocks(dat)
    return dat


def experiment_9_data_ready_for_analysis(dat):
    """ Do all the things needed to the raw data from experiment 9
    for it to be ready for analysis.

    A wrapper function for other functions in data_helpers.

    :param dat:
        the dataframe of all observations from experiment 9.
    :returns:
        the dataframe with invalid eye trials and S9 removed.

    """
    dat = discard_invalid_trials(dat)
    dat = dat.loc[dat['subj'] != 'S9', :]

    ## Combine surround conditions into one variable.
    dat['surround'] = 'na'
    dat.loc[dat['surround_source'] == 'na', 'surround'] = 'blank'

    # conds:
    surr_source = ['nat', 'synth']

    for source in surr_source:
        mask = (dat['surround_source'] == source)
        dat.loc[mask, 'surround'] = source

    # scale -> patch_size
    dat['patch_size'] = _np.round(dat.loc[:, 'scale'] * 10., decimals=2)
    dat.drop('scale', axis=1, inplace=True)

    return dat


def experiment_10_data_ready_for_analysis(dat):
    """ Do all the things needed to the raw data from experiment 10
    for it to be ready for analysis.

    A wrapper function for other functions in data_helpers.

    :param dat:
        the dataframe of all observations from experiment 10.
    :returns:
        the dataframe with invalid eye trials and wrong timing removed.

    """
    dat = discard_invalid_trials(dat)

    # rename surround_cond to "surround":
    dat.rename(columns={'surround_cond': 'surround'},
               inplace=True)

    # scale -> patch_size
    dat['patch_size'] = _np.round(dat.loc[:, 'scale'] * 10., decimals=2)
    dat.drop('scale', axis=1, inplace=True)
    return dat


def experiment_11_data_ready_for_analysis(dat):
    """ Do all the things needed to the raw data from experiment 11
    for it to be ready for analysis.

    A wrapper function for other functions in data_helpers.

    :param dat:
        the dataframe of all observations from experiment 11.
    :returns:
        the dataframe with invalid eye trials removed.

    """
    # dat = discard_invalid_trials(dat)

    ## Combine surround conditions into one variable.
    dat['surround'] = 'na'
    dat.loc[dat['surround_source'] == 'na', 'surround'] = 'blank'

    # conds:
    surr_source = ['nat', 'synth']

    for source in surr_source:
        mask = (dat['surround_source'] == source)
        dat.loc[mask, 'surround'] = source

    # scale -> patch_size
    dat['patch_size'] = _np.round(dat.loc[:, 'scale'] * 10., decimals=2)
    dat.drop('scale', axis=1, inplace=True)

    return dat


def experiment_13_data_ready_for_analysis(dat):
    """ Do all the things needed to the raw data from experiment 13
    for it to be ready for analysis.

    A wrapper function for other functions in data_helpers.

    :param dat:
        the dataframe of all observations from experiment 13.
    :returns:
        the dataframe with invalid eye trials and wrong timing removed.

    """
    dat = discard_invalid_trials(dat)

    # rename surround_cond to "surround":
    dat.rename(columns={'surround_cond': 'surround'}, inplace=True)

    # size -> patch_size
    dat.loc[:, 'patch_size'] = _np.round(dat.loc[:, 'size'], decimals=2)
    dat.drop('size', axis=1, inplace=True)
    return dat


def binomial_binning(dat,
                     grouping_variables,
                     ci=.95,
                     rule_of_succession=True,
                     bernoulli_column='correct'):
    """Bin trials based on grouping variables, returning a new data frame
    with binomial outcome columns (successes, N_trials, plus propotion correct)
    rather than each row being a single trial.
    This data format will significantly speed up model fitting.

    :param dat:
        a pandas dataframe containing the data. Must have
        grouping_variables columns and also a column corresponding to
        bernoulli outcomes (0, 1).
    :param grouping_variables:
        a string or list of strings containing the column names
        to group over.
    :param ci:
        the percentage of the confidence intervals.
    :param rule_of_succession:
        if true, apply a rule-of-succession correction to the data by
        adding 1 success and one failure to the total number of trials.
        This is essentially a prior acknowledging the possibility of both
        successes and failures, and is used to correct for values with
        proportions of 0 or 1 (i.e. allow estimation of beta errors).
    :param bernoulli_column:
        A string naming the column of the dataframe corresponding to bernoulli
        trial outcome. Defaults to "correct".
    :returns:
        a new pandas dataframe where each row is a binomial trial.

    Example
    ----------
    res = binomial_binning(dat, ['subj', 'surround', 'scale'])


    """
    grouped = dat.groupby(grouping_variables, as_index=False)
    res = grouped[bernoulli_column].agg({'n_successes': _np.sum,
                                         'n_trials': _np.size})

    if rule_of_succession:
        res.loc[:, 'n_successes'] += 1
        res.loc[:, 'n_trials'] += 2

    # compute some additional values:
    res.loc[:, 'prop_corr'] = res.n_successes / res.n_trials

    # confidence intervals from a beta distribution:
    cis = _beta.interval(ci, res.n_successes, (res.n_trials-res.n_successes))
    res.loc[:, 'ci_min'] = cis[0]
    res.loc[:, 'ci_max'] = cis[1]
    res.loc[:, 'error_min'] = _np.abs(res['ci_min'].values -
                                      res['prop_corr'].values)
    res.loc[:, 'error_max'] = _np.abs(res['ci_max'].values -
                                      res['prop_corr'].values)

    return(res)
