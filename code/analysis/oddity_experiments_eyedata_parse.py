
# coding: utf-8
""" Must be run in Python 2 (cili can't do Python 3 as of May 2015).

From command line:

python oddity_experiments_eyedata_parse.py N

where N is the number of the experiment you want to parse.
Valid numbers are 9, 10 and 13.

"""

import sys
import numpy as np
import os
import pandas as pd
import data_helpers as helpers
from cili.util import load_eyelink_dataset
from cili.extract import extract_events
import yaml
import re
import psyutils as pu

# set up the figure path:
top_dir = helpers.project_directory()
fig_dir = os.path.join(top_dir, 'figures')
data_dir = os.path.join(top_dir, 'results')

# which experiment to do?
experiment_num = sys.argv[1]  # argv[0] is the script name.

valid_experiments = np.array([9, 10, 13])

if all(int(experiment_num) != valid_experiments):
    raise ValueError('Unknown experiment number!')
else:
    print('\n\nParsing experiment ' + experiment_num + '\n\n')


##### functions for working with files #####
def subj_session(filename):
    """ Extract subject name and session number from filename"""
    session = re.compile("_session_")
    file_ext = re.compile(".asc")
    start_subj = re.compile("sub_")

    # split the string after session with the file extension,
    # returning session num:
    session_num = file_ext.split(session.split(filename)[1])[0]

    # where does subject code start and end:
    start = start_subj.search(filename).end()
    end = session.search(filename).start()
    subj = filename[start:end]
    return(subj, session_num)


##### functions for analyses #####
def sample_rate(samples):
    rate = 1000. / (samples.index[1] - samples.index[0])
    return(rate)


def n_samples(samples, ms):
    # how many samples to take:
    rate = sample_rate(samples)
    n_samples = np.int64((ms / (1000./rate)))  # 200 ms target, plus some leeway
    return(n_samples)


def block_duration_mins(samples):
    return(((samples.index[-1] - samples.index[0]) / 1000.) / 60.)


def circ_dist(points, fix, params):
    # compute circular distance from fixation point, in degrees.
    rel_dist = points - fix
    rad = np.sqrt((rel_dist ** 2).sum(axis=1))

    # convert to degrees:
    rad /= params['pix_per_deg']
    return(rad)


def fix_dists(samples, tracked_eye, params):
    # function to add fixation distances to a dataframe of samples.

    # true fixation position:
    fix_pos = np.array([1920/2., 1200/2.])

    # extract all points from samples:
    if tracked_eye == 'LEFT':
        points = np.array([samples.loc[:, 'x_l'], samples.loc[:, 'y_l']]).T
    elif tracked_eye == 'RIGHT':
        points = np.array([samples.loc[:, 'x_r'], samples.loc[:, 'y_r']]).T
    else:
        print('Not sure which eye was tracked!')

    # compute for each sample:
    samples.loc[:, 'fix_dist_deg_nominal'] = circ_dist(points, fix_pos, params)

    return samples


def is_invalid(samples, tracked_eye, fixation_cutoff_deg=2.):
    """Function to check trial eye sample validity.
    Checks for both distance from fixation point (in degrees)
    and whether any samples are NaN (indicating a blink).

    NOTE: only checks one eye position (since most of
    my data is recorded left only). Will need to be modified for
    binocular data.

    """

    if tracked_eye == 'LEFT':
        eye = 'x_l'
    elif tracked_eye == 'RIGHT':
        eye = 'x_r'
    else:
        print('Which eye was tracked?')

    # check if samples are outside fixation cutoff or blink:
    if np.any(samples['fix_dist_deg_nominal'] > fixation_cutoff_deg) or \
       np.any(np.isnan(samples[eye])):
        res = 1
    else:
        res = 0
    return(res)


#### trial looper ####
def loop_trials(filename, df, exp_params,
                fixation_cutoff_deg=2.):
    """ Function containing the inner analysis loop (at trial level) """

    # load data file:
    samples, events = load_eyelink_dataset(filename)

    # subject and session number:
    subj, session = subj_session(filename)
    print('\nSubject ' + subj + ', Session ' + session)

    rate = sample_rate(samples)
    print('File sampled at ' + str(rate) + ' Hz')

    tracked_eye = events.START.eye.iloc[0]
    print('Tracking the ' + tracked_eye + ' eye')

    duration = block_duration_mins(samples)
    print('Block took ' + str(np.round(duration, decimals=1)) + ' mins')

    start_trials = events.MSG.loc[
        (events.MSG['label'] == 'start_trial:'), :]

    # loop over trials:
    for idx in range(len(start_trials)):
        # idx = 155
        trial_num = start_trials['content'].iloc[idx]
        valid_trial_timing = True
        valid_trial_em = True
        start_time = start_trials.index[idx]

        # trial should take 200+500+200+500+200 ms,
        # with 1200 ms before next trial.
        # i.e. trial duration should be ~ 1600.
        time_buffer = 2000
        trial_duration_tolerance = 1700
        interval_duration_tolerance = 250
        interval_duration_tolerance_lower = 150

        # extract end trial:
        end_trial = events.MSG.loc[(events.MSG['label'] == 'end_trial') &
                                   (events.MSG.index > start_time) &
                                   (events.MSG.index - start_time <
                                    time_buffer), :]

        if len(end_trial) == 1:
            trial_duration = end_trial.index[0] - start_time

            if trial_duration > trial_duration_tolerance:
                print('Trial ' + str(trial_num) +
                      ' too long! -- took ' + str(trial_duration))
                valid_trial_timing = False

            # extract the intervals for this trial:
            interval_events = list()
            interval_duration = list()
            interval_samples = list()
            interval_invalid = list()
            current_time = start_time

            for j in range(3):
                # check that we find a valid event interval:
                this_event = events.MSG.loc[(
                    events.MSG['label'] == 'end_interval:') &
                    (events.MSG['content'] == str(j+1)) &
                    (events.MSG.index > start_time) &
                    (events.MSG.index - start_time <
                     time_buffer), :]

                if len(this_event) == 0:
                    print('Failed to find a valid event \
                           for trial {}, \
                           interval {}'.format(str(trial_num), str(j+1)))
                    interval_events.append(this_event)
                    interval_duration.append(0)
                    valid_trial_timing = False
                else:
                    interval_events.append(this_event)
                    interval_duration.append(
                        interval_events[j].index[0] - current_time)
                    current_time += interval_duration[j] + 500
                    interval_events[j].loc[:, 'duration'] = interval_duration[j]

                    # check timing:
                    if interval_duration[j] > interval_duration_tolerance:
                        print('Trial ' + str(trial_num) +
                              ' interval ' + str(j+1) +
                              ' too long! -- took ' + str(interval_duration[j]))
                        valid_trial_timing = False
                    elif interval_duration[j] < interval_duration_tolerance_lower:
                        print('Trial ' + str(trial_num) +
                              ' interval ' + str(j+1) +
                              ' too short! -- took ' + str(interval_duration[j]))
                        valid_trial_timing = False
                    elif valid_trial_timing is True:
                        # now extract corresponding samples:
                        n_samp = n_samples(samples, ms=interval_duration[j])
                        # go backwards from interval end stamp:
                        interval_samples.append(extract_events(samples,
                                                               interval_events[j],
                                                               offset=-n_samp,
                                                               duration=n_samp))

                        # compute fixation distances:
                        interval_samples[j] = fix_dists(
                            interval_samples[j], tracked_eye, params)

                        interval_invalid.append(
                            is_invalid(interval_samples[j],
                                       tracked_eye=tracked_eye,
                                       fixation_cutoff_deg=fixation_cutoff_deg))

            if valid_trial_timing:
                # what's the overall sd of eye movement
                #  within the three intervals:
                eye_sd = np.mean(interval_samples[0].
                                 fix_dist_deg_nominal.std() +
                                 interval_samples[1].
                                 fix_dist_deg_nominal.std() +
                                 interval_samples[2].
                                 fix_dist_deg_nominal.std())

            if np.any(np.array(interval_invalid) == 1):
                valid_trial_em = False
                print('eye movements invalid!')

        elif len(end_trial) == 0:
            valid_trial_timing = False
            eye_sd = np.nan
            print('Trial ' + str(trial_num) + ' exceeded time buffer!')
        else:
            print('Found two or more end_trials!')

        # determine if overall trial is invalid (eye movements or timing)
        if valid_trial_em is False:
            eye_invalid = 1
        else:
            eye_invalid = 0

        if valid_trial_timing is False:
            time_invalid = 1
            eye_sd = np.nan
        else:
            time_invalid = 0

        df = df.append({'trial': trial_num,
                        'eye_invalid': eye_invalid,
                        'time_invalid': time_invalid,
                        'eye_sd_deg': eye_sd,
                        'subj': subj,
                        'session': int(session)}, ignore_index=True)

    mask = (df.subj == subj) & (df.session == int(session))
    n_eye_invalid = df.loc[mask, 'eye_invalid'].sum()
    n_time_invalid = df.loc[mask, 'time_invalid'].sum()
    pc_eye_invalid = np.round(100*(n_eye_invalid /
                                   len(df.loc[mask, :])), decimals=0)
    pc_time_invalid = np.round(100*(n_time_invalid /
                                    len(df.loc[mask, :])), decimals=0)
    print('Block had ' + str(n_eye_invalid) + ' invalid eye trials' +
          ' (' + str(pc_eye_invalid) + '%)')
    print('Block had ' + str(n_time_invalid) + ' invalid time trials' +
          ' (' + str(pc_time_invalid) + '%)')
    return(df)


""" Loop over oddity experiments

"""

# Analysis parameters:
# the true position of the fixation spot, in pixels.
fix_pos = np.array([1920/2., 1200/2.])
fixation_cutoff_deg = 2.  # cutoff for ruling fixation broken, in degrees.

params_file = 'generation_params_exp_' + str(experiment_num) + '.yaml'
expt_params = os.path.join(top_dir, 'code', 'stimuli', params_file)
with open(expt_params, 'r') as f:
    params = yaml.load(f)

files = helpers.eye_data_list(experiment_num)
df = pd.DataFrame()  # empty dataframe will be filled with eye data.

"""
Loop over trials within this experiment
"""
for f in files:
    df = loop_trials(f, df, params, fixation_cutoff_deg=fixation_cutoff_deg)

print(df.groupby(['subj']).eye_invalid.sum())
print(df.groupby(['subj']).time_invalid.sum())

# save to csv:
out_name = os.path.join(data_dir, 'experiment-' + str(experiment_num),
                        'eye_data_parsed.csv')
df.to_csv(out_name, index=False)

pu.files.session_info()
