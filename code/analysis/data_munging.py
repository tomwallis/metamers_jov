# coding: utf-8

import data_helpers as helpers

""" Script to do data munging for all experiments.


Tom Wallis wrote it.
"""

# set up the figure path:
top_dir = helpers.project_directory()

for i in range(14):
    i += 1
    if i is not 7:
        print('Doing experiment ' + str(i))
        # munge data autosaves the data:
        dat = helpers.munge_data(i)

# pu.files.session_info()
