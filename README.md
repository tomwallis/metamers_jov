# Documentation and overview of materials for Wallis, Bethge & Wichmann

**Tom Wallis, January 2016.**

The following repository provides code and data used in the forthcoming paper "Testing models of peripheral encoding using metamerism in an oddity paradigm" by Wallis, Bethge & Wichmann. I understand that much of this process could be improved, but this is where I currently am for reproducibility in my own workflow. Open science is a learning curve!

## Citation

If you use or modify this code or the data in an academic publication, please cite the paper (please update the citation accordingly once the manuscript is out):

Wallis, T.S.A., Bethge, M. & Wichmann, F. A. (2016). Testing models of peripheral encoding using metamerism in an oddity paradigm. *Journal of Vision 16*(2), 4.


## Code
The code in this git repository is also available with DOI from Zenodo [here](http://doi.org/10.5281/zenodo.34218).

## Data and stimuli

The data and stimuli are available [here](http://doi.org/10.5281/zenodo.32784).

## Directory structure

This git repository contains all code for the project. You can pair this code with the other materials (stimuli and data). To ensure the paths all work correctly, place all the materials into a directory named `metamers-natural-scenes`. Within this top-level folder, the subdirectories should be:

* `/code/` contains all code for running experiments and analyses (what's available on this git repo)
    - `/analysis` = data analysis code
    - `/experiment` = code to run experiments in psychtoolbox
    - `/stimuli` = code to generate stimuli

The rest of the materials are available from [here](http://doi.org/10.5281/zenodo.32784). They should be placed in the top level directory.

* `/raw-data/` contains all raw data (individual session files and eyetracking), subfolder for each experiment. Eyetracking data is in `/raw-data/experiment-N/eye_data_files/`.
* `/results` contains aggregated results files used for analysis and some analysis output. I have not provided the MCMC samples I report in the paper because they are stored as pickle files in python, which constitute a security risk for sharing. They are available upon request (or just re-run the sampling yourself).
* `/stimuli` contains the images shown to observers in the experiments.



## Important note about experiment labelling

I have run a number of related experiments that were not reported in the paper.These experiments tried and failed to make discrimination performance worse (i.e. to generate convincing metamers with the Portilla and Simoncelli texture syntheses), and should be considered unreported pilots. 

Consequently, the data reported in the paper are labelled differently in the code. Sorry if this is confusing; I decided to keep it to maintain code compatibility.

  * Paper "Experiment 1" = code "experiment 13"
  * Paper "Experiment 2" = code "experiment 9"
  * Paper blur supplement (Figure 11) = code "experiment 14"
  * Paper appendix data ("scene scale vs patch size") = code experiment 10

In all documentation hereafter, the experiments are referred to by their numbers *in the code*.


## Software used

* Texture stimulus generation: 
    - Matlab 2014a
* Experiment presentation: 
    - Matlab 2013b (experimental presentation)
    - Psychtoolbox (version 3.0.12)
    - iShow (internal library, available [here](http://dx.doi.org/10.5281/zenodo.34217)
))
* Stimulus generation:
    - Python 3.3.5 (Anaconda 2.0.1) --- see log files in `code/stimuli` for package requirements. You will need a package called `psyutils`; it is available [here](https://github.com/tomwallis/PsyUtils).


## Experiment documentation

### Experiment 13 (paper Experiment 1; discriminating Gaussian blur)

#### Stimulus generation 

The files to reproduce the stimuli are in `/code/stimuli`. To generate stimuli, run `generate_stimuli_13.py`. You will need to change the `source_path` (line 32) to point to the Judd database directory containing [ALLSTIMULI](https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip) and [ALLFIXATIONMAPS](https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip) you want to use locally. This script reads in parameters from `generation_params_exp_13.yaml` and writes images out to `/stimuli/experiment-13/final_ims`.

#### Experiment

The experiment script is `/code/experiment/metamers_experiment_13.m`. It will output new raw data to `/raw-data/experiment-13/`. It requires an appropriately configured Psychtoolbox installation to run. Because I use MATLAB's (new! amazing! Just like R only way clunkier!) Tables data type, you will need a recent version of Matlab.

#### Data analysis

All data analysis scripts are located in `/code/analysis/`. To compile raw data files (located in `/raw-data/experiment-13`) into a single summary file (located in `/results/experiment-13`), run `data_munging.py`. This shouldn't be necessary since I've provided the compiled csv file in this repository.

Eye data was parsed using the script `oddity_experiments_eyedata_parse.py`. I'm not providing raw eye data here (too large; available upon request, or let me know about a better free hosting solution) but I've included the script for reference.

To fit the multilevel model using Stan, run `fit_tanh_model_expt_13.py` (assuming you've correctly installed pyStan). The model itself is specified in `oddity_3AFC_tanh3_full.stan`. I didn't include my model fits in this repository (too large; available upon request, or let me know about a better free hosting solution).

To generate the plots reported in the paper, run `paper_plots_experiment_13.ipynb`. To see the Bayesian ANOVA results, see the JASP file `expt_13_anova.jasp`.

To reproduce the analysis in Figure 6 (the comparison of spectral content), first run `geisler_perry_blurring.m` under Matlab with [Jeff Perry's SVIS toolbox](https://github.com/jeffsp/svis) correctly installed. Then run `spectral_content_analysis_2.ipynb` to make the figures.



### Experiment 9 (paper Experiment 2; discriminating PS textures)

#### Stimulus generation 

The files to reproduce the stimuli are in `/code/stimuli`. To generate stimuli, run 

1. `generate_stimuli_9a.py`. You will need to change the `source_path` (line 28) to the Judd database directory as above. This script reads parameters from `generation_parameters_exp_9.yaml` and writes images to three directories in `/stimuli/experiment-9/`.
2. Now generate PS textures from the source images generated in step 1. I separated this into three separate scripts to make things go faster (distributing source patches across multiple CPU cores). Using a MATLAB installation with the [PS texturesynth toolbox](http://www.cns.nyu.edu/lcv/texture/) appropriately installed, run `p_s_generation_expt_9_inner.m`, `p_s_generation_expt_9_middle.m` and `p_s_generation_expt_9_outer.m`, changing numbers of cores and source paths appropriately.
3. Run `generate_stimuli_9b.py`, which assembles the final patches for the experiment and writes them to `/stimuli/experiment-9/final_ims`.

Phew! You're done.

#### Experiment

The experiment script is `/code/experiment/metamers_experiment_9.m`. It will output new raw data to `/raw-data/experiment-9/`. It requires an appropriately configured Psychtoolbox installation to run. Because I use MATLAB's (new! amazing! Just like R only way clunkier!) Tables data type, you will need a recent version of Matlab.

#### Data analysis

All data analysis scripts are located in `/code/analysis/`. To compile raw data files (located in `/raw-data/experiment-9`) into a single summary file (located in `/results/experiment-9`), run `data_munging.py`. This shouldn't be necessary since I've provided the compiled csv file in this repository.

Eye data was parsed using the script `oddity_experiments_eyedata_parse.py`. I'm not providing raw eye data here (too large; available upon request, or let me know about a better free hosting solution) but I've included the script for reference.

To fit the multilevel model using Stan, run `fit_tanh_model_expt_9.py` (assuming you've correctly installed pyStan). The model itself is specified in `oddity_3AFC_tanh3_full.stan`. I didn't include my model fits in this repository (too large; available upon request, or let me know about a better free hosting solution).

To generate the plots reported in the paper, run `paper_plots_experiment_9.ipynb`. To see the Bayesian ANOVA results, see the JASP file `expt_13_anova.jasp`.

### Experiment 14 (blur supplementary experiment; paper Figure 11)

#### Stimulus generation 

Generate stimuli with `generate_stimuli_14.py`. This script reads in parameters from `generation_params_exp_14.yaml` and writes images out to `/stimuli/experiment-14/final_ims`.

#### Experiment

The experiment script is `/code/experiment/metamers_experiment_14.m`. It will output new raw data to `/raw-data/experiment-14/`. It requires an appropriately configured Psychtoolbox installation to run. Because I use MATLAB's (new! amazing! Just like R only way clunkier!) Tables data type, you will need a recent version of Matlab.

#### Data analysis

As above, compile raw files using `data_munging.py`. To generate the plots reported in the paper, run `paper_plots_experiment_14.ipynb` and `expt_14_spectral_content_analysis.ipynb`. 


### Experiment 10 (paper Appendix 6.3; downsampling instead of cropping)

The procedure here is almost identical to that for Experiment 9, above.

#### Stimulus generation 

The files to reproduce the stimuli are in `/code/stimuli`. To generate stimuli, run 

1. `generate_stimuli_10a.py`. 
2. Generate PS textures: `p_s_generation_expt_10_batch_1.m` ... `p_s_generation_expt_10_batch_4.m`.
3. `generate_stimuli_10b.py`.


#### Experiment

The experiment script is `/code/experiment/metamers_experiment_10.m`. 

#### Data analysis

Having run `data_munging.py` and `oddity_experiments_eyedata_parse.py` above,

1. `fit_tanh_model_expt_10.py`
2. `plot_tanh_fits_experiment_10.ipynb`


