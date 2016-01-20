# Fit a three parameter (critical scale, gain and slope) tanh model
# to data from all observers under all conditions, using orthogonal
# contrasts.
# Tom Wallis wrote it. tsawallis at gmail dot com

functions {
    real dprime_fun(real s, real alpha, real beta, real scale){
        # return a dprime value given a scale level (s) and
        # the parameters of the model.
        real res;
        if (s <= scale){
           res <- 0.0;
           return res;
        }

        else if (s > scale){
            return alpha * tanh(beta * (s - scale));
        }

        else
            return 100.0;
    }

    # function to return pc for a given dprime, using Weibull approx:
    real pc_fun(real x){
        real m;
        real scale;
        real shape;
        real raw_pc;
        real pc;

        m <- 3.0;  // 3-alternative oddity.
        scale <- 2.84268839; // fit externally by minimising sse.
        shape <- 1.86565947;
        raw_pc <- 1.0 - exp(- pow((x / scale), shape));
        pc <- (1.0 / m) + (1.0 - (1/m)) * raw_pc;
        return pc;
    }

    real rescale_param(real upper, real param_raw, vector X, vector offsets) {
        return upper * inv_logit(param_raw + dot_product(X, offsets));
    }

}

data {
  # the dataset:
  int<lower=0> N; // number of rows of data
  int<lower=0> n[N]; // number of trials
  int<lower=0> r[N]; // number of successes in n trials
  vector<lower=0>[N] s; // vector of scales.
  int<lower=1> S;  // number of subjects
  int<lower=1, upper=S> ss[N];  // vector associating each row with a subject.

  # coding for conditions:
  int<lower=0> D; // number of coefficients to estimate.
  vector[D] X[N]; // design matrix EXCLUDING intercept.

  # upper bounds on params:
  real alpha_upper;
  real beta_upper;
  real scale_upper;

  # prediction values for generating yhat:
  int<lower=0> N_pred;  // length of predicted quantities.
  vector<lower=0>[N_pred] s_pred;  // scale factors to predict.
  vector[D] X_pred[N_pred];  // design matrix for predictions, excluding intercept.
  int<lower=1, upper=S> ss_pred[N_pred];  // subject id for each prediction row.

  # design mat for generating parameter posteriors for each subject, condition:
  int<lower=0> N_param_pred;  // length of parameter mat.
  vector[D] X_params_pred[N_param_pred];
  int<lower=1, upper=S> ss_params_pred[N_param_pred];  // subject id for each prediction row.

  # design mat for population-level predictions:
  int<lower=0> N_pop_param_pred;
  vector[D] X_pop_params_pred[N_pop_param_pred];
}

transformed data {

}

parameters {

  # these "raw" params are in the scale of the linear
  # predictor: they get passed through an inverse
  # logit to ensure their sum is within the parameter
  # bounds. I work out the params on a more intuitive scale
  # in the generated params block.

  # population-level params:

  # population means:
  real<lower=-10, upper=10> mu_alpha_raw;
  real<lower=-10, upper=10> mu_beta_raw;
  real<lower=-10, upper=10> mu_scale_raw;

  vector<lower=-10, upper=10>[D] mu_alpha_offsets_raw;
  vector<lower=-10, upper=10>[D] mu_beta_offsets_raw;
  vector<lower=-10, upper=10>[D] mu_scale_offsets_raw;

  # population sds:
  real<lower=0, upper=10> sigma_alpha_raw;
  real<lower=0, upper=10> sigma_beta_raw;
  real<lower=0, upper=10> sigma_scale_raw;

  vector<lower=0, upper=10>[D] sigma_alpha_offsets_raw;
  vector<lower=0, upper=10>[D] sigma_beta_offsets_raw;
  vector<lower=0, upper=10>[D] sigma_scale_offsets_raw;

  # subject level params as errors from population:
  real<lower=-10, upper=10> e_alpha_raw[S];
  real<lower=-10, upper=10> e_beta_raw[S];
  real<lower=-10, upper=10> e_scale_raw[S];

  # offset params:
  vector<lower=-10, upper=10>[D] e_alpha_offsets_raw[S];
  vector<lower=-10, upper=10>[D] e_beta_offsets_raw[S];
  vector<lower=-10, upper=10>[D] e_scale_offsets_raw[S];


  # # collect samples from the prior:
  # population means:
  real<lower=-10, upper=10> prior_mu_alpha_raw;
  real<lower=-10, upper=10> prior_mu_beta_raw;
  real<lower=-10, upper=10> prior_mu_scale_raw;

  vector<lower=-10, upper=10>[D] prior_mu_alpha_offsets_raw;
  vector<lower=-10, upper=10>[D] prior_mu_beta_offsets_raw;
  vector<lower=-10, upper=10>[D] prior_mu_scale_offsets_raw;

  # population sds:
  real<lower=0, upper=10> prior_sigma_alpha_raw;
  real<lower=0, upper=10> prior_sigma_beta_raw;
  real<lower=0, upper=10> prior_sigma_scale_raw;

  vector<lower=0, upper=10>[D] prior_sigma_alpha_offsets_raw;
  vector<lower=0, upper=10>[D] prior_sigma_beta_offsets_raw;
  vector<lower=0, upper=10>[D] prior_sigma_scale_offsets_raw;

  # subject level params as errors from population:
  real<lower=-10, upper=10> prior_e_alpha_raw[S];
  real<lower=-10, upper=10> prior_e_beta_raw[S];
  real<lower=-10, upper=10> prior_e_scale_raw[S];

  # offset params:
  vector<lower=-10, upper=10>[D] prior_e_alpha_offsets_raw[S];
  vector<lower=-10, upper=10>[D] prior_e_beta_offsets_raw[S];
  vector<lower=-10, upper=10>[D] prior_e_scale_offsets_raw[S];

}

transformed parameters {
  # subject coefficients shifted and scaled by population:
  real alpha_raw[S];  // gain
  real beta_raw[S];  // slope
  real scale_raw[S]; // critical scale

  # offset params (as errors from pop mu and sigma):
  vector[D] alpha_offsets_raw[S];
  vector[D] beta_offsets_raw[S];
  vector[D] scale_offsets_raw[S];

  # priors:
  real prior_alpha_raw[S];  // gain
  real prior_beta_raw[S];  // slope
  real prior_scale_raw[S]; // critical scale

  vector[D] prior_alpha_offsets_raw[S];
  vector[D] prior_beta_offsets_raw[S];
  vector[D] prior_scale_offsets_raw[S];

  for (i in 1:S) {
    alpha_raw[i] <- mu_alpha_raw + e_alpha_raw[i] .* sigma_alpha_raw;
    beta_raw[i] <- mu_beta_raw + e_beta_raw[i] .* sigma_beta_raw;
    scale_raw[i] <- mu_scale_raw + e_scale_raw[i] .* sigma_scale_raw;

    alpha_offsets_raw[i] <- mu_alpha_offsets_raw + e_alpha_offsets_raw[i] .* sigma_alpha_offsets_raw;
    beta_offsets_raw[i] <- mu_beta_offsets_raw + e_beta_offsets_raw[i] .* sigma_beta_offsets_raw;
    scale_offsets_raw[i] <- mu_scale_offsets_raw + e_scale_offsets_raw[i] .* sigma_scale_offsets_raw;

    prior_alpha_raw[i] <- prior_mu_alpha_raw + prior_e_alpha_raw[i] .* prior_sigma_alpha_raw;
    prior_beta_raw[i] <- prior_mu_beta_raw + prior_e_beta_raw[i] .* prior_sigma_beta_raw;
    prior_scale_raw[i] <- prior_mu_scale_raw + prior_e_scale_raw[i] .* prior_sigma_scale_raw;

    prior_alpha_offsets_raw[i] <- prior_mu_alpha_offsets_raw + prior_e_alpha_offsets_raw[i] .* prior_sigma_alpha_offsets_raw;
    prior_beta_offsets_raw[i] <- prior_mu_beta_offsets_raw + prior_e_beta_offsets_raw[i] .* prior_sigma_beta_offsets_raw;
    prior_scale_offsets_raw[i] <- prior_mu_scale_offsets_raw + prior_e_scale_offsets_raw[i] .* prior_sigma_scale_offsets_raw;
  }

}

model {
    # Priors (linear predictor scale):
    mu_alpha_raw ~ normal(0, 0.5);
    mu_beta_raw ~ normal(0, 0.5);
    mu_scale_raw ~ normal(0, 0.5);

    mu_alpha_offsets_raw ~ normal(0, 0.5);
    mu_beta_offsets_raw ~ normal(0, 0.5);
    mu_scale_offsets_raw ~ normal(0, 0.5);

    sigma_alpha_raw ~ cauchy(0, 1);
    sigma_beta_raw ~ cauchy(0, 1);
    sigma_scale_raw ~ cauchy(0, 1);

    sigma_alpha_offsets_raw ~ cauchy(0, 1);
    sigma_beta_offsets_raw ~ cauchy(0, 1);
    sigma_scale_offsets_raw ~ cauchy(0, 1);

    # sample from priors for plotting:
    prior_mu_alpha_raw ~ normal(0, 0.5);
    prior_mu_beta_raw ~ normal(0, 0.5);
    prior_mu_scale_raw ~ normal(0, 0.5);

    prior_mu_alpha_offsets_raw ~ normal(0, 0.5);
    prior_mu_beta_offsets_raw ~ normal(0, 0.5);
    prior_mu_scale_offsets_raw ~ normal(0, 0.5);

    prior_sigma_alpha_raw ~ cauchy(0, 1);
    prior_sigma_beta_raw ~ cauchy(0, 1);
    prior_sigma_scale_raw ~ cauchy(0, 1);

    prior_sigma_alpha_offsets_raw ~ cauchy(0, 1);
    prior_sigma_beta_offsets_raw ~ cauchy(0, 1);
    prior_sigma_scale_offsets_raw ~ cauchy(0, 1);

    for (i in 1:S){
        e_alpha_raw[i] ~ normal(0, 1);
        e_beta_raw[i] ~ normal(0, 1);
        e_scale_raw[i] ~ normal(0, 1);

        e_alpha_offsets_raw[i] ~ normal(0, 1);
        e_beta_offsets_raw[i] ~ normal(0, 1);
        e_scale_offsets_raw[i] ~ normal(0, 1);

        prior_e_alpha_raw[i] ~ normal(0, 1);
        prior_e_beta_raw[i] ~ normal(0, 1);
        prior_e_scale_raw[i] ~ normal(0, 1);

        prior_e_alpha_offsets_raw[i] ~ normal(0, 1);
        prior_e_beta_offsets_raw[i] ~ normal(0, 1);
        prior_e_scale_offsets_raw[i] ~ normal(0, 1);
    }

    # loop over data rows:
    for (i in 1:N) {
        real theta;  // predicted prob.
        real dprime; // the dprime returned.
        real this_alpha;
        real this_beta;
        real this_scale;

        # rescale raw alpha and offsets to be in parameter range (0--upper)
        this_alpha <- rescale_param(alpha_upper, alpha_raw[ss[i]], X[i], alpha_offsets_raw[ss[i]]);
        this_beta <- rescale_param(beta_upper, beta_raw[ss[i]], X[i], beta_offsets_raw[ss[i]]);
        this_scale <- rescale_param(scale_upper, scale_raw[ss[i]], X[i], scale_offsets_raw[ss[i]]);

        dprime <- dprime_fun(s[i], this_alpha, this_beta, this_scale);
        theta <- pc_fun(dprime);
        r[i] ~ binomial(n[i], theta);
    }
}

generated quantities {
    vector<lower=0, upper=1>[N_pred] yhat;

    vector<lower=0, upper=alpha_upper>[N_param_pred] alpha_pred;
    vector<lower=0, upper=beta_upper>[N_param_pred] beta_pred;
    vector<lower=0, upper=scale_upper>[N_param_pred] scale_pred;

    vector<lower=0, upper=alpha_upper>[N_param_pred] prior_alpha_pred;
    vector<lower=0, upper=beta_upper>[N_param_pred] prior_beta_pred;
    vector<lower=0, upper=scale_upper>[N_param_pred] prior_scale_pred;

    vector<lower=0, upper=alpha_upper>[N_pop_param_pred] mu_alpha_pred;
    vector<lower=0, upper=beta_upper>[N_pop_param_pred] mu_beta_pred;
    vector<lower=0, upper=scale_upper>[N_pop_param_pred] mu_scale_pred;

    vector<lower=0, upper=alpha_upper>[N_pop_param_pred] prior_mu_alpha_pred;
    vector<lower=0, upper=beta_upper>[N_pop_param_pred] prior_mu_beta_pred;
    vector<lower=0, upper=scale_upper>[N_pop_param_pred] prior_mu_scale_pred;

    for (i in 1:N_pred) {
        real dprime;
        real this_alpha;
        real this_beta;
        real this_scale;

        this_alpha <- rescale_param(alpha_upper, alpha_raw[ss_pred[i]], X_pred[i], alpha_offsets_raw[ss_pred[i]]);
        this_beta <- rescale_param(beta_upper, beta_raw[ss_pred[i]], X_pred[i], beta_offsets_raw[ss_pred[i]]);
        this_scale <- rescale_param(scale_upper, scale_raw[ss_pred[i]], X_pred[i], scale_offsets_raw[ss_pred[i]]);

        dprime <- dprime_fun(s_pred[i], this_alpha, this_beta, this_scale);
        yhat[i] <- pc_fun(dprime);
    }

    for (i in 1:N_param_pred) {
        alpha_pred[i] <- rescale_param(alpha_upper, alpha_raw[ss_params_pred[i]], X_params_pred[i], alpha_offsets_raw[ss_params_pred[i]]);
        beta_pred[i] <- rescale_param(beta_upper, beta_raw[ss_params_pred[i]], X_params_pred[i], beta_offsets_raw[ss_params_pred[i]]);
        scale_pred[i] <- rescale_param(scale_upper, scale_raw[ss_params_pred[i]], X_params_pred[i], scale_offsets_raw[ss_params_pred[i]]);

        # sample from the prior:
        prior_alpha_pred[i] <- rescale_param(alpha_upper, prior_alpha_raw[ss_params_pred[i]], X_params_pred[i], prior_alpha_offsets_raw[ss_params_pred[i]]);
        prior_beta_pred[i] <- rescale_param(beta_upper, prior_beta_raw[ss_params_pred[i]], X_params_pred[i], prior_beta_offsets_raw[ss_params_pred[i]]);
        prior_scale_pred[i] <- rescale_param(scale_upper, prior_scale_raw[ss_params_pred[i]], X_params_pred[i], prior_scale_offsets_raw[ss_params_pred[i]]);
    }

    # population level parameters:
    for (i in 1:N_pop_param_pred) {
        mu_alpha_pred[i] <- rescale_param(alpha_upper, mu_alpha_raw, X_pop_params_pred[i], mu_alpha_offsets_raw);
        mu_beta_pred[i] <- rescale_param(beta_upper, mu_beta_raw, X_pop_params_pred[i], mu_beta_offsets_raw);
        mu_scale_pred[i] <- rescale_param(scale_upper, mu_scale_raw, X_pop_params_pred[i], mu_scale_offsets_raw);

        prior_mu_alpha_pred[i] <- rescale_param(alpha_upper, prior_mu_alpha_raw, X_pop_params_pred[i], prior_mu_alpha_offsets_raw);
        prior_mu_beta_pred[i] <- rescale_param(beta_upper, prior_mu_beta_raw, X_pop_params_pred[i], prior_mu_beta_offsets_raw);
        prior_mu_scale_pred[i] <- rescale_param(scale_upper, prior_mu_scale_raw, X_pop_params_pred[i], prior_mu_scale_offsets_raw);
    }
}
