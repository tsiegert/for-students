data {

  int<lower=0> N;             // number of data points
  vector[N] x_obs;            // x observations
  vector[N] y_obs;            // y observations  
  vector<lower=0>[N] sigma_x; // measurement uncertainty in x
  vector<lower=0>[N] sigma_y; // measurement uncertainty in y

  
  int<lower=0> N_model;       // number of point to evaluate the fitted model
  vector[N_model] x_model;    // where to evaluate the model (for plotting)

  // priors on the fitted parameters
  // (can be very broad to avoid bias)
  // real mu_a;
  // real sigma_a;
  // real mu_b;
  // real sigma_b;

}

parameters {
  
  real a;  // y intercept
  real b;  // slope
  vector[N] x_latent;  // latent x location (x_obs + randomn(sigma_x))
  
}


transformed parameters {

  // latent y values, not obscured by measurement error
  // (the model or correlation you are fitting for)
  vector[N] y_true = a + b*x_latent;

}


model {

  // priors
  // a ~ normal(mu_a,sigma_a);
  // b ~ normal(mu_b,sigma_b);

  // where x can be found 
  x_latent ~ normal(x_obs,sigma_x);

  // likelihood
  y_obs ~ normal(y_true, sigma_y);
  
}


generated quantities {

  vector[N] ppc_x;
  vector[N] ppc_y;
  
  vector[N_model] line;

  // generate posteriors
  // fitted model
  line = a + b * x_model;

  // create posterior samples for PPC
  // generates new data given the likelihood and posterior (best fit)
  // to check how often you get a data set that looks like the one you used
  for (n in 1:N) {

    ppc_x[n] = normal_rng(x_latent[n], sigma_x[n]);
    ppc_y[n] = normal_rng(a + b*x_latent[n], sigma_y[n]);
    
  }
  

}