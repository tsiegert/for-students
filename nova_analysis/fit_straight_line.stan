data {

  int<lower=0> N; // number of data points
  vector[N] x; // x observations
  vector[N] y; // y observations  
  vector[N] y_err; // y observation uncertainty

  int<lower=0> N_model; // number of data points for line
  vector[N_model] x_model; //where to evaluate the model
  
  //priors on the fitted parameters
  real mu_a;
  real sigma_a;
  real mu_b;
  real sigma_b;
  
}

parameters {
  
  real a; //intercept of straight line
  real b; //slop of straight line

}

model {

  // weakly informative priors
// no priors at the moment
  a ~ normal(mu_a,sigma_a);
  b ~ normal(mu_b,sigma_b);

  // likelihood

  y ~ normal(a+b*x, y_err);

  

}

generated quantities {

  vector[N] ppc;
  vector[N_model] straight_line;

  // generate the posterior of the
  // fitted straight line
  straight_line = a + b * x_model;

  // create posterior samples for PPC
  for (n in 1:N) {
    
    ppc[n] = normal_rng(a + b * x[n], y_err[n]);

  }

}