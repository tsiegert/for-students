data {

    // this is the input data that is given from python to stan via a dictionary
    
    int<lower=0> N;           // number of response matrix points
    int<lower=0> Np;          // number of pointings in observation data set
    int<lower=0> y[Np*N];     // y observations = counts 
    int<lower=1> Nsky;        // number of sky models

    int<lower=0> Ngood;       // only live detectors
    int good_indices[Ngood];  // pointing indices with failed detectors
    
    real bg_model_lines[Np,N];      // BG model (response) lines
    real bg_model_conti[Np,N];      // BG model (response) continuum
    real conv_sky[Nsky,Np,N]; // SKY model(s) (already convolved)
  
    // background re-normalisation times
    int<lower=0> Ncuts;       // number of time nodes where/when the background is rescaled
    int bg_cuts[Np];          // translated cut positions (which pointings are the change points)
    int bg_idx_arr[Np];       // data space indices where the Ncuts cuts are applied
  
    //priors on the fitted parameters
    real mu_flux[Nsky];             // prior for the scaling of each sky model 
    real sigma_flux[Nsky];          // sky model prior width for scaling
    
    real mu_Abg_lines;         // prior for background scaling lines
    real sigma_Abg_lines;      // prior width background lines

    real mu_Abg_conti;         // prior for background scaling continuum
    real sigma_Abg_conti;      // prior width background continuum
    
}



transformed data {
 
    int<lower=0> y_g[Ngood];

    /*
    for (ng in 1:Ngood) {
        
        y_g[ng] = y[good_indices[ng]];
        
    }
    */
    y_g = y[good_indices];
    
}



parameters {

    real<lower=1e-8> flux[Nsky]; // scaling to each sky model
    real<lower=1e-8> Abg_lines[Ncuts]; // background model amplitude(s) lines
    real<lower=1e-8> Abg_conti[Ncuts]; // background model amplitude(s) continuum
    
}



transformed parameters {

    // definition of our model

    real model_values[Np*N];
    real model_values_g[Ngood];
    
    for (np in 1:Np) {
        
        for (nn in 1:N) {
            
            model_values[N*(np-1)+nn] = Abg_lines[bg_idx_arr[np]] * bg_model_lines[np,nn] + Abg_conti[bg_idx_arr[np]] * bg_model_conti[np,nn];

            for (ns in 1:Nsky) {
                
                model_values[N*(np-1)+nn] += flux[ns] * conv_sky[ns,np,nn];    
                
            }
            
        }
        
    }
    
    
    for (ng in 1:Ngood) {
        
        model_values_g[ng] = model_values[good_indices[ng]];
        
    }
    
}



model {

    // initialisation of model, i.e. parameters, their priors (if any) and likelihood
    
    // normal priors for flux and bg scaling parameters
    flux ~ normal(mu_flux,sigma_flux);     // normal prior(s) for flux 
    Abg_lines ~ normal(mu_Abg_lines,sigma_Abg_lines);        // normal prior(s) for bg lines
    Abg_conti ~ normal(mu_Abg_conti,sigma_Abg_conti);        // normal prior(s) for bg continuum
    
    // likelihood: since we are dealing with count data, this is just the poisson likelihood
    y_g ~ poisson(model_values_g);

}



generated quantities {

    vector[Ngood] ppc;

    vector[Np] model_tot = rep_vector(0,Np);
    vector[Np] model_bg_lines = rep_vector(0,Np);
    vector[Np] model_bg_conti = rep_vector(0,Np);
    matrix[Nsky,Np] model_sky = rep_matrix(0,Nsky,Np);

  // create posterior samples for PPC
  // and
  // generate the posterior of the model
  for (np in 1:Np) {
      
      for (nn in 1:N) {
         
         for (ns in 1:Nsky) {
             
             model_sky[ns,np] += flux[ns] * conv_sky[ns,np,nn];
             model_tot[np] += flux[ns] * conv_sky[ns,np,nn];
             
         }
         
         model_bg_lines[np] += Abg_lines[bg_idx_arr[np]] * bg_model_lines[np,nn];
         model_bg_conti[np] += Abg_conti[bg_idx_arr[np]] * bg_model_conti[np,nn];         
         model_tot[np] += Abg_lines[bg_idx_arr[np]] * bg_model_lines[np,nn] + Abg_conti[bg_idx_arr[np]] * bg_model_conti[np,nn];
         //model_tot[np] += model_bg_lines[np] + model_bg_conti[np];
         
      }
      
  }
  
  
  for (ng in 1:Ngood) {
      
      ppc[ng] = poisson_rng(model_values_g[ng]);
      
  }

}
