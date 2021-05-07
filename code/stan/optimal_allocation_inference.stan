data { 
    // Dimensional parameters
    int<lower=1> N_frac; // Number of mass fraction measurements
    int<lower=1> N_elong; // Number of elongation measurements
    int<lower=1, upper=N_frac> elong_idx[N_elong];


    // Data
    vector<lower=0>[N_frac] growth_rate;
    vector<lower=0, upper=1>[N_frac] mass_frac;
    vector<lower=0>[N_elong] elong_rate;

    // Informative priors
    real<lower=0> gamma_max_mu;
    real<lower=0> gamma_max_sigma;
    real<lower=0> phiO;

}

transformed data {
    vector<lower=0>[N_elong] resc_elong_rate = elong_rate * 7459/3600;
}

parameters {
    real<lower=0> gamma_max;
    vector<lower=0>[N_frac] nu_max;
    real<lower=0> Kd;
    real<lower=0> mass_frac_sigma;
    real<lower=0> elong_sigma;
}

model { 
    // Compute the parameters
    vector[N_frac] a = gamma_max * nu_max;
    vector[N_frac] b = growth_rate .* (gamma_max + nu_max) + gamma_max * nu_max * (phiO - 1);
    vector[N_frac] c = growth_rate .* (nu_max * (phiO -1) - growth_rate * (Kd-1));
    vector[N_frac] frac_mu =  (-b - sqrt(b^2 - 4 * a .* c)) ./ (2 * a);
    vector[N_elong] c_AA = -1 + nu_max[elong_idx] .* (1 - phiO - frac_mu[elong_idx]) ./ growth_rate[elong_idx];
    vector[N_elong] elong_mu = gamma_max * c_AA ./(c_AA + Kd);

    // Define the priors
    nu_max ~ normal(0, 10);
    gamma_max ~ normal(gamma_max_mu, gamma_max_sigma);
    // phiO ~ normal(phiO_mu, phiO_sigma);
    Kd ~ gamma(1, 5);
    mass_frac_sigma ~ normal(0, 0.1);
    elong_sigma ~ normal(0, 0.5);

    // Define the likelihood
    mass_frac ~ normal(frac_mu, mass_frac_sigma);
    resc_elong_rate ~ normal(elong_mu, elong_sigma);
}
