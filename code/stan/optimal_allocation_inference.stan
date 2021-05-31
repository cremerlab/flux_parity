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
    real<lower=0> lam_sigma;
}

model { 
    // Compute the optimal allocation 
    real frac_term_a = phiO - 1;
    vector[N_frac] frac_term_b = -nu_max .* (-2 * Kd * gamma_max + gamma_max + nu_max);
    vector[N_frac] frac_term_c = sqrt(Kd * gamma_max * nu_max) .* (-gamma_max + nu_max);
    vector[N_frac] frac_denom = -4 * Kd * gamma_max * nu_max + gamma_max^2 + 2 * gamma_max * nu_max + nu_max^2;
    vector[N_frac] frac_mu = (frac_term_a * (frac_term_b + frac_term_c)) ./ frac_denom;

    // Compute the maximum growth rate 
    real lam_term_a = Kd - 1;
    vector[N_frac] lam_term_b = nu_max .* (1 - phiO - frac_mu) + gamma_max * frac_mu;
    vector[N_frac] lam_term_c = nu_max .* (1 - phiO - frac_mu) * gamma_max .* frac_mu;
    vector[N_frac] lam_mu = (-lam_term_b + sqrt(lam_term_b^2 + 4 * lam_term_a * lam_term_c)) ./ (2 * lam_term_a);

    // Compute the elongation rate
    vector[N_elong] c_AA = -1 + nu_max[elong_idx] .* (1 - phiO - frac_mu[elong_idx]) ./ growth_rate[elong_idx];
    vector[N_elong] elong_mu = gamma_max * c_AA ./(c_AA + Kd);

    // Define the priors
    nu_max ~ normal(0, 10);
    gamma_max ~ normal(gamma_max_mu, gamma_max_sigma);

    // phiO ~ normal(phiO_mu, phiO_sigma);
    Kd ~ gamma(1, 5);
    mass_frac_sigma ~ normal(0, 0.1);
    lam_sigma ~ normal(0, 1);
    elong_sigma ~ normal(0, 0.5);

    // Define the likelihood
    mass_frac ~ normal(frac_mu, mass_frac_sigma);
    growth_rate ~ normal(lam_mu, lam_sigma);
    resc_elong_rate ~ normal(elong_mu, elong_sigma);
}
