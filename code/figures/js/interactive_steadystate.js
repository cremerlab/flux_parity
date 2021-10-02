

// Variable definition
let gamma_max = gamma_slider.value;
let nu_max = nu_slider.value;
let Kd_cAA = Math.pow(10, Kd_cAA_slider.value);
let data = source.data;
let phiR = data['phiR'];

// Compute the maximum phiR and thus maximum growth rate
let opt_phiR = optimalAllocation(gamma_max, nu_max, Kd_cAA);
let max_mu = steadyStateGrowthRate(gamma_max, opt_phiR,nu_max, Kd_cAA);

// Compute the properties
for (let i = 0; i < phiR.length; i++) {

    data['mu'][i] = steadyStateGrowthRate(gamma_max, phiR[i], nu_max, Kd_cAA) / max_mu;
    data['cAA'][i] = steadyStatePrecursorConc(gamma_max, phiR[i], nu_max, Kd_cAA) / Kd_cAA;
    data['gamma'][i] = steadyStateTransEff(gamma_max, phiR[i], nu_max, Kd_cAA) / gamma_max;
}

source.change.emit();
