

// Variable definition
let gamma_max = gamma_slider.value;
let nu_max = nu_slider.value;
let Kd_cAA = Math.pow(10, Kd_cAA_slider.value);
let data = source.data;
let phiR = data['phiR'];



// Compute the properties
for (let i = 0; i < phiR.length; i++) {
    data['mu'][i] = steadyStateGrowthRate(gamma_max, phiR[i], nu_max, Kd_cAA);
    data['cAA'][i] = steadyStatePrecursorConc(gamma_max, phiR[i], nu_max, Kd_cAA) / Kd_cAA;
    data['gamma'][i] = steadyStateTransEff(gamma_max, phiR[i], nu_max, Kd_cAA) / gamma_max;
}

console.log(data['mu'])
// data['mu'] = steadyStateGrowthRate(gamma_max, phiR, nu_max, Kd_cAA);
// data['cAA'] = steadyStatePrecursorConc(gamma_max, phiR, nu_max, Kd_cAA);
// data['gamma'] = steadyStateTransEff(gamma_max, phiR, nu_max, Kd_cAA);


// for (let i = 0; i < data['cAA'].length; i++) {
//     data['cAA'][i] = data['cAA'][i] / Kd_cAA;
//     data['gamma'][i] = data['gamma'][i] / gamma_max;
// }

// Update
source.change.emit();
