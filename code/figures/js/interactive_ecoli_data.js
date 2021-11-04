let phi_O = phiO_slider.value;
let Kd_cpc = Math.pow(10, Kd_cpc_slider.value)
let gamma_max = gamma_slider.value;
let const_phiRb = phiRb_slider.value;
let data = source.data;

// Compute the different scenarios
let phiRbs= [[], [], []]
let gammas = [[], [], []]
let lams = [[], [], []]

for (var i=0 ; i < nu_max.length; i++) {
    // Compute the scenarios
    phiRbs[0].push(const_phiRb);
    gammas[0].push(steadyStateGamma(gamma_max, const_phiRb, nu_max[i], Kd_cpc, phi_O) * 7459 / 3600);
    lams[0].push(steadyStateGrowthRate(gamma_max, const_phiRb, nu_max[i], Kd_cpc, phi_O))
    let sc2_phiRb = (1 - phi_O) * (nu_max[i] / (nu_max[i] + gamma_max));
    phiRbs[1].push(sc2_phiRb);
    gammas[1].push(steadyStateGamma(gamma_max, sc2_phiRb, nu_max[i], Kd_cpc, phi_O) * 7459 / 3600);
    lams[1].push(steadyStateGrowthRate(gamma_max, sc2_phiRb, nu_max[i], Kd_cpc, phi_O))
    let sc3_phiRb = optimalAllocation(gamma_max, nu_max[i], Kd_cpc, phi_O);
    phiRbs[2].push(sc3_phiRb);
    gammas[2].push(steadyStateGamma(gamma_max, sc3_phiRb, nu_max[i], Kd_cpc, phi_O) * 7459 / 3600);
    lams[2].push(steadyStateGrowthRate(gamma_max, sc3_phiRb, nu_max[i], Kd_cpc, phi_O))

}

// Update the source and emit the change
data['phiRb'] = phiRbs;
data['gamma'] = gammas;
data['lam'] = lams;
source.change.emit()

