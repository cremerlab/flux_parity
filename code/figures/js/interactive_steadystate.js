

// Variable definition
let gamma_max = gamma_slider.value * 3600 / 7459;
let nu_max = nu_slider.value;
let Kd_cpc = Math.pow(10, Kd_cpc_slider.value);
let phi_O = phiO_slider.value;
let data = source.data;

// Identify the color index for nu
let nu_start = nu_slider.start;
let nu_step = nu_slider.step;
let nu_ind = parseInt((nu_max - nu_start) / nu_step);

// Set the phiR spacing given the supplied phiO
let nSteps = data['phiRb'][0].length;
let phiRb_range = linSpace(nu_start, 1 - phi_O - nu_start, nSteps)

// Compute the properties
data['color'][0] = colorArray[nu_ind];
nu_slider.bar_color = colorArray[nu_ind];
let phiRb = [];
let lam = [];
let cpc = [];
let gamma = [];
let max_lam = 0
for (let i = 0; i < phiRb_range.length; i++) {
    phiRb.push(phiRb_range[i]);
    let lam_ = steadyStateGrowthRate(gamma_max, phiRb_range[i], nu_max, Kd_cpc, phi_O); 
    if (lam_ > max_lam) { 
        max_lam = lam_;
    }
    lam.push(lam_)
    cpc.push(steadyStatePrecursorConc(gamma_max, phiRb_range[i], nu_max, Kd_cpc, phi_O) / Kd_cpc);
    let gamma_ = steadyStateGamma(gamma_max, phiRb_range[i], nu_max, Kd_cpc, phi_O)
    gamma.push(gamma_ / gamma_max);
}

for (let i =0 ; i < lam.length; i++) {
    lam[i] = lam[i] / max_lam;
}
data['phiRb'] = [phiRb];
data['cpc'] = [cpc];
data['gamma'] = [gamma];
data['lam'] = [lam];
source.change.emit();
