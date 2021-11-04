

// Variable definition
let gamma_max = gamma_slider.value;
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
for (let i = 0; i < phiRb_range.length; i++) {
    phiRb.push(phiRb_range[i]);
    lam.push(steadyStateGrowthRate(gamma_max, phiRb_range[i], nu_max, Kd_cpc, phi_O));
    cpc.push(steadyStatePrecursorConc(gamma_max, phiRb_range[i], nu_max, Kd_cpc, phi_O) / Kd_cpc);
    gamma.push(steadyStateGamma(gamma_max, phiRb_range[i], nu_max, Kd_cpc, phi_O));
}
data['phiRb'] = [phiRb];
data['cpc'] = [cpc];
data['gamma'] = [gamma];
data['lam'] = [lam];
source.change.emit();
