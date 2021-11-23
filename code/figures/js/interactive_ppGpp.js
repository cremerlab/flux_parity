// Define variables
const gamma_max = 9.6527;
const Kd_cpc = 0.01;
const phi_O = 0.55;
let Kd_TAA = Math.pow(10, Kd_TAA_slider.value);
let Kd_TAA_star = Math.pow(10, Kd_TAA_star_slider.value);
let kappa_max = Math.pow(10, kappa_slider.value);
let tau = tau_slider.value;

// Load and link data
let data = source.data;
let nu_max = data.nu_max[0];
let out = [];

for (var i = 0; i < nu_max.length; i++) {

    // Equilibrate
    let eq = ppGppEquilibrate(gamma_max, 
                               nu_max[i],
                               tau,
                               Kd_TAA,
                               Kd_TAA_star,
                               kappa_max,
                               phi_O);
    // Compute translation rate
    let gamma = gamma_max * eq[3] / (eq[3] + Kd_TAA_star);

    // Compute the growth rate
    let lam = gamma * eq[0];

    // Update source
    data['gamma'][1][i] = gamma * 7459 / 3600;
    data['lam'][1][i] = lam;
    data['phiRb'][1][i] = eq[0];
    data['tRNA_ribo'][1][i] = (eq[2] + eq[3]) / (eq[0] / 7459)
}

source.change.emit();

