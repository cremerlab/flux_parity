let nu_max_preshift = nu_max_preshift_slider.value;
let nu_max_postshift = nu_max_postshift_slider.value;
let phiO_preshift = phiO_preshift_slider.value;
let phiO_postshift = phiO_postshift_slider.value;
let data = source.data;


// Get the pre and postshift equilibrium
let preshift_params = ppGppEquilibrate(gamma_max,
        nu_max_preshift,phiO_preshift, Kd_TAA, Kd_TAA_star,
        kappa_max)
let postshift_params = ppGppEquilibrate(gamma_max,
        nu_max_postshift,phiO_postshift, Kd_TAA, Kd_TAA_star,
        kappa_max)
console.log(postshift_params)

// Perform the integration for the dynamic shift
const M0 = 1E9;
const dt = preshift_time[1] - preshift_time[0];
let preshiftNSteps = preshift_time.length; 
let postshiftNSteps = postshift_time.length; 
let init_params = [M0, M0 * preshift_params[0], M0 * preshift_params[1],
                  preshift_params[2], preshift_params[3]];

let init_args = [gamma_max, nu_max_preshift, phiO_preshift, Kd_TAA, Kd_TAA_star, phiO_preshift,  0, true];
let dynamic_preshift = odeintForwardEuler(ppGppSelfReplicator, init_params, init_args, dt, preshiftNSteps);
init_params = dynamic_preshift.slice(-1);
init_args = [gamma_max, nu_max_postshift, phiO_postshift, Kd_TAA, Kd_TAA_star, phiO_postshift, 0, true];
let dynamic_postshift = odeintForwardEuler(ppGppSelfReplicator, init_params, init_args, dt, postshiftNSteps);
dynamic_postshift = dynamic_postshift.slice(1, dynamic_postshift.length + 1);

// Merge
let dynamic = dynamic_preshift.concat(dynamic_postshift);
let dynamicPhiRb = [];
let dynamicMRbM = []; 
let dynamicTRNA = [];
let dynamicGr = [];

// Preshift dynamics
for (var i = 0; i < preshiftNSteps + postshiftNSteps; i++) {
    let ratio = dynamic[i][4] / dynamic[i][3];
    let tRNA = (dynamic[i][3] / Kd_TAA) + (dynamic[i][4] / Kd_TAA_star);
    dynamicPhiRb.push(ratio / (ratio + tau));
    dynamicMRbM.push(dynamic[i][1] / dynamic[i][0]);
    dynamicTRNA.push(tRNA);
    if (i > 0) {
        dynamicGr.push( Math.log(dynamic[i][0] / dynamic[i-1][0]) / dt)
    }
}

// Update the source 
data['phiRb'][1] = dynamicPhiRb;
data['Mrb_M'][1] = dynamicMRbM;
data['tRNA'][1] = dynamicTRNA;
data['lam'][1] = dynamicGr;

source.change.emit()