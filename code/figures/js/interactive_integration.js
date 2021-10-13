// Define constants
const M0 = 0.01 * 1.5E17;
const cAA_0 = 1E-3;

// Parse the slider inputs
let phi_R = phiR_slider.value;
let gamma_max = gamma_slider.value;
let nu_max = nu_slider.value;
let Kd_cAA = Math.pow(10, Kd_cAA_slider.value);
let Kd_cN = Math.pow(10, Kd_cN_slider.value);
let cN = Math.pow(10, cN_slider.value);
let omega = Math.pow(10, omega_slider.value);

// Parse the data inputs
let data = source.data;

// Given the inputs, set initial conditions and instantiate params and arg
let MR_0 = phi_R * M0;
let MP_0 = (1 - phi_R) * M0;
let cN_0 = cN;
let params = [MR_0, MP_0, cAA_0, cN_0];
let args = [gamma_max, nu_max, omega, phi_R, Kd_cAA, Kd_cN];

// Get the time range and peform integration
let time = data['time'];
let dynamics = odeintForwardEuler(batchCultureSelfReplicator, params, args, 0.001, time.length);


// Update the source
for (let i=0; i < time.length; i++ ) {
    data['rel_M'][i] = (dynamics[i][0] + dynamics[i][1]) / M0;
    data['cAA_Kd'][i] = dynamics[i][2] / Kd_cAA;
    data['cN_Kd'][i] = dynamics[i][3] / Kd_cN;
    data['gamma'][i] = dynamics[i][2] / (dynamics[i][2] + Kd_cAA);
    data['nu'][i] = dynamics[i][3] / (dynamics[i][3] + Kd_cN);
}
source.change.emit()


