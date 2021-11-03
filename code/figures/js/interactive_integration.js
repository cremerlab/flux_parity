
// Define constants
const M0 = 0.01 * 1.5E17;
const cpc_0 = 1E-3;

// Parse the slider inputs
let phi_Rb = phiRb_slider.value;
let phi_O = phiO_slider.value;
let gamma_max = gamma_slider.value;
let nu_max = nu_slider.value;
let Kd_cpc = Math.pow(10, Kd_cpc_slider.value);
let Kd_cnt = Math.pow(10, Kd_cnt_slider.value);
let c_nt = Math.pow(10, cnt_slider.value);
let Y  = Math.pow(10, Y_slider.value);

// Parse the data inputs
let data = source.data;
let pie = pie_source.data;

// Given the inputs, set initial conditions and instantiate params and arg
let M_Rb_0 = phi_Rb * M0;
let M_Mb_0 = (1 - phi_Rb - phi_O) * M0;
let cnt_0 = c_nt;
let params = [M0, M_Rb_0, M_Mb_0, cpc_0, cnt_0];
let args = [gamma_max, nu_max, Y, phi_Rb, phi_O, Kd_cpc, Kd_cnt];

// Get the time range and peform integration
let time = data['time'];
let dynamics = odeintForwardEuler(selfReplicator, params, args, 0.001, time.length);


// Update the source
for (let i=0; i < time.length; i++ ) {
    data['rel_M'][i] = dynamics[i][0]/ M0;
    data['cpc_Kd'][i] = dynamics[i][3] / Kd_cpc;
    data['cnt_rel'][i] = dynamics[i][4] / cnt_0;
    data['gamma'][i] = dynamics[i][3] / (dynamics[i][3] + Kd_cpc);
    data['nu'][i] = dynamics[i][4] / (dynamics[i][4] + Kd_cnt);
}
source.change.emit()

// Update the pie chart
let angles = [2 * Math.PI * (1 - phi_O - phi_Rb), 2 * Math.PI * phi_O, 2 * Math.PI * phi_Rb];

let start_angles = [0, angles[0], angles[0] + angles[1]];
let end_angles = [angles[0], angles[0] + angles[1], angles[0] + angles[1] + angles[2]];
for (var i = 0; i < 3; i++) {
    pie['start_angle'][i] = start_angles[i];
    pie['end_angle'][i] = end_angles[i];
}
pie_source.change.emit()


