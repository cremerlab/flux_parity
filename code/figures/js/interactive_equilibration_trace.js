let phiRb_init = phiRb_slider.value;
let ind = parseInt((phiRb_init - phiRb_slider.start) / phiRb_slider.step);
let start_TAA = TAA[ind];
let start_TAA_star = TAA_star[ind]; 
let start_ratio = start_TAA_star / start_TAA;
let trace_data = trace_source.data;
equilibrate.label = 'Equilibrating...'

// Instantiate storage vecs
let args = [gamma_max, nu_max, tau, Kd_TAA, Kd_TAA_star, kappa_max, phi_O];

// let out = odeintForwardEuler(fpmSelfReplicator, params, args, step, trace_data['time'].length); 
let phiRb_out = [];
let metab_flux_out = [];
let trans_flux_out = [];
// Update the traces
for (var i = 0; i < trace_data['time'].length; i++) {
    if (i == 0) { 
        var params = [1, phiRb_init , (1 - phi_O - phiRb_init), start_TAA, start_TAA_star];
        var out = fpmSelfReplicator(params, args, step)
    }

    else { 
        out = fpmSelfReplicator(out, args, step)
    }

    let TAA_ = out[3]
    let TAA_star_ = out[4]
    let ratio_ = TAA_star_ / TAA_
    let gamma_ = gamma_max * (TAA_star_ / (TAA_star_ + Kd_TAA_star));
    let nu_ = nu_max * (TAA_ / (TAA_ + Kd_TAA));
    let kappa_ = kappa_max * ratio_ / (ratio_ + tau);
    let phiRb_ = (1 - phi_O) * ratio_ / (ratio_ + tau);

    phiRb_out.push(phiRb_)
    metab_flux_out.push(kappa_ + nu_ * out[2]/out[0]);
    trans_flux_out.push(gamma_ * out[1]/out[0] * (1 - TAA_ - TAA_star_));
}
console.log(phiRb_out)
// Clear the previous trace
trace_data['metab_flux'] = metab_flux_out;
trace_data['trans_flux'] = trans_flux_out;
trace_data['phi_Rb'] = phiRb_out;

trace_source.change.emit();
equilibrate.label = 'Equilibrate!'

// // Update the trace
// let interval = 0
// for (var i = 0; i < metab_flux_out.length; i++) {
//     trace_data


// }

// function updateTrace(i) { 
//     trace_
// }