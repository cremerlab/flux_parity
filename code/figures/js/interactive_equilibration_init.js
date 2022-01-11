let phiRb = phiRb_slider.value;
let ind = parseInt((phiRb - phiRb_slider.start) / phiRb_slider.step);
let start_gamma = gamma[ind];
let start_nu = nu[ind];
let tRNA = tot_tRNA[ind];
let fluxData = flux_source.data;
let pointData = point_source.data;

// Given an input, adjust the tent plot
for (var i = 0; i < phiRb_range.length; i++) {
    fluxData['metab_flux'][i] = start_nu * (1 - phi_O - phiRb_range[i]);
    fluxData['trans_flux'][i] = start_gamma * phiRb_range[i] * (1 - tRNA);
}
pointData['growth_rate'][0] = growth_rate_range[ind];
pointData['phiRb'][0] = phiRb;

// Update sources
flux_source.change.emit();
point_source.change.emit(); 