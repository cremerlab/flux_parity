
// /////////////////////////////////////////////////////////////////////////////
// FUNCTIONS DEFINING SELF-REPLICATOR MODEL
// /////////////////////////////////////////////////////////////////////////////

function selfReplicator(params,args, dt) { 
    let [M, M_Rb, M_Mb, c_pc, c_nt] = params;
    let [gamma_max, nu_max, Y, phi_Rb, phi_O, Kd_cpc, Kd_cnt] = args;

    // Compute the transalational efficiency
    let gamma = gamma_max * (c_pc / (c_pc + Kd_cpc));
    let nu = nu_max * (c_nt / (c_nt + Kd_cnt));

    // Define the biomass dynamics
    let dM_dt = gamma * M_Rb;
    
    // Define allocation dynamics
    let dM_Rb_dt = phi_Rb * dM_dt;
    let dM_Mb_dt = (1 - phi_Rb - phi_O) * dM_dt;

    // Define precursor dynamics
    let dcpc_dt = (nu * M_Mb - dM_dt * (1 + c_pc)) / M;

    // Define nutrient dynamics
    let dcnt_dt = -nu * M_Mb / Y;
    return [dM_dt * dt, dM_Rb_dt * dt, dM_Mb_dt * dt, dcpc_dt * dt, dcnt_dt * dt]
}

function fpmSelfReplicator(params, args, dt) {
    // Unpack params
    let [M, M_Rb, M_Mb, T_AA, T_AA_star] = params;
 
    let [gamma_max, nu_max, tau, Kd_TAA, Kd_TAA_star, kappa_max, phi_O] = args;

    // Compute the rates
    let nu = nu_max * (T_AA / (T_AA + Kd_TAA));
    let gamma = gamma_max * (T_AA_star / (T_AA_star + Kd_TAA_star));

    // Compute the allocation
    let ratio = (T_AA_star / T_AA);
    let phi_Rb = (1 - phi_O) * ratio / (ratio + tau); 
    let kappa = kappa_max * ratio / (ratio + tau);

    // Compute dynamics
    let dM_dt = gamma * M_Rb
    let dTAA_star_dt = (nu * M_Mb - dM_dt - dM_dt * T_AA_star) / M
    let dTAA_dt = kappa + ((dM_dt - dM_dt * T_AA - nu * M_Mb) / M)
    let dM_Rb_dt = phi_Rb * dM_dt
    let dM_Mb_dt = (1 - phi_Rb - phi_O) * dM_dt

    return [M + (dM_dt * dt), M_Rb + (dM_Rb_dt * dt), M_Mb + (dM_Mb_dt * dt), T_AA + (dTAA_dt * dt), T_AA_star + (dTAA_star_dt * dt)]
}

// /////////////////////////////////////////////////////////////////////////////
// FUNCTIONS DEFINING STEADY-STATE PARAMETERS
// /////////////////////////////////////////////////////////////////////////////
function steadyStatePrecursorConc(gamma_max, phi_Rb, nu_max, Kd_cpc, phi_O) {
    let growth_rate = steadyStateGrowthRate(gamma_max, phi_Rb, nu_max, Kd_cpc, phi_O)
    return (nu_max * (1 - phi_Rb - phi_O) / growth_rate) - 1
}

function steadyStateGrowthRate(gamma_max, phi_Rb, nu_max, Kd_cpc, phi_O) {
    let Gamma = gamma_max * phi_Rb;
    let Nu = nu_max * (1 - phi_Rb - phi_O);
    let numer = Nu + Gamma - Math.sqrt(Math.pow(Nu + Gamma, 2) - 4 * (1 - Kd_cpc) * Nu * Gamma)
    let denom = 2 * (1 - Kd_cpc);
    return numer / denom
}

function steadyStateGamma(gamma_max, phi_Rb, nu_max, Kd_cpc, phi_O) {
    let cpc = steadyStatePrecursorConc(gamma_max, phi_Rb, nu_max, Kd_cpc, phi_O);
    return gamma_max * (cpc / (cpc + Kd_cpc));
}

// /////////////////////////////////////////////////////////////////////////////
// FUNCTIONS DEFINING ALLOCATION STRATEGIES
// /////////////////////////////////////////////////////////////////////////////
function optimalAllocation(gamma_max, nu_max, Kd_cpc, phi_O) {
    let prefix = 1 / (4 * Kd_cpc * gamma_max * nu_max - Math.pow(nu_max + gamma_max, 2))
    let bracket = 2 * Kd_cpc * gamma_max * nu_max - gamma_max * nu_max + Math.sqrt(Kd_cpc * gamma_max * nu_max) * (nu_max - gamma_max) - Math.pow(nu_max, 2)
    return (1 - phi_O) * prefix * bracket
}

function constTranslation(gamma_max, nu_max, cpc_Kd, Kd_cpc, phi_O) { 
    let _cpc = cpc_Kd * Kd_cpc
    return (1 - phi_O) * nu_max * (_cpc + Kd_cpc) / (nu_max  * (_cpc + Kd_cpc) + gamma_max * _cpc * (_cpc + 1));
}

// /////////////////////////////////////////////////////////////////////////////
// FUNCTIONS FOR NUMERICAL INTEGRATION
// /////////////////////////////////////////////////////////////////////////////
function odeintForwardEuler(fun, params, args, dt, nSteps) { 
    let out = [params]; 
    for (let i=1; i < nSteps ; i++) { 
        let deriv = fun(out[i-1], args, dt)
        for (let j=0; j < deriv.length; j++) {
            deriv[j] += out[i-1][j];
        }
        out.push(deriv);
    }
    return out
}

function ppGppEquilibrate(gamma_max, nu_max, tau, Kd_TAA, Kd_TAA_star, kappa_max, phi_O) {
    // Determine 'optimal' allocation from simple model
    let optPhiRb = optimalAllocation(gamma_max, nu_max, 0.01, phi_O);

    // Set initial conditions
    let M0 = 1E9;
    let MRb = optPhiRb * M0;
    let MMb = (1 - optPhiRb - phi_O) * M0;
    let tAA = 0.00002; 
    let tAAStar = 0.00002;

    // Define the timescale to integrate
    let dt = 0.001;
    let nSteps = parseInt(100 / dt);

    // Pack parameters and args
    let params = [M0, MRb, MMb, tAA, tAAStar];
    let args = [gamma_max, nu_max, tau, Kd_TAA, Kd_TAA_star, kappa_max, phi_O];

    // Perform integration
    let out = odeintForwardEuler(fpmSelfReplicator, params, args, dt, nSteps)

    // Identify output
    let eq = out.slice(-1)[0]   

    // Compute steady-state params.
    let eqPhiRb = eq[1] / eq[0];
    let eqPhiMb = eq[2] / eq[0];
    let eqTAA = eq[3];
    let eqTAAStar = eq[4]; 
    let equil = [eqPhiRb, eqPhiMb, eqTAA, eqTAAStar];
    return  equil
}

// /////////////////////////////////////////////////////////////////////////////
// MISCELLANEOUS
// /////////////////////////////////////////////////////////////////////////////

// Function for linearly spacing values. Adapted from StackOverflow user mhodges
// https://stackoverflow.com/questions/40475155/does-javascript-have-a-method-that-returns-an-array-of-numbers-based-on-start-s
function linSpace(startValue, stopValue, nSteps) {
  let arr = [];
  let step = (stopValue - startValue) / nSteps;
  for (var i = 0; i < nSteps; i++) {
    arr.push(startValue + (step * i));
  }
  return arr;
}
