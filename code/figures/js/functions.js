
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

function ppGppSelfReplicator(params, args, dt) {
    // Unpack params
    let [M, M_Rb, M_Mb, T_AA, T_AA_star] = params;
    let [gamma_max, nu_max, tau, Kd_TAA, Kd_TAA_star, kappa_max, phi_O, phi_Rb, dynamic] = args;
    
    // Compute rates
    let ratio = T_AA_star / T_AA;
    let gamma = gamma_max * (T_AA_star / (T_AA_star + Kd_TAA_star));
    let nu = nu_max * (T_AA / (T_AA + Kd_TAA));

    if (dynamic == true) { 
        let phi_Rb = ratio / (ratio + tau); 
        let kappa = kappa_max * phi_Rb;
    }

    else { 
        let kappa = kappa_max;
    }

    // Define dynamics
    let dM_dt = gamma * M_Rb;
    let dT_AA_star_dt = (nu * M_Mb - dM_dt * (1 + T_AA_star)) / M;
    let dT_AA_dt = kappa + (dM_dt - nu * M_Mb - dM_dt) / M;

    // Define allocation
    let dM_Rb_dt = phi_Rb * dM_dt;
    let dM_Mb_dt = (1 - phi_O - phi_Rb) * dM_dt

    // Pack and return
    return [dM_dt * dt, dM_Rb_dt * dt, dM_Mb_dt * dt, dT_AA_dt * dt, dT_AA_star_dt * dt]
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

function constTranslation(gamma_max, nu_max, phi_O) { 
    return (1 - phi_O) * (nu_max / (nu_max + gamma_max))
}

// /////////////////////////////////////////////////////////////////////////////
// FUNCTIONS FOR NUMERICAL INTEGRATION
// /////////////////////////////////////////////////////////////////////////////
function odeintForwardEuler(fun, params, args, dt, nSteps) { 
    let out = [params]; 
    for (let i=1; i < nSteps ; i++) { 
        let deriv = fun(out[i-1], args, dt)
        for (let j=0; j < deriv.length; j++) {
            deriv[j] = out[i-1][j] + deriv[j];
        }
        out.push(deriv);
    }
    return out
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
