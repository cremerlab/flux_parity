
// /////////////////////////////////////////////////////////////////////////////
// FUNCTIONS DEFINING SELF-REPLICATOR MODEL
// /////////////////////////////////////////////////////////////////////////////

function batchCultureSelfReplicator(params,args, dt) { 
    let [Mr, Mp, cAA, cN] = params;
    let [gamma_max, nu_max, omega, phi_R, Kd_cAA, Kd_cN] = args;

    // Compute the transalational efficiency
    let gamma = gamma_max * (cAA / (cAA + Kd_cAA));
    let nu = nu_max * (cN / (cN + Kd_cN));

    // Define the biomass dynamics
    let dM_dt = gamma * Mr;
    
    // Define allocation dynamics
    let dMr_dt = phi_R * dM_dt;
    let dMp_dt = (1 - phi_R) * dM_dt;

    // Define precursor dynamics
    let dcAA_dt = (nu * Mp - dM_dt * (1 + cAA)) / (Mr + Mp);

    // Define nutrient dynamics
    let dcN_dt = -nu * Mp / omega;
    return [dMr_dt * dt, dMp_dt * dt, dcAA_dt * dt, dcN_dt * dt]
}

// /////////////////////////////////////////////////////////////////////////////
// FUNCTIONS DEFINING STEADY-STATE PARAMETERS
// /////////////////////////////////////////////////////////////////////////////
function steadyStatePrecursorConc(gamma_max, phi_R, nu_max, Kd_cAA) {
    let growth_rate = steadyStateGrowthRate(gamma_max, phi_R, nu_max, Kd_cAA)
    return (nu_max * (1 - phi_R) / growth_rate) - 1
}

function steadyStateGrowthRate(gamma_max, phi_R, nu_max, Kd_cAA) {
    let Gamma = gamma_max * phi_R;
    let Nu = nu_max * (1 - phi_R);
    let numer = Nu + Gamma - Math.sqrt(Math.pow((Nu + Gamma, 2) - 4 * (1 - Kd_cAA) * Nu * Gamma))
    let denom = 2 * (1 - Kd_cAA);
    return numer / denom
}

function steadyStateTransEff(gamma_max, phi_R, nu_max, Kd_cAA) {
    let cAA = steadyStatePrecursorConc(gamma_max, phi_R, nu_max, Kd_cAA);
    return gamma_max * (cAA / (cAA + Kd_cAA));
}

// /////////////////////////////////////////////////////////////////////////////
// FUNCTIONS DEFINING ALLOCATION STRATEGIES
// /////////////////////////////////////////////////////////////////////////////
function optimalAllocation(gamma_max, nu_max, Kd_cAA) {
    let prefix = 1 / (4 * Kd_cAA * gamma_max * nu_max - Math.pow(nu_max + gamma_max, 2))
    let bracket = 2 * Kd_cAA * gamma_max * nu_max - gamma_max * nu_max + Math.sqrt(Kd_cAA * gamma_max * nu_max) * (nu_max - gamma_max) - Math.pow(nu_max, 2)
    return prefix * bracket
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


// Functions below aren't quite working yet

function odeintMidpointMethod(fun, params, args, dt, nSteps){
    let out = [params];
    for (let i=1; i < nSteps; i++) {
        let k1 = fun(out[i-1], args, dt);
        for (let j=1; j < k1.length; j++) {
            k1[j] = out[i-1][j] + (k1[j] /2);
        }
        let k2 = fun(k1, args, dt/2);
        for (let j=1; j < k1.length; j++) {
            k2[j] = out[i-1][j] + k2[j];
        }
        out.push(k2)
    }
    return out
}

function odeintRK4(fun, params, args, dt, nSteps) { 
        let out = [params];
        for (let i=1; i < nSteps; i++) { 
            let k1 = fun(out[i-1], args, dt);
            let k2 = fun(out[i-1] + k1/2, args, dt/2);
            let k3 = fun(out[i-1] + k2/2, args, dt/2);
            let k4 = fun(out[i-1] + k3, args, dt);
            let deriv = out[i-1] + dt * (k1 + 2*k2 + 2 * k3 + k4)/6;
            out.push(deriv);
        }
        return out
}
