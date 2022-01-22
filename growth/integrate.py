import numpy as np 
import pandas as pd 
import scipy.integrate
import tqdm
from .model import self_replicator_FPM 

def equilibrate_FPM(args, 
                    tol=3, 
                    max_iter=10, 
                    dt=0.0001, 
                    t_return=1):
    """
    Numerically integrates the flux-parity model until steady-state is reached. 

    Parameters 
    -----------
    args: dict
        Dictionary of arguments to be passed to the system of ODES via 
        scipy.integrate.odeint. See documentation for self_replicator_FPM
        for documentation on args.
    tol: int 
        Absolute tolerance for finding equilibrium. Corresponds to the decimal
        limit of the ratio of precursor concentrations between successive time 
        steps. Default is 3 decimal places
    max_iter: int
        The maximum number of interations over which to run the integration. Default 
        is 10 iterations.
    dt: float
        Size of timestep to be taken. Default is 0.0001 time units
    t_return: int
        The number of final N time points to return. Default is 1, returning 
        the final time step

    Returns
    -------
    out: list or list of lists
        Returns the number of elements of the integrator for the final  
        time point or points, given the value of t_return. 
    """
    M0 = 1E9
    alloc_space = (1 - args['phi_O']) / 2
    phi_Rb = alloc_space
    phi_Mb = alloc_space
    if 'nutrients' in args.keys():
        init_params = [M0, phi_Rb * M0, phi_Mb * M0, args['nutrients']['c_nt'], 1E-5, 1E-5]
    else:
        init_params = [M0, phi_Rb * M0, phi_Mb * M0, 1E-5, 1E-5]

    iterations = 1 
    converged = False
    max_time = 200 
    while (iterations <= max_iter) & (converged == False):
        time = np.arange(0, max_time, dt)
        out = scipy.integrate.odeint(self_replicator_FPM, 
                                    init_params, 
                                    time,
                                    args=(args,)) 
    
        # Determine if a steady state has been reached
        ratio = out[-1][-1] / out[-1][-2]
        MRb_M = out[-1][1] / out[-1][0]
        if 'phiRb' not in args.keys():
            phiRb = (1 - args['phi_O'] ) * ratio / (ratio + args['tau'])
        else:
            phiRb = args['phiRb']
        ribo_ratio = MRb_M / phiRb
        if np.round(ribo_ratio, decimals=tol) == 1:
            converged = True
        else:
            iterations +=1
            max_time += 10
       
        if iterations == max_iter:
            print(f'Steady state was not reached (ratio of Mrb_M / phiRb= {np.round(ribo_ratio, decimals=tol)}. Returning output anyway.')
    if t_return != 1:
        return out[-t_return:]
    else: 
        return out[-1]


def compute_nu(gamma_max, 
               Kd, 
               phiRb, 
               lam, 
               phi_O):
    """
    Estimates the metabolic rate given measured params under the simple
    self-replication model.
    """
    return (lam / (1 - phiRb - phi_O)) * (((lam * Kd)/(gamma_max * phiRb - lam)) + 1)

def estimate_nu_FPM(phiRb, 
                    lam, 
                    const, 
                    phi_O, 
                    nu_buffer=1, 
                    dt=0.0001, 
                    tol=2, 
                    guess=False,
                    verbose=False):
    """
    Integrates the FPM model to find the metabolic rate which yields a given 
    growth rate. 

    Parameters
    ----------
    phiRb: float, [0, 1)
        The desired allocation parameter towards ribosomal proteins
    lam: float [0, inf)
        The desired steady-state growth rate in units of invers time.
    const: dict
        A dictionary of model constants to be used.
    phi_O : float [0, 1)
        The allocation parameter towards other proteins.
    nu_buffer: int
        After estimating the metabolic rate under the simple model, a new 
        range of metabolic rates is defined with bounds of +/- nu_buffer. If 
        nu_buffer - 1 < 0, a value of 0.00001 is used.
    dt : float
        The timestep over which to integrate.
    tol : 2
        The decimal tolerance in difference between desired and realized growth 
        rate.
    guess : float [0, inf)
        Your best guess at finding nu. If not provided, the optimal allocation 
        will be used to estimate it.
    verbose: bool
        If True, progess will be pushed to console.

    Returns
    -------
    nu: float, [0, inf)
        Returns the metabolic rate which yields a growth rate with the 
        tolerance. If the tolerance is not met, the closest value will be
        returned.
    """
    if guess == False:
        nu = compute_nu(const['gamma_max'], const['Kd_cpc'], phiRb, lam, phi_O)
    else:
        nu = guess
    lower = nu - nu_buffer
    if lower <= 0:
        lower = 0.001
    upper = nu + nu_buffer
    nu_range = np.linspace(lower, upper, 400)
    converged = False
    ind = 0 
    diffs = []
    if verbose:
        iterator = enumerate(tqdm.tqdm(nu_range))
    else:
        iterator = enumerate(nu_range)
    for _, n in iterator:
       args = {'gamma_max': const['gamma_max'],
               'nu_max': n,
               'tau': const['tau'],
               'Kd_TAA': const['Kd_TAA'],
               'Kd_TAA_star': const['Kd_TAA_star'],
               'kappa_max':const['kappa_max'], 
               'phi_O':  phi_O}

       out = equilibrate_FPM(args, dt=dt, tol=tol, t_return=2) 
       gr = np.log(out[1][0] / out[0][0]) / dt
       diff = np.round(gr / lam, decimals=tol)
       diffs.append(diff)

       if diff == 1: 
          converged = True
          break
    if converged:
        return n
    else:
        diffs = np.array(diffs)
        print('Metabolic rate not found over range. Try rerunning over a larger range.')
        return nu_range[np.argmin(np.abs(diffs - 1))]



def nutrient_shift_FPM(args,
                       shift_time=2,
                       total_time=10,
                       dt=0.001):
    """
    Performs a simple nutrient upshift under flux-parity allocation given 
    arguments for integration. 

    Parameters 
    -----------
    args: list of dictionaries
        A list of dictionaries that are passed to the `self_replicator_FPM`
        function in the `model` submodule. 
    shift_time : float
        Time at which the shift whould be applied.
    total_time : float
        The total time the integration should be run
    dt : float
        The time step for the integration.

    Returns
    -------
    df : pandas DataFrame
        A pandas DataFrame of teh shift with columns corresponding to 
        the total biomass `M`, total ribosomal biomass `M_Rb`, 
        total metabolic biomass `M_Mb`, uncharged-tRNA concentration `TAA`,
        and the charged-tRNA concentration `TAA_star`. Details of the 
        shift and time are also provided as columns.
    """
    cols = ['M', 'M_Rb', 'M_Mb', 'TAA', 'TAA_star']

    # Set the timespans
    preshift_time = np.arange(0, shift_time, dt)
    postshift_time = np.arange(shift_time - dt, total_time, dt)

    # Equilibrate
    preshift_out = equilibrate_FPM(args[0])
    eq_phiRb_preshift = preshift_out[1] / preshift_out[0]
    eq_phiMb_preshift = preshift_out[2] / preshift_out[0]
    eq_TAA_preshift = preshift_out[-2]
    eq_TAA_star_preshift = preshift_out[-1]

    # Pack the params   
    M0 = 1
    init_params = [M0, 
                   M0 * eq_phiRb_preshift,  
                   M0 * eq_phiMb_preshift, 
                   eq_TAA_preshift, 
                   eq_TAA_star_preshift]
    init_args = (args[0],)
    shift_args =(args[1],)

    # Integrate the shifts
    preshift_out = scipy.integrate.odeint(self_replicator_FPM,
                                          init_params, 
                                          preshift_time, 
                                          args=init_args)
    shift_params = preshift_out[-1] 
    postshift_out = scipy.integrate.odeint(self_replicator_FPM,
                                          shift_params, 
                                          postshift_time, 
                                          args=shift_args)
    postshift_out = postshift_out[1:]

    # Form dataframes
    preshift_df = pd.DataFrame(preshift_out, columns=cols)
    preshift_df['time'] = preshift_time
    postshift_df = pd.DataFrame(postshift_out, columns=cols)
    postshift_df['time'] = postshift_time[1:]
    df = pd.concat([preshift_df, postshift_df], sort=False)
    df['shifted_time'] = df['time'].values - shift_time
    return df
 