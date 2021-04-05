import numpy as np 

def single_nutrient_dynamics(params, time, gamma_max, nu_max, precursor_mass_ref, 
                             Km, omega, phi_R, phi_P, num_muts=1, volume=1E-3):
    """
    Defines the system of ordinary differenetial equations (ODEs) which describe 
    accumulation of biomass on a single nutrient source. 

    Parameters
    ----------

    params: list, [M, Mr, Mp, precursors, nutrients]
        A list of the parameters whose dynamics are described by the ODEs.
        M : positive float
            Total protein biomass of the system
        Mr : positive float, must be < M 
            Ribosomal protein biomass of the system
        Mp : positive float, must be < M
            Metabbolic protein biomass of the system 
        precursors : positive float
            Mass of precursors in the cell. This is normalized to 
            total protein biomass when calculating the translational 
            capacity.
        nutrients  : positive float
            Mass of nutrients in the system.
    time : float
        Evaluated time step of the system.
    gamma_max: positive float 
        The maximum translational capacity in units of inverse time.
    nu_max : positive float
        The maximum nutritional capacity in units of inverse time. 
    precursor_conc_ref : positive float 
        The dissociation constant of charged tRNA to the elongating ribosome.   
    Km : positive float
        The Monod constant for growth on the specific nutrient source. 
        This is in units of molar.
    omega: positive float
        The yield coefficient of the nutrient source in mass of amino acid 
        produced per mass of nutrient.
    phi_R : float, [0, 1]
        The fraction of the proteome occupied by ribosomal protein mass
    phi_P : float, [0, 1] 
        The fraction of the proteome occupied by metabolic protein mass
    num_muts: int
        The number of mutants whose dynamics need to be tracked.
    volume: float, default 1 mL
        The volume of the system for calculation of concentrations.

    Returns
    -------
    out: list, [dM_dt, dMr_dt, dMp_dt, dprecursors_dt, dnutrients_dt]
        A list of the evaluated ODEs at the specified time step.

        dM_dt : The dynamics of the total protein biomass.
        dMr_dt : The dynamics of the ribosomal protein biomass.
        dMp_dt : the dynamics of the metabolic protein biomass.
        dprecursors_dt : The dynamics of the precursor/charged-tRNA pool.
        dnutrients_dt :  The dynamics of the nutrients in the growth medium
    """
    # Define constants 
    AVO = 6.022E23
    OD_CONV = 6E17
    #TODO: Put in data validation
        
    # Unpack the parameters
    if num_muts > 1:
        nutrients = params[-1]
        M, Mr, Mp, precursors = np.reshape(params[:-1], (4, num_muts))
    else: 
        M, Mr, Mp, precursors, nutrients = params

    # Compute the precursor mass fraction and nutrient concentration
    precursor_mass_frac = precursors / M
    nutrient_conc = nutrients / (AVO * volume)

    # Compute the two capacities
    gamma = gamma_max * precursor_mass_frac / (precursor_mass_frac + precursor_mass_ref)
    nu = nu_max * nutrient_conc / (nutrient_conc + Km)

    # ODEs for biomass accumulation
    dM_dt = gamma * Mr
    dMr_dt = phi_R * dM_dt
    dMp_dt = phi_P * dM_dt
    # ODE for precursors and nutrients
    dprecursors_dt = nu * Mp - dM_dt
    dnutrients_dt = -nu * Mp/ omega

    _out = [dM_dt, dMr_dt, dMp_dt, dprecursors_dt]
    if num_muts > 1:
        dnutrients_dt = np.sum(dnutrients_dt)
        out = [value for deriv in _out for value in deriv]
    out.append(dnutrients_dt)
    return out


def growth_rate(nu_max, gamma_max, phi_R, phi_P, Kd, f_a=0.9):
    term_a = Kd - 1
    term_b = nu_max * phi_P + gamma_max * phi_R * f_a
    term_c = nu_max * phi_P * gamma_max * phi_R * f_a
    return (-term_b + np.sqrt(term_b**2 + 4 * term_a * term_c)) / (2 * term_a)

def tRNA_balance(nu_max, phi_P, growth_rate):
    return (nu_max * phi_P / growth_rate) - 1

def translation_rate(gamma_max, c_AA, Kd):
    return gamma_max * c_AA  / (c_AA + Kd)

def optimal_phi_R(gamma_max, nu_max, Kd, phi_O, f_a=0.9):
    term_a = phi_O - 1
    term_b = -nu_max * (-2 * Kd * gamma_max * f_a  + gamma_max * f_a + nu_max)
    term_c = np.sqrt(Kd * gamma_max * f_a * nu_max) * (-gamma_max * f_a + nu_max)
    denom = -4 * Kd * gamma_max * f_a * nu_max + (f_a * gamma_max)**2 + 2 * gamma_max * f_a * nu_max + nu_max**2
    return (term_a * (term_b + term_c)) / denom
