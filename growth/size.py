import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
def func(x, a, c, d):
    return a*np.exp(-c*x)+d
#%
# The functions below are used to estimate cell size, surface area, and
# total protein abundance per cell as a function of growth rate. The functions
# for calculate cell length, width, size, and surface area are empirical fits
# based on the measurements of cell length and width for cell strain MG1655
# in Si et al. 2017, 2019.

def lambda2width(x):
    """
    Esimates the cell width based on the growth rate.

    Parameters
    ----------
    x : float
        growth rate (1/hr)

    Returns
    -------
     a*np.exp(-c*x)+d : float
        Cell width \mu m
    """
    a, c = 0.63830175, -0.24341639
    return a*np.exp(-c*x)

def lambda2length(x):
    """
    Esimates the cell length based on the growth rate.

    Parameters
    ----------
    x : float
        growth rate (1/hr)

    Returns
    -------
     a*np.exp(-c*x)+d : float
        Cell length \mu m
    """
    a, c, d = 0.49656209, -1.09303027,  1.75967254
    return a*np.exp(-c*x)+d

def lambda2size(x):
    """
    Esimates the cell volume based on the growth rate.

    Parameters
    ----------
    x : float
        growth rate (1/hr)

    Returns
    -------
    a*np.exp(-c*x) : float
        Cell volume in \mu m**3
    """
    a, c = 0.53319063, -1.0372483
    return a*np.exp(-c*x)

def lambda2SV(x):
    """
    Esimates the surface area to volume ratio based on the growth rate.

    Parameters
    ----------
    x : float
        growth rate (1/hr)

    Returns
    -------
    SA_data / V_data : float
        surface area to volume ratio in \mu m**-1
    """
    V_data = lambda2size(x)
    l = lambda2length(x)
    w = lambda2width(x)
    SA_data = rod_SA(l, w, V_data)

    return SA_data / V_data

def rod_SA(l, w, V):
    """
    Computes the total cell surface area; cell is assumed as a cylinder with two
    hemispherical ends.

    Parameters
    ----------
    l : float
        cell length in \mu m
    w : float
        cell width in \mu m
    V : float
        cell volume in \mu m**3

    Returns
    -------
    gamma * V**(2/3) : float
        estimate of surface area in \mu m**2
    """
    asp_ratio = l/w
    gamma = asp_ratio * np.pi * (asp_ratio * np.pi /4 - np.pi/12)**(-2/3)
    return gamma * V**(2/3)

def lambda2SA(x):
    """
    Computes the cellular surface area as a function of the growth rate. It is
    assumed that the cell is cylinder capped with hemispherical ends.

    Parameters
    ----------
    x : int, float, or list/array of ints and floats
        The growth rate in units of hr^-1

    Returns
    -------
    SA: int, float, or list/array of ints and floats
        The computed surface area in units of square microns for the provided
        growth rates
    """

    # Compute the length, width, and volume as a function of growth rate
    length = lambda2length(x)
    width = lambda2width(x)
    vol = lambda2size(x)

    # Compute the rod surface area
    SA = rod_SA(length, width, vol)
    return SA


def lambda2P(x):
    """
    Computes the total protein mass per cell in fg. This is based
    on an assumption that the mass density is relatively constant w.r.t. growth
    rate, and also assumes that the total protein + DNA + RNA account for roughly
    90 percent of the total cell mass (Basan et al. 2015).

    Parameters
    ----------
    x : float
        growth rate (1/hr)

    Returns
    -------
    pred_proteinmass : float
        estimate of total protein mass in fg
    """

    # Data from Hwa lab (Dai et al. 2016, Basan et al. 2015) for RNA/Protein
    # ratios and total DNA content per cell. Note that the protein measurements
    # still have caveat that it is based on a bulk colorimetric assay, but I think
    # this is the best that we can do; I believe the ratio is a pretty reliable
    # result. The data was extracted from these papers:
    [lambda_dai, R_P_dai] = np.array([\
            # [[0.0, 0.08853279947047754],
            # [0.03324706890845652, 0.09742834356027935],
            # [0.12844176066233703, 0.12157153165470358],
            # [0.19652012565308674, 0.12917148174069676],
            # [0.23055930814846148, 0.13297145678369332],
            # [0.2849547247405284, 0.14694954887503597],
            # [0.33601892391911736, 0.15201256530868013],
            # [0.4074068045812377, 0.17107537557577435],
            # [0.417639175984852, 0.16979497279144085],
            # [0.4517000602223341, 0.17104716331103476],
            # [0.485674137491387, 0.18249049192423916],
            # [0.5503561798423366, 0.1888187199227418],
            [0.6727865579409387, 0.21549233114688282],
            [0.6864152519843529, 0.21548365045003987],
            [0.7000547968988209, 0.21420107749149564],
            [0.7170798135820351, 0.21546411888214323],
            [0.744196140345166, 0.2320073568905744],
            [0.9177883754618401, 0.25227895419304786],
            [0.9448830004828637, 0.27136997672488156],
            [0.9926268331190284, 0.2662440252391261],
            [0.9753630972726335, 0.29300661360590724],
            [1.0979236858238794, 0.3043935176896325],
            [1.1624538159800777, 0.32855623735195344],
            [1.2677832212980895, 0.36288405301735593],
            [1.566952587118931, 0.4404005056505911],
            [1.7949076862145108, 0.4784718718295111]]).T

    # DNA measurements from Basan et al. 2015, growth rate (hr-1), dna
    # (femtogram per cell)
    lambda_basan_dna = [#0.42, 0.45, 
                        0.7, 0.98, 1.27, 1.84]
    dna_basan = 1E15*np.divide(1E-6*np.array([#16.5, 14.2, 
                                                14.1, 11.9, 11.3, 11.1]),
                          1E8*np.array([#19.4, 17.1, 
                                16.0, 10.7, 7.93, 3.43]))

    # The RNA/Protein ratios exhibit two linear regimes; fit each to a line.
    # slope_dia_RP_A, intercept_dia_RP_A, r_value, p_value, std_err = stats.linregress(lambda_dai[:14],
    #                                             R_P_dai[:14])
    slope_dia_RP_B, intercept_dia_RP_B, r_value, p_value, std_err = stats.linregress(lambda_dai,
                                                R_P_dai)

    # The DNA shows an exponential scaling; use scipy curve fit function to
    # fit the data to an exponential
    popt_dna, pcov_dan = curve_fit(func, lambda_basan_dna, dna_basan, p0=(1, 1e-6, 1))

    # predict cell size
    V = lambda2size(x)

    # predict total dry mass in fg (1.1 g/ml mass density; 30% dry mass)
    pred_drymass = ((1.1 * 0.3)*(V*1E-12)*1E15)

    # predict total DNA mass
    pred_dnamass =  func(x, *popt_dna)

    # calculate RNA + protein mass (90% dry mass - DNA mass)
    pred_RNA_protein_mass = pred_drymass * 0.90 - pred_dnamass

    #### predict RNA/Protein ratio:
    pred_RNA_protein_ratio = []
    if (isinstance(x, int)) | (isinstance(x, float)) :
        x = [x]

    for l in x:
        # if l <= 0.69:
            # pred_RNA_protein_ratio.append(slope_dia_RP_A * l + intercept_dia_RP_A)
        # else:
            pred_RNA_protein_ratio.append(slope_dia_RP_B * l + intercept_dia_RP_B)

    # calculate total protein mass
    pred_proteinmass =  (pred_RNA_protein_mass/ (1+ np.array(pred_RNA_protein_ratio)))
    # pred_RNAmass = (pred_RNA_protein_ratio * pred_RNA_protein_mass) / (1+ pred_RNA_protein_ratio)

    return pred_proteinmass
