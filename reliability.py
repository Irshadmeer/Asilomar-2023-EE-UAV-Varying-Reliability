import warnings

import numpy as np
from scipy import stats
from scipy import special
from scipy import linalg


def calculate_outage_probabilities(
    user_locs: np.array,
    bs_locs: np.array,
    bs_powers: np.array,
    carrier_freq: float,
    snr_threshold: float,
    los: np.array = None,
):
    """Calculate the outage probabilities for each user

    Calculate the outage probabilities for each user given their positions and
    the power levels of the base stations.


    Parameters
    ----------
    user_locs : np.array
        Array of size :math:`N \\times 3`, where :math:`N` is the number of
        users/UAVs and the columns represent their :math:`(x, y, z)`
        coordinates.

    bs_locs : np.array
        Array of size :math:`B \\times 3`, where :math:`B` is the number of
        base stations and the columns represent their :math:`(x, y, z)`
        coordinates.

    bs_powers : np.array
        Array of size :math:`N \\times B`, where element :math:`p_{nb}`
        indicates the transmit power level of base station :math:`b` to user
        :math:`n`.

    carrier_freq : float
        Carrier frequency in GHz.

    snr_threshold : float
        Receive power threshold below which an outage occurs.

    los : np.array, optional
        Boolean array of size :math:`N \\times B` indicates whether a
        line-of-sight (LoS) connection is established.


    Returns
    -------
    outage_probabilities : list
        List of length :math:`N` that contains the outage probabilities for the
        individual users.
    """
    bs_locs = np.array(bs_locs)
    num_users, num_bs = np.shape(bs_powers)
    d_3d = np.linalg.norm(
        np.expand_dims(user_locs, 1) - np.expand_dims(bs_locs, 0), axis=2
    )  # N x B

    _offset = 10 ** (-92.45 / 10)
    scale_parameters_los = _offset / (carrier_freq * d_3d / 1000) ** 2
    bs_heights = bs_locs[:, 2]
    _offset_nlos = 10 ** (-32.4 / 10)
    _scale_nlos = _offset_nlos / (
        carrier_freq**2 * d_3d ** (4.32 - 0.76 * np.log10(bs_heights))
    )
    scale_parameters_nlos = np.minimum(scale_parameters_los, _scale_nlos)

    if los is None:
        los = np.ones_like(bs_powers)
    scale_parameters = np.where(los == 1, scale_parameters_los, scale_parameters_nlos)
    scale_parameters = bs_powers * scale_parameters  # N x B

    cdf = []
    for user_params in scale_parameters:
        num_summands = len(user_params)
        user_rate_params = 1.0 / user_params
        _matrix = np.tile(user_rate_params, (num_summands, 1))
        _diff_matrix = _matrix - _matrix.T
        with np.errstate(divide="ignore"):
            _ai_matrix = np.log(_matrix) - np.log(_diff_matrix + 0j)
        np.fill_diagonal(_ai_matrix, 0)
        const_ai = np.sum(_ai_matrix, axis=1)
        log_sf = special.logsumexp(const_ai - user_rate_params * snr_threshold)
        log_sf = np.real(log_sf)
        # print(log_sf)
        _cdf = 1.0 - np.exp(log_sf)
        cdf.append(_cdf)
    cdf = np.maximum(cdf, np.finfo(float).eps)
    return np.ravel(cdf)


if __name__ == "__main__":
    # uav_locations = [[10, 10, 50],
    #                 [50, 50, 10],
    #                 [80, 80, 30]]
    # bs_locations = [[0, 0, 50],
    #                [100, 100, 25]]
    # bs_powers_db = [[10, 5],
    #                #[4, 0],
    #                [7, 5],
    #                [1, 30]]
    uav_locations = [
        [10, 10, 50],
        [50, 50, 10],
        [80, 80, 30],
        [250, 125, 15],
        [327, 234, 60],
    ]
    bs_locations = [[100, 100, 50], [100, 400, 25], [400, 400, 15], [400, 100, 30]]
    bs_powers_db = [
        [10, 1, -2, -1],
        [2, 1, -2, 0],
        [4, 10, 10, 15],
        [10, 0, 4, 2],
        [-2, 3, 5, 6],
    ]
    los = np.random.randint(0, 2, size=np.shape(bs_powers_db))
    print(los)
    bs_powers = 10 ** (np.array(bs_powers_db) / 10)
    freq = 2.4
    # snr_threshold_db = -70
    # for snr_threshold_db in [-110, -100, -90, -80, -70]:
    # for snr_threshold_db in [-70, -50]:
    for snr_threshold_db in [-120, -90, -88, -86, -84, -82, -80]:
        print(f"SNR threshold: {snr_threshold_db:.1f}dB")
        snr_threshold = 10 ** (snr_threshold_db / 10)
        # path_loss = calculate_path_loss(uav_locations, bs_locations, freq)
        # print(path_loss)
        outage_prob = calculate_outage_probabilities(
            uav_locations, bs_locations, bs_powers, freq, snr_threshold, los=los
        )
        print(outage_prob)
