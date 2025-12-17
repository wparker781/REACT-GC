# Re-import after kernel reset
# Standard scientific Python imports and utilities
from unicodedata import normalize
import numpy as np
from traj_predict_tle import get_data_from_tles
import matplotlib.pyplot as plt

# Configure matplotlib to use Arial for all figures
plt.rcParams['font.family'] = 'Arial'


def main():
    """
    Main driver routine.

    This function:
    1. Loads training and test data derived from TLEs
    2. Evaluates how prediction uncertainty changes as more support satellites are included
    3. Produces diagnostic plots of prediction mean and uncertainty
    4. Overlays space weather drivers for physical context
    """

    # Toggle for plotting normalized quantities versus raw values
    plot_normalized = False  # plot normalized y or actual?

    # Load data derived from TLE processing
    # This includes satellite responses, space weather drivers,
    # normalization statistics, and time vectors
    (
        x_train, y_train,
        x_test, y_test,
        satcat, alt_by_obj, dadt,
        t_train, t_test,
        dalt_mean, dalt_std,
        norm_factor_test,
        f10_mean, f10_std,
        ap_mean, ap_std
    ) = get_data_from_tles()

    # Index of the satellite being predicted (the "query" satellite)
    query_idx = 0

    # Lists of support satellite indices to test
    # Each entry corresponds to progressively increasing the number
    # of reference satellites used for conditioning
    support_idx_vec = [
        [],
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ]

    # Space weather inputs used for prediction
    query_X = x_test

    # Rescale normalized space weather drivers back to physical units
    f10_scaled = x_test[:, 0] * f10_std + f10_mean
    ap_scaled = x_test[:, 1] * ap_std + ap_mean

    # ------------------------------------------------------------------
    # Study how prediction variance changes with number of support sats
    # ------------------------------------------------------------------
    plt.figure(figsize=(5, 4))

    var_theory_lst = []    # Variance from Gaussian conditioning formula
    var_practice_lst = []  # Variance estimated from residuals on training data

    for i in range(len(support_idx_vec)):
        support_idx = support_idx_vec[i]

        # Extract support satellite time series
        support_Y_test = y_test[:, support_idx]
        support_Y_train = y_train[:, support_idx]

        # Compute theoretical vs empirical uncertainty
        var_theory, var_practice = get_unc_theory_practice(
            x_train, y_train,
            x_test, y_test,
            query_X, query_idx,
            support_Y_test, support_Y_train,
            support_idx
        )

        var_theory_lst.append(var_theory)
        var_practice_lst.append(var_practice)

    # Plot variance as a function of number of support satellites
    plt.plot(
        range(len(support_idx_vec)),
        var_theory_lst,
        'o-',
        color='firebrick',
        alpha=0.6,
        label='Theoretical'
    )
    plt.xlabel('Number of reference satellites')
    plt.ylabel('Variance of prediction')
    plt.title('Variance of prediction vs number of support satellites')
    plt.grid(axis='y', color='lightgray', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Plot predicted time series and uncertainty envelopes
    # ------------------------------------------------------------------
    plt.figure(1, figsize=(4, 8))

    for i in range(len(support_idx_vec)):
        support_idx = support_idx_vec[i]

        support_Y_test = y_test[:, support_idx]
        support_Y_train = y_train[:, support_idx]

        # Predict mean and uncertainty for the query satellite
        mu_pred, std_pred = model_with_unc(
            x_train, y_train,
            x_test,
            query_X, query_idx,
            support_Y_test, support_Y_train,
            support_idx
        )

        if plot_normalized == True:
            # Plot raw normalized values
            y_test_trans = y_test[:, query_idx]
        else:
            # Transform predictions back to physical units
            mu_pred = (mu_pred * dalt_std[query_idx] + dalt_mean[query_idx]) * norm_factor_test[:, query_idx]
            std_pred = (std_pred * dalt_std[query_idx] + dalt_mean[query_idx]) * norm_factor_test[:, query_idx]
            y_test_trans = (y_test[:, query_idx] * dalt_std[query_idx] + dalt_mean[query_idx]) * norm_factor_test[:, query_idx]

        plt.figure(1)
        plt.subplot(len(support_idx_vec) + 1, 1, i + 1)
        plt.grid(axis='y', color='lightgray', linewidth=0.5)

        # Shade training interval
        plt.fill_between(t_train, -10, 5, color='gainsboro', alpha=0.5)

        # True vs predicted response
        plt.plot(t_test, y_test_trans, 'k', label='True', linewidth=1)
        plt.plot(t_test, mu_pred, 'r', label='Predicted', linewidth=1)

        # Plot ±2σ uncertainty envelope
        plt.fill_between(
            t_test,
            mu_pred - 2 * std_pred,
            mu_pred + 2 * std_pred,
            color='r',
            alpha=0.2
        )

        if plot_normalized == True: 
            plt.ylim([-10, 5])
            plt.ylabel(r'$d_{sn}$')

        else: 
            plt.ylim([-1.3e-5, 0.3e-5])
            plt.ylabel(r'\dot{a} [km/s]')


        plt.gca().set_xticklabels([])

    # ------------------------------------------------------------------
    # Plot space weather drivers beneath prediction panels
    # ------------------------------------------------------------------
    plt.figure(1)
    plt.subplot(len(support_idx_vec) + 1, 1, len(support_idx_vec) + 1)

    # F10.7 solar flux
    plt.plot(
        t_test, f10_scaled,
        'darkorange',
        label='F10',
        alpha=0.8,
        linewidth=1.5
    )
    plt.ylabel('F10.7 [sfu]', color='darkorange')
    plt.gca().tick_params(axis='y', labelcolor='darkorange')
    plt.gca().set_ylim([100, 426])
    plt.xticks(rotation=45)
    plt.fill_between(t_train, 100, 426, color='gainsboro', alpha=0.5)
    plt.xlabel('Date')

    # Geomagnetic activity index (ap) on secondary axis
    ax2 = plt.gca().twinx()
    ax2.plot(
        t_test, ap_scaled,
        'teal',
        label='ap',
        alpha=0.8,
        linewidth=1.5
    )
    ax2.tick_params(axis='y', labelcolor='teal')
    ax2.set_ylabel('ap', color='teal')

    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.show()


def get_unc_theory_practice(
    x_train, y_train,
    x_test, y_test,
    query_X, query_idx,
    support_Y_test, support_Y_train,
    support_idx
):
    """
    Compare theoretical uncertainty from Gaussian conditioning
    against empirical uncertainty estimated from training residuals.
    """

    # Compute theoretical posterior variance using training data
    mean_theory, var_theory = predict_satellite_response_v2(
        y_train, x_train,
        x_train,
        support_Y_test,
        query_idx,
        support_idx
    )

    # Residuals between predicted mean and true training values
    resid = mean_theory - y_train[:, query_idx]

    # Empirical variance of residuals
    var_practice = np.var(resid)

    return var_theory[0], var_practice


def model_with_unc(
    x_train, y_train,
    x_test,
    query_X, query_idx,
    support_Y_test, support_Y_train,
    support_idx
):
    """
    Predict mean response and estimate uncertainty by learning a
    mapping from space weather inputs to squared residuals.
    """

    # Predict mean and theoretical variance for test period
    mu_pred, var_pred = predict_satellite_response_v2(
        y_train, x_train,
        query_X,
        support_Y_test,
        query_idx,
        support_idx
    )

    # Predict on training data to assess residual structure
    mu_train, var_train = predict_satellite_response_v2(
        y_train, x_train,
        x_train,
        support_Y_train,
        query_idx,
        support_idx
    )

    # Compute squared residuals on training set
    resid = mu_train - y_train[:, query_idx]
    sq_resid = resid ** 2

    # Normalize squared residuals
    sq_resid_n = (sq_resid - np.mean(sq_resid)) / np.std(sq_resid)

    # Fit linear mapping from space weather inputs to residual variance
    W = np.linalg.lstsq(x_train, sq_resid_n, rcond=None)[0]

    # Predict variance for test inputs
    var_pred2_n = x_test @ W
    var_pred2 = var_pred2_n * np.std(sq_resid) + np.mean(sq_resid)

    # Convert variance to standard deviation
    var_pred2 = np.maximum(var_pred2, 0.0) # clip to prevent negatives
    std_pred = np.sqrt(var_pred2)

    # Replace invalid values with zero
    std_pred[np.isnan(std_pred)] = 0

    return mu_pred, std_pred


def predict_satellite_response_v2(
    Y_train, X_train,
    query_X_series,
    support_Y_series,
    query_idx,
    support_idx
):
    """
    Predict the time series response of a query satellite using
    multivariate Gaussian conditioning on:
      - Support satellite responses
      - Space weather drivers
    """

    # Transpose data so rows correspond to variables
    Y_train_T = Y_train.T   # (n_sats, t_train)
    X_train_T = X_train.T   # (n_drivers, t_train)

    n_sats, t_train = Y_train_T.shape
    n_drivers = X_train_T.shape[0]
    t_pred = query_X_series.shape[0]

    # Stack satellite responses and drivers into a joint state vector
    YX_train = np.vstack([Y_train_T, X_train_T])

    # Compute mean and covariance of joint distribution
    mu_train = np.mean(YX_train, axis=1, keepdims=True)
    YX_centered = YX_train - mu_train
    C = np.cov(YX_centered)

    # Index bookkeeping
    query_full_idx = query_idx
    support_full_idx = support_idx
    X_full_idx = list(range(n_sats, n_sats + n_drivers))
    known_idx = support_full_idx + X_full_idx

    # Partition covariance matrix
    C_UU = C[query_full_idx, query_full_idx]
    C_Uknown = C[query_full_idx, known_idx].reshape(1, -1)
    C_knownU = C_Uknown.T
    C_known = C[np.ix_(known_idx, known_idx)]

    # Means of known and unknown components
    mu_SX = mu_train[known_idx].flatten()
    mu_U = mu_train[query_full_idx, 0]

    mu_preds = np.zeros(t_pred)
    var_preds = np.zeros(t_pred)

    # Time-stepping Gaussian conditioning
    for t in range(t_pred):
        support_Y_t = support_Y_series[t]
        query_X_t = query_X_series[t]

        known_vals = np.hstack([support_Y_t, query_X_t])
        delta = known_vals - mu_SX

        mu_post = mu_U + C_Uknown @ np.linalg.inv(C_known) @ delta
        var_post = C_UU - C_Uknown @ np.linalg.inv(C_known) @ C_knownU

        mu_preds[t] = mu_post[0]
        var_preds[t] = var_post[0,0]

    return mu_preds, var_preds


if __name__ == "__main__":
    main()
