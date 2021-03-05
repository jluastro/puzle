import numpy as np
import matplotlib.pyplot as plt
from puzle.stats import calculate_eta


def calculate_eta_arr(size, sigma, N_samples=20000):
    eta_arr = []
    for i in range(N_samples):
        sample = np.random.normal(0, sigma, size=size)
        eta = calculate_eta(sample)
        eta_arr.append(eta)
    print('size: %i | sigma: %.2f' % (size, sigma))
    print('-- median(eta): %.2f' % np.median(eta_arr))
    print('-- std(eta): %.2f' % np.std(eta_arr))
    print('-- thresh(eta): %.2f' % np.percentile(eta_arr, 1))
    return eta_arr


def generate_eta_gaussian_dist_plot():
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    for a in ax:
        a.clear()
    for size in np.arange(25, 226, 50):
        eta_arr = calculate_eta_arr(size=size, sigma=1)
        ax[0].hist(eta_arr, histtype='step', label='Size = %i' % int(size))
    ax[0].set_title('Size Sampling')
    ax[0].legend()
    for sigma in np.arange(0.5, 2.6, 0.5):
        eta_arr = calculate_eta_arr(size=100, sigma=sigma)
        ax[1].hist(eta_arr, histtype='step', label='Sigma = %.2f' % sigma)
    ax[1].set_title('Sigma Sampling')
    ax[1].legend()
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    fig.suptitle('eta Gaussian Distributions', fontsize=12)


def generate_eta_gaussian_threshold_plot():
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    for a in ax:
        a.clear()
    size_arr = np.logspace(np.log10(20), np.log10(2000), 100).astype(int)
    eta_thresh_arr = []
    for size in size_arr:
        eta_arr = calculate_eta_arr(size=size, sigma=1)
        eta_thresh_arr.append(np.percentile(eta_arr, 1))
    eta_thresh_arr = np.array(eta_thresh_arr)
    ax[0].plot(np.log10(size_arr), eta_thresh_arr, marker='.', color='k', ms=5)
    cond = size_arr < 120
    m, b = np.polyfit(np.log10(size_arr[cond]), eta_thresh_arr[cond], deg=1)
    ax[0].plot(np.log10(size_arr), np.log10(size_arr)*m+b, color='g',
               label='1st Log Fit', linewidth=5, alpha=.5)
    m, b = np.polyfit(np.log10(size_arr[~cond]), eta_thresh_arr[~cond], deg=1)
    ax[0].plot(np.log10(size_arr), np.log10(size_arr)*m+b, color='r',
               label='2nd Log Fit', linewidth=5, alpha=.5)
    ax[0].set_xlabel('Log(Sample Size)', fontsize=12)
    ax[0].set_ylabel('Eta Threshold', fontsize=12)
    ax[0].set_title('(sigma = 1)', fontsize=8)
    ax[0].legend()
    sigma_arr = np.arange(0.5, 2.6, 0.05)
    eta_thresh_arr = []
    for sigma in sigma_arr:
        eta_arr = calculate_eta_arr(size=100, sigma=sigma)
        eta_thresh_arr.append(np.percentile(eta_arr, 1))
    ax[1].plot(sigma_arr, eta_thresh_arr, marker='.')
    ax[1].set_xlabel('Sample Sigma', fontsize=12)
    ax[1].set_ylabel('Eta Threshold', fontsize=12)
    ax[1].set_title('(size = 100)', fontsize=8)
    ax[1].set_ylim(ax[0].get_ylim())
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    fig.suptitle('eta Gaussian Thresholds', fontsize=12)