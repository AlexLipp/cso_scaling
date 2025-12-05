"""Module to demonstrate how to fit heavy-tailed distributions to water company EDM duration data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw


class WCDurationDistribution:
    """
    Class that explores the distribution of durations in water company EDM data.

    This class is designed to work with the water company EDM data monitoring dataset.
    Automatically rounds the durations to 15 minute intervals and drops any durations
    equal to 0. Fits the duration data to a power law, lognormal, exponential, and
    stretched exponential distribution using the powerlaw package.

    Parameters
    ----------
    durations (pd.Series): Series of durations to be analyzed.
    name (str): Name of the dataset for labeling purposes.

    Methods
    -------
    fit_distributions(plot: bool = True) -> None:
        Fits the duration data to a power law, lognormal, exponential, and stretched
        exponential distribution. Plots the results if plot is True.
    bootstrap_beta(n_bootstraps: int = 100, n_jobs: int = None) -> Tuple[float, float]:
        Performs bootstrap analysis on the stretched exponential beta parameter.
    compare_distributions(fit: powerlaw.Fit) -> None:
        Compares distributions and prints the parameters of each fit.
    print_results(fit: powerlaw.Fit) -> None:
        Prints the results of the fitted distributions.

    Attributes
    ----------
    durations_raw (pd.Series): Original series of durations.
    durations (pd.Series): Rounded series of durations.
    name (str): Name of the dataset.
    results (powerlaw.Fit): Fitted powerlaw results.

    """

    def __init__(self, durations: pd.Series, name: str) -> None:
        """Initialize the class with a series of durations."""
        # Check that durations is a pandas Series which has header "Duration"
        if not isinstance(durations, pd.Series):
            raise TypeError("Durations must be a pandas Series.")
        if durations.name != "Duration":
            raise ValueError("Durations must have header 'Duration'.")
        self.durations_raw = durations
        # Now round the durations up to 15 minute intervals
        self.durations = self.durations_raw.copy()
        self.durations = np.ceil(self.durations_raw / 15) * 15
        # Drop values equal to 0
        # Count number of zeros
        num_zeros = (self.durations == 0).sum()
        print(f"Number of zero durations being dropped: {num_zeros}")
        self.durations = self.durations[self.durations > 0]
        print(f"Total number of durations: {len(self.durations)}")
        self.name = name

    def fit_distributions(self, plot: bool = True) -> None:
        """Fit the duration data to diverse heavy tailed distributions."""
        # Fit the data to a power law
        results = powerlaw.Fit(
            self.durations,
            discrete=False,
            xmin=120,
            xmax=None,  # np.amax(self.durations)
        )
        # Print the xmin and xmax values
        print(f"xmin: {results.xmin}")
        print(f"xmax: {results.xmax}")
        # Fit the data to a lognormal distribution
        results.lognormal.fit()
        # Fit the data to an exponential distribution
        results.stretched_exponential.fit()
        # Plot the results
        # Store results (an instance of powerlaw.Fit as an attribute in the class
        self.results = results
        if plot:
            pdf_vals = results.pdf()[1]
            # Only get the values that are greater than 0
            pdf_vals = pdf_vals[pdf_vals > 0]
            min, max = np.min(pdf_vals), np.max(pdf_vals)
            plt.figure(figsize=(10, 6))
            # Plot the empirical data
            results.plot_pdf(color="k", linestyle="-", label="Data")
            # Add a power law fit in red
            results.power_law.plot_pdf(color="r", linestyle="--", label="Power Law Fit")
            # Add a lognormal fit in green
            results.lognormal.plot_pdf(color="g", linestyle="--", label="Lognormal Fit")
            # Add a stretched exponential fit in purple
            results.stretched_exponential.plot_pdf(
                color="purple", linestyle="--", label="Stretched Exponential Fit"
            )
            # Add an exponential fit in orange
            results.exponential.plot_pdf(
                color="orange", linestyle="--", label="Exponential Fit"
            )
            # Add a legend and labels
            plt.ylim(0.5 * min, 2 * max)
            # Set the x limits to be between 0 and the maximum of
            plt.legend()
            plt.xlabel("Duration [mins]")
            plt.ylabel("Probability Density")
            plt.title(f"Fitting distributions to {self.name}")

    def bootstrap_beta(
        self, n_bootstraps: int = 100, n_jobs: int = None
    ) -> tuple[float, float]:
        """
        Perform bootstrap analysis on the stretched exponential beta parameter.

        Parameters
        ----------
        n_bootstraps : int, default 100
            Number of bootstrap samples to generate
        n_jobs : int, default None
            Number of CPU cores to use. If None, uses all available cores.

        Returns
        -------
        tuple[float, float]
            Mean and standard deviation of the bootstrap beta values

        """
        import multiprocessing as mp

        # Check if results exist (need to run fit_distributions first)
        if not hasattr(self, "results"):
            raise ValueError(
                "Must run fit_distributions() first before bootstrap analysis"
            )

        # Set number of jobs
        if n_jobs is None:
            n_jobs = mp.cpu_count()

        print(f"Running {n_bootstraps} bootstrap samples using {n_jobs} CPU cores...")

        # Prepare arguments for parallel processing
        args = [(self.durations, i) for i in range(n_bootstraps)]

        # Use multiprocessing to parallelize bootstrap
        with mp.Pool(processes=n_jobs) as pool:
            betas = pool.map(_bootstrap_fit, args)

        mean_beta = np.mean(betas)
        std_beta = np.std(betas)

        print(f"Bootstrap results: Mean = {mean_beta:.4f}, Std Dev = {std_beta:.4f}")

        # Store bootstrap results as attributes
        self.bootstrap_betas = betas
        self.bootstrap_mean = mean_beta
        self.bootstrap_std = std_beta

        return mean_beta, std_beta


def _bootstrap_fit(args):
    """Fit a single bootstrap sample."""
    durations, seed = args
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Resample with replacement
    resampled = durations.sample(frac=1, replace=True)
    results = powerlaw.Fit(
        resampled, discrete=False, xmin=120, xmax=None  # np.amax(resampled)
    )
    results.stretched_exponential.fit()
    return results.stretched_exponential.beta


def compare_distributions(fit: powerlaw.Fit) -> None:
    """
    Compare distributions and prints the parameters of each fit.

    R is the log-likelihood ratio statistic and p is the p-value.
    Negative R indicates that the second distribution is a better fit than
    the first, with a p-value indicating the significance of the difference.
    """
    print("#" * 50)
    r_ln, p_ln = fit.distribution_compare(
        "power_law", "lognormal", normalized_ratio=True
    )
    print(f"Powerlaw vs Lognormal: R = {r_ln}, p = {p_ln}")
    # Say which distribution is better
    if r_ln < 0:
        print("Lognormal is a better fit than Power Law.")
    else:
        print("Power Law is a better fit than Lognormal.")
    if p_ln < 0.05:
        print("The difference is statistically significant.")
    print("#" * 50)
    r_exp, p_exp = fit.distribution_compare(
        "power_law", "exponential", normalized_ratio=True
    )
    print(f"Power Law vs Exponential: R = {r_exp}, p = {p_exp}")

    # Say which distribution is better
    if r_exp < 0:
        print("Exponential is a better fit than Power Law.")
    else:
        print("Power Law is a better fit than Exponential.")
    if p_exp < 0.05:
        print("The difference is statistically significant.")

    print("#" * 50)
    r_stretch, p_stretch = fit.distribution_compare(
        "power_law", "stretched_exponential", normalized_ratio=True
    )
    print(f"Power Law vs Stretched Exponential: R = {r_stretch}, p = {p_stretch}")
    # Say which distribution is better
    if r_stretch < 0:
        print("Stretched Exponential is a better fit than Power Law.")
    else:
        print("Power Law is a better fit than Stretched Exponential.")
    if p_stretch < 0.05:
        print("The difference is statistically significant.")
    print("#" * 50)

    # Compare stretched exponential and lognormal
    r_stretch_ln, p_stretch_ln = fit.distribution_compare(
        "stretched_exponential", "lognormal", normalized_ratio=True
    )
    print(f"Stretched Exponential vs Lognormal: R = {r_stretch_ln}, p = {p_stretch_ln}")
    # Say which distribution is better
    if r_stretch_ln < 0:
        print("Lognormal is a better fit than Stretched Exponential.")
    else:
        print("Stretched Exponential is a better fit than Lognormal.")
    if p_stretch_ln < 0.05:
        print("The difference is statistically significant.")


def print_results(fit: powerlaw.Fit) -> None:
    """Print the results of the fitted distributions."""
    print(f"Power Law: alpha = {fit.power_law.alpha}, xmin = {fit.power_law.xmin}")
    print(f"Lognormal: mu = {fit.lognormal.mu}, sigma = {fit.lognormal.sigma}")
    print(f"Exponential: lambda = {fit.exponential.Lambda}")
    print(
        f"Stretched Exponential: beta = {fit.stretched_exponential.beta}, "
        f"lambda = {fit.stretched_exponential.Lambda}"
    )

def main():
    """Run an example fitting water company EDM data."""
    # Load the Southern Water EDM data for 2023 as an example
    southern23 = pd.read_csv("edm_data/southern/Southern23.csv")
    southern23["Start"] = pd.to_datetime(southern23["Start Time"], dayfirst=True)
    southern23["Stop"] = pd.to_datetime(southern23["End Time"], dayfirst=True)
    southern23["Duration"] = (
        southern23["Stop"] - southern23["Start"]
    ).dt.total_seconds() / 60
    southern_dd = WCDurationDistribution(southern23["Duration"], "Southern Water 2023")
    southern_dd.fit_distributions()
    plt.show()
    # Compare distributions
    print("-" * 50)
    compare_distributions(southern_dd.results)
    print("-" * 50)
    print_results(southern_dd.results)

if __name__ == "__main__":
    main()