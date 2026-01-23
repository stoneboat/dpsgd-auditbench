import numpy as np

def sample_gaussian(sequence_length, num_samples, sigma, rng):
    return rng.normal(loc=0.0, scale=sigma, size=(num_samples, sequence_length))

def sample_mixture(sequence_length, num_samples, q, mu, sigma, rng):
    # With prob q: N(mu, sigma^2); else: N(0, sigma^2)
    indicators = rng.random((num_samples, sequence_length)) < q
    means = np.where(indicators, mu, 0.0)
    samples = rng.normal(loc=means, scale=sigma, size=(num_samples, sequence_length))
    
    return samples, indicators

class ThresholdAuditor:
    """
    Threshold-based auditor for 1D score samples:
      out_scores ~ P (null; 'no record')
      in_scores  ~ Q (alt;  'with record')

    Threshold classifier: predict 1 iff score >= t.
      alpha(t) = P_out[score >= t]  (FPR)
      beta(t)  = P_in [score < t]  (FNR)
    """

    def __init__(self, in_scores, out_scores):
        self.in_scores = np.asarray(in_scores, dtype=float).ravel()
        self.out_scores = np.asarray(out_scores, dtype=float).ravel()

        self._curve_cache = None

    @staticmethod
    def _smooth_rate(k, n, smoothing):
        """
        Convert counts k/n to a probability.
        smoothing:
          None        -> k/n (may be 0 or 1)
          'jeffreys'  -> (k+0.5)/(n+1)
          'laplace'   -> (k+1)/(n+2)
          float a     -> (k+a)/(n+2a)
        """
        k = np.asarray(k, dtype=float)
        n = float(n)

        if smoothing is None:
            return k / n

        if smoothing == "jeffreys":
            a = 0.5
        elif smoothing == "laplace":
            a = 1.0
        elif isinstance(smoothing, (int, float)):
            a = float(smoothing)
            if a <= 0:
                raise ValueError("If smoothing is numeric, it must be > 0.")
        else:
            raise ValueError("smoothing must be None, 'jeffreys', 'laplace', or a positive float.")

        return (k + a) / (n + 2.0 * a)

    def alpha_beta_curve(self, smoothing="jeffreys"):
        """
        Returns a dict with:
          thresholds (ascending),
          alpha, beta, tpr,
          k_out_ge, k_in_ge, n_out, n_in

        smoothing controls how alpha/tpr are estimated from the counts.
        """
        # Cache is only valid for the default smoothing; keep it simple:
        if self._curve_cache is not None and smoothing == self._curve_cache.get("_smoothing", None):
            return self._curve_cache

        out_sorted = np.sort(self.out_scores)  # ascending
        in_sorted  = np.sort(self.in_scores)   # ascending
        n_out = out_sorted.size
        n_in  = in_sorted.size

        uniq = np.unique(np.concatenate([out_sorted, in_sorted]))
        thresholds = np.concatenate(([-np.inf], uniq, [np.inf]))  # ascending

        # counts >= t
        out_left = np.searchsorted(out_sorted, thresholds, side="left")
        in_left  = np.searchsorted(in_sorted,  thresholds, side="left")
        k_out_ge = n_out - out_left
        k_in_ge  = n_in  - in_left

        alpha = self._smooth_rate(k_out_ge, n_out, smoothing)  # P(S)
        tpr   = self._smooth_rate(k_in_ge,  n_in,  smoothing)  # Q(S)
        beta  = 1.0 - tpr

        curve = {
            "_smoothing": smoothing,
            "thresholds": thresholds,
            "alpha": alpha,
            "beta": beta,
            "tpr": tpr,
            "k_out_ge": k_out_ge,
            "k_in_ge": k_in_ge,
            "n_out": n_out,
            "n_in": n_in,
        }
        self._curve_cache = curve
        return curve

    @staticmethod
    def _log_ratio(num, den):
        """
        Compute log(num/den) elementwise, returning -inf when num<=0 or den<=0.
        No divide-by-zero warnings.
        """
        num = np.asarray(num, dtype=float)
        den = np.asarray(den, dtype=float)
        out = np.full_like(num, -np.inf, dtype=float)
        ok = (num > 0) & (den > 0)
        out[ok] = np.log(num[ok]) - np.log(den[ok])
        return out

    def epsilon_one_run(self, delta, smoothing="jeffreys", return_details=False):
        """
        Sweep all thresholds and return:
          eps_lb = max_t eps_required(t)

        smoothing defaults to 'jeffreys' to avoid alpha==0 or tpr==0 artifacts.
        """
        delta = float(delta)
        assert (0.0 <= delta < 1.0), "delta must be a float"

        curve = self.alpha_beta_curve(smoothing=smoothing)
        alpha = curve["alpha"]  # P(S)
        tpr   = curve["tpr"]    # Q(S)

        eps1 = self._log_ratio(tpr - delta, alpha)
        eps2 = self._log_ratio(alpha - delta, tpr)
        eps_req = np.maximum(eps1, eps2)
        eps_lb = float(np.nanmax(eps_req))

        if eps_lb == -np.inf or np.isnan(eps_lb):
            eps_lb = 0.0
        else:
            eps_lb = max(0.0, eps_lb)

        if return_details:
            best_idx = int(np.nanargmax(eps_req))
            details = {
                "eps_lb": eps_lb,
                "best_threshold": curve["thresholds"][best_idx],
                "best_alpha": alpha[best_idx],
                "best_beta": curve["beta"][best_idx],
                "best_tpr": tpr[best_idx],
                "best_eps_req": float(eps_req[best_idx]),
                "smoothing": smoothing,
            }
            return eps_lb, details

        return eps_lb


class GaussianLLRAuditor:
    """
    Auditor that:
      1) computes approximate Gaussian-vs-Gaussian LLR on scalar averages (out_scores/in_scores),
      2) runs ThresholdAuditor on LLR values.

    Initialize with T, q, mu, sigma to set:
      H0: N(mean0, var0) with mean0=0, var0=sigma^2/T
      H1: N(mean1, var1) with mean1=q*mu, var1=(sigma^2 + q(1-q)mu^2)/T
    """

    def __init__(self, in_scores, out_scores, T, q, mu, sigma):
        self.in_scores = np.asarray(in_scores, dtype=float).ravel()
        self.out_scores = np.asarray(out_scores, dtype=float).ravel()

        self.T = float(T)
        self.q = float(q)
        self.mu = float(mu)
        self.sigma = float(sigma)

        if self.T <= 0:
            raise ValueError("T must be positive.")
        if not (0.0 <= self.q <= 1.0):
            raise ValueError("q must be in [0,1].")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

        # Gaussian approximation parameters
        self.mean0 = 0.0
        self.var0  = (self.sigma ** 2) / self.T
        self.mean1 = self.q * self.mu
        self.var1  = (self.sigma ** 2 + self.q * (1.0 - self.q) * (self.mu ** 2)) / self.T

        if self.var0 <= 0 or self.var1 <= 0:
            raise ValueError("Computed variances must be positive.")

        self._llr_cache = None
        self._threshold_auditor = None

    @staticmethod
    def llr_gauss_vs_gauss(x, mean0, var0, mean1, var1):
        x = np.asarray(x, dtype=float)
        return -0.5*np.log(var1/var0) - 0.5*((x-mean1)**2/var1 - (x-mean0)**2/var0)

    def llr_scores(self):
        """
        Returns (llr_in, llr_out).
        """
        if self._llr_cache is None:
            llr_out = self.llr_gauss_vs_gauss(self.out_scores, self.mean0, self.var0, self.mean1, self.var1)
            llr_in  = self.llr_gauss_vs_gauss(self.in_scores,  self.mean0, self.var0, self.mean1, self.var1)
            self._llr_cache = (llr_in, llr_out)
        return self._llr_cache

    def _get_threshold_auditor(self):
        if self._threshold_auditor is None:
            llr_in, llr_out = self.llr_scores()
            self._threshold_auditor = ThresholdAuditor(llr_in, llr_out)
        return self._threshold_auditor

    def alpha_beta_curve(self, smoothing="jeffreys"):
        """
        Alpha-beta curve for LLR-threshold classifiers.
        """
        return self._get_threshold_auditor().alpha_beta_curve(smoothing=smoothing)

    def epsilon_one_run(self, delta, smoothing="jeffreys", return_details=False):
        """
        Epsilon lower bound computed from the LLR threshold family.
        """
        return self._get_threshold_auditor().epsilon_one_run(
            delta=delta, smoothing=smoothing, return_details=return_details
        )

    def lrt_operating_point(self, threshold=0.0):
        """
        Convenience: report FPR/TPR at the canonical LRT threshold 0 on LLR.
        """
        llr_in, llr_out = self.llr_scores()
        fpr = (llr_out >= threshold).mean()
        tpr = (llr_in  >= threshold).mean()
        return dict(threshold=threshold, fpr=float(fpr), tpr=float(tpr), fnr=float(1.0-tpr))


class MixtureSequenceLLRAuditor:
    """
    NP-optimal auditor under the per-step mixture-vs-null model.

    H0 per step: X ~ N(0, sigma^2)
    H1 per step: X ~ (1-q) N(0, sigma^2) + q N(mu, sigma^2)

    Input matrices:
      out_obs: shape (n_out, T)
      in_obs:  shape (n_in,  T)

    We compute per-sample sequence LLR:
      L = sum_{t=1}^T log( ((1-q)phi0(x_t) + q phi1(x_t)) / phi0(x_t) )
    Then threshold L and convert to eps lower bound via ThresholdAuditor.
    """

    def __init__(self, in_obs, out_obs, q, mu, sigma):
        self.in_obs = np.asarray(in_obs, dtype=float)
        self.out_obs = np.asarray(out_obs, dtype=float)

        if self.in_obs.ndim != 2 or self.out_obs.ndim != 2:
            raise ValueError("in_obs and out_obs must both be 2D arrays of shape (num_samples, steps).")
        if self.in_obs.shape[1] != self.out_obs.shape[1]:
            raise ValueError("in_obs and out_obs must have the same number of columns (steps).")
        if not np.isfinite(self.in_obs).all() or not np.isfinite(self.out_obs).all():
            raise ValueError("observations contain non-finite values.")

        self.q = float(q)
        self.mu = float(mu)
        self.sigma = float(sigma)

        if not (0.0 < self.q < 1.0):
            # q=0 or q=1 are degenerate; you can relax if you really want.
            raise ValueError("q must be in (0,1) for the mixture LLR to be well-defined.")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

        self._llr_cache = None
        self._threshold_auditor = None

    @staticmethod
    def _log_phi(x, mean, sigma):
        return -0.5*np.log(2*np.pi*sigma*sigma) - 0.5*((x-mean)/sigma)**2

    def _llr_step(self, x):
        """
        Per-step LLR: log( (1-q)phi0 + q phi1 ) - log(phi0)
        Stable via log-sum-exp.
        """
        q = self.q
        mu = self.mu
        sigma = self.sigma

        log_phi0 = self._log_phi(x, 0.0, sigma)
        log_phi1 = self._log_phi(x, mu,  sigma)

        a = np.log1p(-q) + log_phi0
        b = np.log(q)    + log_phi1
        m = np.maximum(a, b)
        log_mix = m + np.log(np.exp(a - m) + np.exp(b - m))

        return log_mix - log_phi0

    def llr_scores(self):
        """
        Returns (llr_in, llr_out), each shape (num_samples,).
        """
        if self._llr_cache is None:
            llr_out = self._llr_step(self.out_obs).sum(axis=1)
            llr_in  = self._llr_step(self.in_obs).sum(axis=1)
            self._llr_cache = (llr_in, llr_out)
        return self._llr_cache

    def _get_threshold_auditor(self):
        if self._threshold_auditor is None:
            llr_in, llr_out = self.llr_scores()
            self._threshold_auditor = ThresholdAuditor(llr_in, llr_out)
        return self._threshold_auditor

    def alpha_beta_curve(self, smoothing="jeffreys"):
        return self._get_threshold_auditor().alpha_beta_curve(smoothing=smoothing)

    def epsilon_one_run(self, delta, smoothing="jeffreys", return_details=False):
        return self._get_threshold_auditor().epsilon_one_run(
            delta=delta, smoothing=smoothing, return_details=return_details
        )

    def lrt_operating_point(self, threshold=0.0):
        llr_in, llr_out = self.llr_scores()
        fpr = (llr_out >= threshold).mean()
        tpr = (llr_in  >= threshold).mean()
        return dict(threshold=threshold, fpr=float(fpr), tpr=float(tpr), fnr=float(1.0-tpr))
