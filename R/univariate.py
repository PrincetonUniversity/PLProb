from abc import ABC, abstractmethod
import warnings

import numpy as np
from numpy import nan, inf, isnan
from scipy.stats import percentileofscore
from scipy.optimize import minimize
from statsmodels.distributions.empirical_distribution import ECDF

def ecdf(x):
      return np.array([percentileofscore(x, xi, kind='weak') for xi in x])

def warning(msg: str, warn: bool = True):
      """
      This function prints a warning if warn is True.
      """
      if warn:
            warnings.warn(msg)

SHAPE_XI = True
LL_DEFAULT = -10**10
#Amit: All variable names lambda were renamed to lmbd; lambda is a reserved word in Python and cannot be used as an object's name.

class GPD(ABC):
      """
      Basic GPD class.
      """      
      def _update_stats(self, tail: str, n: int, data: float):
            self.tail = tail
            self.n = n
            self.data = data
      
      @abstractmethod
      def qgpd(self, p: np.ndarray) -> np.ndarray:
            raise NotImplementedError("Not implemented")
      
      @abstractmethod
      def pgpd(self, x: np.ndarray) -> np.ndarray:
            raise NotImplementedError("Not implemented")

class GPDUpperTail(GPD):
      """
      GPD upper tail.
      """
      def __init__(self, tail: str, n: int, data: float,
                  upper_exceedances: float, 
                  upper_threshold: float,
                  p_less_upper_threshold: float, 
                  n_upper_exceedances: float, 
                  upper_method: str, 
                  upper_par_ests: dict, 
                  upper_converged: bool,
                  update_stats: bool = True):
            
            if update_stats:
                  super()._update_stats(tail, n, data)
            self.upper_exceedances        = upper_exceedances 
            self.upper_threshold          = upper_threshold
            self.p_less_upper_threshold   = p_less_upper_threshold
            self.n_upper_exceedances      = n_upper_exceedances
            self.upper_method             = upper_method
            self.upper_par_ests           = upper_par_ests
            self.upper_converged          = upper_converged
      

      def qgpd(obj, p: np.ndarray, warn: bool = False) -> np.ndarray:
            p_orig = p.copy()
            p = np.sort(p)

            n = len(p)
            val = np.zeros(n, dtype=float)
            goodp = (p >= 0) & (p <= 1)
            if sum(goodp)<n:
                  val[~goodp] = nan
                  warning("NaNs are returned for values of the parameter p outside [0,1]", warn)
            k = obj.upper_par_ests["xi"]
            if k<0:
                  val = np.quantile(object.data, p)
                  warning("The empirical quantile function is returned because the tail of the distribution is not treated as heavy since xi is negative", warn)
            else:
                  if not SHAPE_XI:
                        k =  - k
                  ndata = obj.n
                  u = obj.upper_threshold

                  tempf = ECDF(obj.data)
                  pu = tempf(u)
                  smallP = (p < pu) & goodp

                  #The following was creating NAs when the distribution had point masses
                  #I changed it to the empirical quantile function on 7/28/2021
                  #    smallX <- obj@data <= u
                  #    Xgrid <- seq(from=min(obj@data[smallX]), to = max(obj@data[smallX]), length=length(obj@data[smallX]))
                  #    smallQ  <- approxfun(tempf(Xgrid),Xgrid)
                  #    val[smallP] <- smallQ(p[smallP])
                  
                  val[smallP] = np.quantile(obj.data, p[smallP])
                  # Estimate of the quantile function above the threshold
                  quant = u + (obj.upper_par_ests["lambda"] * (((1 - p[~smallP & goodp])/(1 - pu))**( - k) - 1))/k
                  val[(~smallP & goodp)] = quant

            val_orig = val.copy()
            val_orig[np.argsort(p_orig)] = val
            return val_orig


      # def qgpd(self, p: np.ndarray) -> np.ndarray:
      #       p_orig = p.copy()
      #       p.sort() #Ascending
      #       n = len(p)
      #       val = np.zeros(n, dtype=float)
      #       goodp = (p >= 0) & (p <= 1)
      #       val[~goodp] = nan
      #       k = self.upper_par_ests["xi"]
      #       if (not SHAPE_XI):
      #             k =  - k

      #       u = self.upper_threshold

      #       tempf = ecdf(self.data)
      #       pu = np.percentile(tempf, u)
      #       smallP = (p < pu) & goodp
            
      #       val[smallP] = np.quantile(self.data, p[smallP])
      #       # Quantile function above the threshold    
      #       quant = u + (self.upper_par_ests["lambda"] * (((1 - p[~smallP & goodp])/(1 - pu))**( - k) - 1))/k
      #       val[(~smallP & goodp)] = quant

      #       val_orig = val.copy()
      #       val_orig[np.argsort(p_orig)] = val
      #       return val_orig
      

      def pgpd(self, x: np.ndarray, warn: bool = False) -> np.ndarray:

            x_orig = x.copy()
            x.sort() #Ascending

            n = len(x)
            tempf = ecdf(self.data)
            val = np.zeros(n, dtype=float)
            k = self.upper_par_ests["xi"]
            if k<0:
                  val = tempf(x)
                  warning("The empirical cdf is returned since the tail of the distribution is not treated as heavy because xi is negative", warn)
            else:
                  if(not SHAPE_XI):
                        k =  - k
                  u = self.upper_threshold
                  
                  small = (x <= u)

                  
                  val[small] = tempf(x[small])
                  # val[small] = np.percentile(tempf, x[small])

                  # this is the estimate of F above the threshold:
                  pu = tempf(u)
                  # pu = np.percentile(tempf, u)
                  valsm = 1 - (1 - pu) * (1 + (k * (x[~small] - u))/self.upper_par_ests["lambda"])**(-1/k)
                  valsm[((k * (x[~small] - u))/self.upper_par_ests["lambda"]) <= -1] = 1
                  val[~small] = valsm

            val_orig = val.copy()
            val_orig[np.argsort(x_orig)] = val
            return val_orig


class GPDLowerTail(GPD):
      """
      GPD lower tail.
      """
      def __init__(self, tail: str, n: int, data: float,
                  lower_exceedances: float, 
                  lower_threshold: float, 
                  p_larger_lower_threshold: float, 
                  n_lower_exceedances: float, 
                  lower_method: str,
                  lower_par_ests: dict, 
                  lower_converged: bool,
                  update_stats: bool = True):
            
            if update_stats:
                  super()._update_stats(tail, n, data)
            self.lower_exceedances        = lower_exceedances
            self.lower_threshold          = lower_threshold
            self.p_larger_lower_threshold = p_larger_lower_threshold
            self.n_lower_exceedances      = n_lower_exceedances
            self.lower_method             = lower_method
            self.lower_par_ests           = lower_par_ests
            self.lower_converged          = lower_converged
      
      def qgpd(self, p: np.array) -> np.ndarray:
            tmpdist = GPDUpperTail(
                              tail = "upper",
                              n = self.n,   
                              data = - self.data,
                              upper_exceedances = - self.lower_exceedances, 
                              upper_threshold = - self.lower_threshold,
                              p_less_upper_threshold= self.p_larger_lower_threshold, 
                              n_upper_exceedances = self.n_lower_exceedances, 
                              upper_method = self.lower_method, 
                              upper_par_ests = self.lower_par_ests, 
                              upper_converged = self.lower_converged) 
            p = tmpdist.qgpd(-p)
            return p
      
      def pgpd(self, p: np.ndarray) -> np.ndarray:
            tmpdist = GPDUpperTail(
                        tail = "upper",
                        n = self.n,   
                        data = - self.data,
                        upper_exceedances = - self.lower_exceedances, 
                        upper_threshold = - self.lower_threshold,
                        p_less_upper_threshold= self.p_larger_lower_threshold, 
                        n_upper_exceedances = self.n_lower_exceedances, 
                        upper_method = self.lower_method, 
                        upper_par_ests = self.lower_par_ests, 
                        upper_converged = self.lower_converged) 
            p = 1 - tmpdist.pgpd(-p)
            return p


class GPDTwoTails(GPD):
      """
      GPD two tails.
      """
      def __init__(self, tail: str, n: int, data: float,
                  upper_exceedances: float, 
                  lower_exceedances: float, 
                  upper_threshold: float,
                  lower_threshold: float, 
                  p_less_upper_threshold: float, 
                  p_larger_lower_threshold: float, 
                  n_upper_exceedances: float, 
                  n_lower_exceedances: float, 
                  upper_method: str, 
                  lower_method: str,
                  upper_par_ests: dict, 
                  lower_par_ests: dict, 
                  upper_converged: bool, 
                  lower_converged: bool):
            super()._update_stats(tail, n, data)
            self.upper_tail = GPDUpperTail(tail=tail, n=n, data=data,
                  upper_exceedances=upper_exceedances, 
                  upper_threshold=upper_threshold,
                  p_less_upper_threshold=p_less_upper_threshold, 
                  n_upper_exceedances=n_upper_exceedances,
                  upper_method=upper_method,
                  upper_par_ests=upper_par_ests, 
                  upper_converged=upper_converged,
                  update_stats=False)
            self.lower_tail = GPDLowerTail(tail=tail, n=n, data=data,
                  lower_exceedances=lower_exceedances, 
                  lower_threshold=lower_threshold, 
                  p_larger_lower_threshold=p_larger_lower_threshold, 
                  n_lower_exceedances=n_lower_exceedances, 
                  lower_method=lower_method,
                  lower_par_ests=lower_par_ests, 
                  lower_converged=lower_converged,
                  update_stats=False)


      def qgpd(self, p: np.ndarray, warn: bool = False) -> np.ndarray:
            p_orig = p.copy()	
            p = np.sort(p)

            N = len(p)
            val = np.zeros(N, dtype=float)
            goodp = (p >= 0) & (p <= 1)
            if len(goodp)<N:
                  val[~goodp] = nan
                  warning("NaNs are returned for values of the parameter p outside [0,1]", warn)


            tempf = ECDF(self.data)
            p_upper = tempf(self.upper_tail.upper_threshold)
            p_lower = tempf(self.lower_tail.lower_threshold)
            midP = (p_lower <= p) & (p <= p_upper)
            # midX = (self.lower_tail.lower_threshold <= self.data) & (self.data <= self.upper_tail.upper_threshold)
            # Xgrid = np.linspace(min(self.data[midX]), max(self.data[midX]), len(self.data[midX]))
            # Xgrid = seq(from=min(self.data[midX]), to = max(self.data[midX]), length=length(self.data[midX]))
            #    midQ  <- approxfun(tempf(Xgrid),Xgrid)
            #   this was changed on 7-31-2021 to
            #    midQ  <- approxfun(tempf(Xgrid),Xgrid,yleft=obj@lower.threshold,yright=obj@upper.threshold,ties=mean)
            #    val[midP] <- midQ(p[midP])
            #   this was further changed on 8-10-2021 to the empirical quantile function to allow point masses in the center
            
            val[midP] = np.quantile(self.data, p[midP])
            
            # this is the estimate of the quantile function in the tails
            upper_tail_x = (p > p_upper) & goodp
            lower_tail_x = (p < p_lower) & goodp

            # First the upper tail 
            k = self.upper_tail.upper_par_ests["xi"]
            if k<0:
                  val[upper_tail_x] = np.quantile(self.data, p[upper_tail_x])
                  warning("The empirical quantile function is returned because the upper tail of the distribution is not treated as heavy since xi is negative", warn)
            else:
                  if not SHAPE_XI:
                        k =  - k
                  # Estimate of the quantile function above the threshold  
                  a = self.upper_tail.upper_par_ests["lambda"]
                  b = self.upper_tail.upper_threshold
                  val[upper_tail_x] = b + (a * (((1 - p[upper_tail_x])/(1 - p_upper))**( - k) -1))/k

            # Next the lower tail
            k = self.lower_tail.lower_par_ests["xi"]
            if k<0:
                  val[lower_tail_x] = np.quantile(self.data, p[lower_tail_x])
                  warning("The empirical quantile function is returned because the lower tail of the distribution is not treated as heavy since xi is negative", warn)
            else:
                  if not SHAPE_XI:
                        k = - k
                  # this is the estimate of F below the threshold:
                  a = self.lower_tail.lower_par_ests["lambda"]
                  b = self.lower_tail.lower_threshold
                  val[lower_tail_x] = b - (a * (((p[lower_tail_x])/(p_lower))**( - k) - 1))/k

            val_orig = val.copy()
            val_orig[np.argsort(p_orig)] = val
            return val_orig


      def pgpd(self, x: np.ndarray, warn: bool = False) -> np.ndarray:
            x_orig = x.copy()
            x.sort() #Ascending

            N = len(x)
            val = np.zeros(N, dtype=float)
            # n = self.n

            tempf = ECDF(self.data)
            midX = (self.lower_tail.lower_threshold <= x) & (x <= self.upper_tail.upper_threshold)
            val[midX] = tempf(x[midX])
            # val[midX] = np.percentile(tempf, x[midX])

            # this is the estimate of F in the tails:
            p_upper = tempf(self.upper_tail.upper_threshold)
            p_lower = tempf(self.lower_tail.lower_threshold)
            # p_upper = np.percentile(tempf, self.upper_threshold)
            # p_lower = np.percentile(tempf, self.lower_tail)

            upper_tail_x = (x > self.upper_tail.upper_threshold)
            lower_tail_x = (x < self.lower_tail.lower_threshold)
            
            # First the upper tail  
            k = self.upper_tail.upper_par_ests["xi"]
            if k<0:
                  val[upper_tail_x] = tempf(x[upper_tail_x])
                  warning("The empirical cdf is returned for the upper tail since the latter is not treated as heavy because xi is negative", warn)
            else:
                  if not SHAPE_XI:
                        k =  - k
                  # this is the estimate of F above the threshold when xi is positive
                  a = self.upper_tail.upper_par_ests["lambda"]
                  b = self.upper_tail.upper_threshold
                  val[upper_tail_x] = 1 - (1 - p_upper) * (1 + (k * (x[upper_tail_x] - b))/a)**(-1/k)
            # if (k < 0 & (sum(x > b - a/k) > 0)):
            #       val[x > b - a/k] = 1.

            # Next the lower tail
            k = self.lower_tail.lower_par_ests["xi"]
            if k<0:
                  val[lower_tail_x] = tempf(x[lower_tail_x])
                  warning("The empirical cdf is returned for the lower tail since the latter is not treated as heavy because xi is negative", warn)
            else:
                  if not SHAPE_XI:
                        k = -k
                  # this is the estimate of F below the threshold:
                  a = self.lower_tail.lower_par_ests["lambda"]
                  b = self.lower_tail.lower_threshold
                  val[lower_tail_x] = p_lower * (1 - (k * (x[lower_tail_x] - b))/a)**(-1/k)
            # if (k < 0 & sum(x < b + a/k) > 0):
            #       val[x < b + a/k] = 0

            val_orig = val.copy()
            val_orig[np.argsort(x_orig)] = val
            return val_orig            


def sample_LMOM(x):
      x = sorted(x)
      N = len(x)
      i = np.arange(1, N+1)

      fn1 = N - i
      fn2 = (N - i - 1)*fn1
      fn3 = (N - i - 2)*fn2
      
      a1 = sum(fn1/(N-1) * x)/N
      a2 = sum(fn2/(N-1)/(N-2) * x)/N
      a3 = sum(fn3/(N-1)/(N-2)/(N-3) * x)/N

      l1 = np.mean(x)
      l2 = l1 - 2*a1
      tau2 = (l1 - 6.0*a1 + 6.0*a2)/l2
      tau3 = (l1 - 12.0*a1 + 30.0*a2 - 20.0*a3)/l2   

      # val = dict()
      # val["ell_1"] = l1
      # val["ell_2"] = l2
      # val["tau_3"] = tau2
      # val["tau_4"] = tau3
      val = [l1, l2, tau2, tau3]
      return val


def gpd_lmom(lmom, location = nan, sample = nan):
      paramest = [nan]*3

      if not isnan(location):
            if (len(lmom) > 4):
                  sample = lmom
                  lmom = sample_LMOM(sample)
            k = lmom[0]/lmom[1] -2
            lmbd = (1+k) * lmom[0]
            m = location

      else:
            if (len(lmom) > 4):
                  sample = lmom
                  lmom = sample_LMOM(lmom)

            if (isnan(sample[0])):
                  raise ValueError("Problem in function gpd_lmom: either location parameter",
                        " or the sample observations should be specified")
            #CONTINUR HRTR
            xx = min(sample)
            n = len(sample)
            k = (n*(lmom[1] - xx) - 2*(n - 1) * lmom[2])/((n - 1)*lmom[2] - (lmom[1] - xx))
            lmbd = (1 + k) *(2 + k) * lmom[2]
            m = xx - lmbd/(n + k)


      paramest[1] = lmbd
      paramest[0] = m
      if SHAPE_XI:
            paramest[2] = -k
      else:
            paramest[2] = k

      return paramest


def gpd_ml(sample, location = nan, init_est = nan, epsilon = 1e-6):
    
      n = len(sample)
      lmomest = init_est  
      #TempX and TempN were defined with "assign" initially, I think it's the same as defining a global variable? I'm not sure.
      # global tempX_global 
      # global tempN_global
      if (not isnan(location)):
            tmpX = sample-location
            tmpX = tmpX[tmpX>0]
            tmpN = len(tmpX)            
            tempX_global = tmpX            
            tempN_global = tmpN
            if isnan(lmomest) or (isnan(lmomest[0])):
                  lmomest = gpd_lmom(lmom=sample_LMOM(sample), location = location)
                  if (SHAPE_XI): lmomest[2] = -lmomest[2]
            
            x0 = [lmomest[1],lmomest[2]]
            def negative_log_likelihood(theta):
                  # I use ll <- -10^10 to avoid NA's or Inf
                  k = theta[1]
                  lmbd = theta[0]
                  xsc = 1 - (k*(tempX_global))/lmbd
                  ll = LL_DEFAULT
                  if (sum(xsc <= 0) > 0 or lmbd <= 0 or k == 0):
                        ll = LL_DEFAULT
                  else:
                        ll = -tempN_global*np.log(lmbd) + (1/k - 1)*sum(np.log(xsc))
                  return -ll

            fit = minimize(negative_log_likelihood, x0, method="Nelder-Mead", options={"maxiter": 200})
            if (not fit.success):
                  #LMOM estimate might be bad... Try moment etimate as the intial starting point...
                  tempMean = np.mean(tempX_global)
                  CV = (tempMean * tempMean)/np.var(tempX_global)
                  x0[0] = 0.5 * tempMean * (CV+ 1)
                  x0[1] = 0.5 *(CV - 1)
                  fit = minimize(negative_log_likelihood, x0, method="Nelder-Mead", options={"maxiter": 200})
                  if (not fit.success):
                        warnings.warn("Maximum Likelihood Method for the GPD did not converge")              

            paramest = dict()
            paramest["m"] = location
            paramest["lambda"] = fit.x[0]
            if SHAPE_XI:
                  paramest["xi"] = -fit.x[1]
            else:
                  paramest["k"] = fit.x[1]

      else:
            tempX_global = sample
            tempN_global = len(sample)       
            if (isnan(lmomest[0])):
                  lmomest = gpd_lmom(lmom=sample_LMOM(sample), sample = sample)
                  if SHAPE_XI:  lmomest[2] = -lmomest[2]

            def negative_log_likelihood(theta):
                  # I use ll <- -10^10 to avoid NA's or Inf
                  k = theta[2]
                  m = theta[0]
                  #Amit: lambda was renamed to lmbd; lambda is a reserved word in Python and cannot be used as an object's name.
                  lmbd = theta[1]
                  xsc = 1 - k*(tempX_global-m)/lmbd
                  ll = LL_DEFAULT
                  if (sum(xsc < 0) > 0):
                        ll = LL_DEFAULT
                  else:
                        ll = -tempN_global*np.log(lmbd) + (1/k - 1)*sum(np.log(xsc))
                  return -ll

            fit = minimize(negative_log_likelihood, lmomest, method="L-BFGS-B", bounds = [(-inf, min(tempX_global)), (0, inf), (-inf, inf)])
            if not fit.success:
                  warnings.warn("Maximum Likelihood Method for the GPD did not converge")

            paramest = dict()
            paramest["m"] = fit.x[0]
            paramest["lambda"] = fit.x[1]
            if SHAPE_XI:
                  paramest["xi"] = -fit.x[2]
            else:
                  paramest["k"] = fit.x[2]
            

      val = {"n": n, "data": sample, "param_est": paramest, "converged": fit.success}
      return val


def fit_gpd(data, tail = "two", upper = nan, lower = nan, upper_method = "ml", lower_method = "ml", plot = True, warn = False, *args):
      # returns an object of class "gpd". Used to be gpd.tail
      # Called "gpd.tail" to avoid confusion with McNeil's function "gpd"
      # Was "pot.1tail.est' and "pot.2tails.est" in EVANESCE
      l1_thresh = 0.6
      l2_thresh = 0.98
      x_samples = 30
      def quick_xi(u, tmp_data):
            xx = tmp_data[tmp_data>u]
            excess = xx - u
            gpd_est = gpd_ml(sample=excess, location=0)["param_est"]
            return gpd_est["xi"]
      
      if plot and warn:
            warnings.warn("Plot is currently not supported: the flag plot=True is ignored.")

      if (not (tail=="upper" or tail=="lower" or tail=="two")):
            raise ValueError("The parameter tail should be one of the character strings 'upper', 'lower' or 'two'") 

      n = len(data)
      if (tail=="two" or tail=="upper") and isnan(upper):
            tmp_data = np.array(sorted(data))
            #l1 and l2 indeces don't need -1 because I believe the original R code needed +1
            l1 = tmp_data[int(l1_thresh*n)]
            l2 = tmp_data[int(l2_thresh*n)]
            x = np.linspace(l1, l2, x_samples)
            x = x[x<tmp_data[n-3]]
            pretty_xis = np.array([quick_xi(u, tmp_data) for u in x])
            if sum(pretty_xis.flatten()>0)==0:
                  warning("The MLE estimate of the shape parameter xi for the upper tail is likely to be negative, so you should not fit a GPD to the upper tail", warn)
                  if (n <= 100/0.15):
                        uu1 = tmp_data[n - int(n * 0.15) - 1]
                        uu2 = tmp_data[n - int(n * 0.15) - 2]
                        upper = (uu1 + uu2)/2
                  else:
                        upper = tmp_data[n - 101]
            else:
                  upper = np.mean(x[pretty_xis>0])
                  warning(f"In order to find a positive value for xi, the MLE estimate of the shape parameter xi for the upper tail was done with {sum(tmp_data.flatten()>upper)} data points", warn)

      if (tail=="two" or tail=="lower") and isnan(lower):
            tmp_data = np.array(sorted(-data))
            #l1 and l2 indeces don't need -1 because I believe the original R code needed +1
            l1 = tmp_data[int(l1_thresh*n)]
            l2 = tmp_data[int(l2_thresh*n)]
            x = np.linspace(l1, l2, x_samples)
            x = x[x<tmp_data[n-3]]
            pretty_xis = np.array([quick_xi(u, tmp_data) for u in x])
            if sum(pretty_xis.flatten()>0)==0:
                  warning("The MLE estimate of the shape parameter xi for the lower tail is likely to be negative, so you should not fit a GPD to the lower tail", warn)
                  
                  if n <= 100/0.15:
                        #Arrays in R are 1 indexed
                        uu1 = tmp_data[n - int(n * 0.15) - 1]
                        uu2 = tmp_data[n - int(n * 0.15) - 2]
                        lower = -(uu1 + uu2)/2
                  else:
                        lower = -tmp_data[n-101]
            else:
                  lower = -np.mean(x[pretty_xis>0])
                  warning(f"In order to find a positive value of xi, the MLE estimate of the shape parameter xi for the lower tail was done with {sum(tmp_data.flatten()>-lower)} data points", warn)

      if (tail == "two" or tail == "upper"):
            # Analysis of the upper tail!
            upper_exceed = data[data > upper]
            if len(upper_exceed) == 0:
                  raise ValueError("Trying to fit an upper tail when there is no upper tail")
            upper_excess = upper_exceed - upper
            if upper_method.lower()=="ml":
                  gpd_est_res = gpd_ml(sample = upper_excess, location = 0)
                  gpd_est = gpd_est_res["param_est"]
                  upper_converged = gpd_est_res["converged"]
                  if not upper_converged:
                        warning(" MLE method for GPD did not converge for the upper tail. You can try to set the option upper.method = \"lmom\" for upper tail", warn)
            elif upper_method.lower() == "lmom":
                  lmom = sample_LMOM(upper_excess)
                  gpd_est = gpd_lmom(lmom, sample = upper_excess, location = 0)["param_est"]
                  upper_converged = nan
            else:
                  raise ValueError(f"Unknown method for the parameter estimation: {upper_method}")

            # upper_par_ests = [gpd_est["lambda"], gpd_est["xi"]]
            upper_par_ests = gpd_est
            n_upper_exceed = len(upper_excess)
            p_less_upper_thresh = 1 - n_upper_exceed/n

      if tail == "two" or tail == "lower":
            # Analysis of the lower tail!
            lower_exceed = data[data < lower]
            if len(lower_exceed) == 0:
                  raise ValueError("Trying to fit a lower tail when there is no lower tail")
            lower_excess = lower - lower_exceed
            if lower_method.lower()=="ml":
                  gpd_est_res = gpd_ml(sample = lower_excess,location = 0)
                  gpd_est = gpd_est_res["param_est"]
                  lower_converged = gpd_est_res["converged"]
                  if not lower_converged:
                        warning(" MLE method for GPD did not converge for the lower tail. You can try to set the option lower.method = \"lmom\" for the lower tail", warn)

            elif lower_method.lower() == "lmom":
                  gpd_est = gpd_lmom(sample = lower_excess, location = 0)["param_est"]
                  lower_converged = nan
            else:
                  raise ValueError(f"Unknown method for the parameter estimation: {lower_method}")

            # lower_par_ests = [gpd_est["lambda"], gpd_est["xi"]]
            lower_par_ests = gpd_est
            n_lower_exceed = len(lower_excess)
            p_larger_lower_thresh = 1 - n_lower_exceed/n

      if tail == "two":
            out = GPDTwoTails(
                  tail = tail,
                  n = len(data),   
                  data = sorted(data),
                  upper_exceedances = upper_exceed, 
                  lower_exceedances = lower_exceed, 
                  upper_threshold = upper,
                  lower_threshold = lower, 
                  p_less_upper_threshold= p_less_upper_thresh, 
                  p_larger_lower_threshold =  p_larger_lower_thresh, 
                  n_upper_exceedances = n_upper_exceed, 
                  n_lower_exceedances = n_lower_exceed, 
                  upper_method = upper_method, 
                  lower_method = lower_method,
                  upper_par_ests = upper_par_ests, 
                  lower_par_ests = lower_par_ests, 
                  upper_converged = upper_converged, 
                  lower_converged = lower_converged) 
            # if(plot) 
            # {
            #       par_orig = par()
            #             par(mfrow = c(2, 1))
            #             qq = qpareto(ppoints(upper_excess), xi = upper_par_ests["xi"])
            #       plot(qq, sort(upper_excess), xlab = paste("GPD Quantiles, for xi = ", format(upper_par_ests["xi"],digits=3,nsmall=2)),ylab="Excess over threshold", ...)
            #             title("Upper Tail")
            #             qq = qpareto(ppoints(lower_excess), xi = lower_par_ests["xi"])
            #       plot(qq, sort(lower_excess), xlab = paste("GPD Quantiles, for xi = ", format(lower_par_ests["xi"],digits=3,nsmall=2)),ylab="Excess over threshold", ...)
            #             title("Lower Tail")
            #             par(mfrow = c(1, 1))
            # }
      elif tail == "upper":
            out = GPDUpperTail(
                  tail = tail,
                  n = len(data),   
                  data = sorted(data),
                  upper_exceedances = upper_exceed, 
                  upper_threshold = upper,
                  p_less_upper_threshold= p_less_upper_thresh, 
                  n_upper_exceedances = n_upper_exceed, 
                  upper_method = upper_method, 
                  upper_par_ests = upper_par_ests, 
                  upper_converged = upper_converged)
            # if(plot) 
            # {
            #             par_orig = par()
            #             qq = qpareto(ppoints(upper_excess), xi = upper_par_ests["xi"])
            #       plot(qq, sort(upper_excess), xlab = paste("GPD Quantiles, for xi = ", format(upper_par_ests["xi"],digits=3,nsmall=2)),ylab="Excess over threshold", ...)
            #             title("Upper Tail")
            # }
      elif tail == "lower":

            out = GPDUpperTail(		
                  tail = tail,
                  n = len(data),   
                  data = sorted(data),
                  lower_exceedances = lower_exceed, 
                  lower_threshold = lower, 
                  p_larger_lower_threshold =  p_larger_lower_thresh, 
                  n_lower_exceedances = n_lower_exceed, 
                  lower_method = lower_method,
                  lower_par_ests = lower_par_ests, 
                  lower_converged = lower_converged) 
            # if(plot) 
            # {
            #       par_orig = par()
            #             qq = qpareto(ppoints(lower_excess), xi = lower_par_ests["xi"])
            #       plot(qq, sort(lower_excess), xlab = paste("GPD Quantiles, for xi = ", format(lower_par_ests["xi"],digits=3,nsmall=2)),ylab="Excess over threshold", ...)
            #             title("Lower Tail")
            # }

      return out
