"""
Name : portfolio_optimizer.py
Author : Yinsen Miao
 Contact : yinsenm@gmail.com
Time    : 7/21/2021
Desc: Solve mean-variance optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from gerber import gerber_cov_stat1, gerber_cov_stat2
from ledoit import ledoit


def set_eps_wgt_to_zeros(in_array, eps=1e-4):
    # set small weights to 0 and return a list
    out_array = np.array(in_array)
    out_array[np.abs(in_array) < eps] = 0
    out_array = np.array(out_array) / np.sum(out_array)
    return out_array


class portfolio_optimizer:
    def __init__(self, min_weight: float = 0., max_weight: float = 1.0,
                 cov_function: str = "HC",
                 freq: str = "monthly",
                 gs_threshold: float = 0.5):
        """
        :param min_weight:
        :param max_weight:
        :param cov_function: can be one of the HC (historical covariance matrix), GS1 (Gerber Stat1), GS2 (Gerber Stat2)
        :param freq: frequency of the returns series either daily or monthly
        :param gs_threshold: threshold of Gerber statistics between 0 and 1
        """
        # check arguments
        assert cov_function in ['HC', 'GS1', 'GS2', 'SM'], "The covariance function must be one from HC, SM, GS1, and GS2"
        assert freq in ['daily', 'monthly'], "The return series can only be either daily or monthly"
        assert 1 > min_weight >= 0, "The minimal weight shall be in [0, 1)"
        assert 1 >= max_weight > 0, "The maximum weight shall be in (0, 1]"
        assert 1 >= gs_threshold > 0, "The Gerber shrinkage threshold shall be in (0, 1]"

        self.min_weight = min_weight
        self.max_weight = max_weight

        self.factor = 252 if freq == "daily" else 12  # annual converter
        self.cov_function = cov_function  # covariance function can be one of HC, GS1, GS2
        self.freq = freq  # freq of return series can be either daily or monthly
        self.init_weights = None  # initial portfolio weights
        self.covariance = None
        self.returns_df = None
        self.negative_returns_df = None
        self.covariance_neg = None  # covariance matrix of only negative returns for sortino ratio
        self.obj_function = None
        self.by_risk = None
        self.gs_threshold = gs_threshold

    def set_returns(self, returns_df: pd.DataFrame):
        """
        pass the return series to the class
        :param returns_df: pd.DataFrame of historical daily or monthly returns
        """
        self.returns_df = returns_df.copy(deep=True)
        self.negative_returns_df = returns_df[returns_df < 0].fillna(0)  # keep only the negative returns

    def optimize(self, obj_function: str,
                 target_std: float = None,
                 target_return: float = None,
                 prev_weights: np.array = None,
                 init_weights: np.array = None,
                 cost: float = None) -> np.array:
        """
        Perform portfolio optimization given a series of returns
        :param obj_function:
        :param target_std: targeted annaulized portfolio standard deviation (std)
        :param target_return: targeted annaulized portfolio return deviation
        :param prev_weights: previous weights
        :param prices: current price level when we rebalance our portfolio
        :param cost: cost of transaction fee and slippage in bps or 0.01%
        :return: an array of portfolio weights p x 1
        """
        n, p = self.returns_df.shape  # n is number of observations, p is number of assets

        if init_weights is None:
            self.init_weights = np.array(p * [1. / p])  # initialize weights: equal weighting
        else:
            self.init_weights = init_weights  # otherwise use the nearby weights as hot start for MVO

        self.obj_function = obj_function

        # get covariance matrix
        if self.cov_function == "HC":
            self.covariance = self.returns_df.cov().to_numpy()  # convert to numpy
            self.covariance_neg = self.negative_returns_df.cov().to_numpy()  # convert to numpy
        elif self.cov_function == "SM":
            self.covariance, _ = ledoit(self.returns_df.values)
            self.covariance_neg, _ = ledoit(self.negative_returns_df.values)
        elif self.cov_function == "GS1":
            self.covariance, _ = gerber_cov_stat1(self.returns_df.values, threshold=self.gs_threshold)
            self.covariance_neg, _ = gerber_cov_stat1(self.negative_returns_df.values, threshold=self.gs_threshold)
        elif self.cov_function == "GS2":
            self.covariance, _ = gerber_cov_stat2(self.returns_df.values, threshold=self.gs_threshold)
            self.covariance_neg, _ = gerber_cov_stat2(self.negative_returns_df.values, threshold=self.gs_threshold)

        # set objective function
        if obj_function == "equalWeighting":
            self.init_weights = np.array(p * [1. / p])  # initialize weights: equal weighting
            return self.init_weights

        # set the bounds of each asset holding from 0 to 1
        bounds = tuple((self.min_weight, self.max_weight) for k in range(p))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]  # fully invest

        if obj_function == 'meanVariance':
            if target_std is not None:
                self.by_risk = True
                # optimize under risk constraint
                constraints.append({'type': 'eq', 'fun': lambda weights: \
                    self.calc_annualized_portfolio_std(weights) - target_std})
            else:
                # optimize under return constraint
                self.by_risk = False
                constraints.append({'type': 'eq', 'fun': lambda weights: \
                    self.calc_annualized_portfolio_return(weights) - target_return})

        if prev_weights is not None and cost is not None:
            # cost function with transaction fee
            cost_fun = lambda weights: self.object_function(weights) +\
                                       np.abs(weights - prev_weights).sum() * cost / 10000.
        else:
            # cost function without any transaction fee
            cost_fun = lambda weights: self.object_function(weights)

        # trust-constr, SLSQP, L-BFGS-B
        try:
            opt = minimize(cost_fun, x0=self.init_weights, bounds=bounds, constraints=constraints, method="SLSQP")
        except:
            # if SLSQP fails then switch to trust-constr
            opt = minimize(cost_fun, x0=self.init_weights, bounds=bounds, constraints=constraints, method="trust-constr")

        return set_eps_wgt_to_zeros(opt['x'])   # pull small values to zeros

    def object_function(self, weights: np.array) -> float:
        """
        :param weights: current weights to be optimized
        """

        if self.obj_function == "maxReturn":
            f = self.calc_annualized_portfolio_return(weights)
            return -f
        elif self.obj_function == "minVariance":
            f = self.calc_annualized_portfolio_std(weights)
            return f
        elif self.obj_function == "meanVariance" and self.by_risk:
            f = self.calc_annualized_portfolio_return(weights)  # maximize target return level
            return -f
        elif self.obj_function == "meanVariance" and not self.by_risk:
            f = self.calc_annualized_portfolio_std(weights)  # minimize target risk or std level
            return f
        elif self.obj_function == "maxSharpe":
            f = self.calc_annualized_portfolio_sharpe_ratio(weights)
            return -f
        elif self.obj_function == "maxSortino":
            f = self.calc_annualized_sortino_ratio(weights)
            return -f
        elif self.obj_function == 'riskParity':
            f = self.calc_risk_parity_func(weights)
            return f
        else:
            raise ValueError("Object function shall be one of the equalWeighting, maxReturn, minVariance, " +
                             "meanVariance, maxSharpe, maxSortino or riskParity")

    def calc_annualized_portfolio_return(self, weights: np.array) -> float:
        # calculate the annualized standard returns
        annualized_portfolio_return = float(np.sum(self.returns_df.mean() * self.factor * weights))
        #float(np.sum(((1 + self.returns_df.mean()) ** self.factor - 1) * weights))
        return annualized_portfolio_return

    def calc_annualized_portfolio_std(self, weights: np.array) -> float:
        if self.obj_function == "equalWeighting":
            # if equal weight then set the off diagonal of covariance matrix to zero
            annualized_portfolio_std = np.sqrt(np.dot(weights.T, np.dot(np.diag(self.covariance.diagonal()) * self.factor, weights)))
        else:
            temp = np.dot(weights.T, np.dot(self.covariance * self.factor, weights))
            if temp <= 0:
                temp = 1e-20  # set std to a tiny number
            annualized_portfolio_std = np.sqrt(temp)
        if annualized_portfolio_std <= 0:
            raise ValueError('annualized_portfolio_std cannot be zero. Weights: {weights}')
        return annualized_portfolio_std

    def calc_annualized_portfolio_neg_std(self, weights: np.array) -> float:
        if self.obj_function == "equalWeighting":
            # if equal weight then set the off diagonal of covariance matrix to zero
            annualized_portfolio_neg_std = np.sqrt(np.dot(weights.T, np.dot(np.diag(self.covariance_neg.diagonal()) * self.factor, weights)))
        else:
            annualized_portfolio_neg_std = np.sqrt(np.dot(weights.T, np.dot(self.covariance_neg * self.factor, weights)))
        if annualized_portfolio_neg_std == 0:
            raise ValueError('annualized_portfolio_std cannot be zero. Weights: {weights}')
        return annualized_portfolio_neg_std

    def calc_annualized_portfolio_moments(self, weights: np.array) -> tuple:
        # calculate the annualized portfolio returns as well as its standard deviation
        return self.calc_annualized_portfolio_return(weights), self.calc_annualized_portfolio_std(weights)

    def calc_annualized_portfolio_sharpe_ratio(self, weights: np.array) -> float:
        # calculate the annualized Sharpe Ratio
        return self.calc_annualized_portfolio_return(weights) / self.calc_annualized_portfolio_std(weights)

    def calc_annualized_sortino_ratio(self, weights: np.array) -> float:
        # calculate the annualized Sortino Ratio
        return self.calc_annualized_portfolio_return(weights) / self.calc_annualized_portfolio_neg_std(weights)

    def calc_risk_parity_func(self, weights):
        # Spinu formulation of risk parity portfolio
        assets_risk_budget = self.init_weights
        portfolio_volatility = self.calc_annualized_portfolio_std(weights)

        x = weights / portfolio_volatility
        risk_parity = (np.dot(x.T, np.dot(self.covariance * self.factor, x)) / 2.) - np.dot(assets_risk_budget.T, np.log(x + 1e-10))
        return risk_parity

    def calc_relative_risk_contributions(self, weights):
        # calculate the relative risk contributions for each asset given returns and weights
        rrc = weights * np.dot(weights.T, self.covariance) / np.dot(weights.T, np.dot(self.covariance, weights))
        return rrc


# unitest the code
if __name__ == "__main__":
    bgn_date = "2016-01-01"
    end_date = "2020-01-01"
    file_path = "../data/prcs.csv"
    rets_df = pd.read_csv(file_path, parse_dates=['Date'], index_col=["Date"]).pct_change()[bgn_date: end_date]
    rets = rets_df.values

    # test objective function list
    obj_function_list = ['equalWeighting', 'minVariance', 'maxReturn', 'maxSharpe', 'maxSortino', 'riskParity']
    cov_function_list = ["HC", "SM", "GS1", "GS2"]

    for cov_fun in cov_function_list:
        print("MVO based on %s covariance function ..." % cov_fun)
        port_opt = portfolio_optimizer(min_weight=0, max_weight=1, cov_function=cov_fun, freq="monthly")
        port_opt.set_returns(returns_df=rets_df)
        # run MVO under various optimization goals

        for obj_fun_str in obj_function_list:
            weights = port_opt.optimize(obj_fun_str)
            ret, std = port_opt.calc_annualized_portfolio_moments(weights=weights)
            sharpe = port_opt.calc_annualized_portfolio_sharpe_ratio(weights=weights)
            sortino = port_opt.calc_annualized_sortino_ratio(weights=weights)
            print("%20s: ret %.3f, std %.3f, Sharpe %.3f, Sortino %.3f" % (obj_fun_str, ret, std, sharpe, sortino))
        obj_fun_str = "meanVariance"
        # optimize for target std levels
        target_stds = [3, 6, 9, 12, 15]
        for target_std in target_stds:
            weights = port_opt.optimize(obj_fun_str, target_std / 100.)
            # print(weights)
            ret, std = port_opt.calc_annualized_portfolio_moments(weights=weights)
            sharpe = port_opt.calc_annualized_portfolio_sharpe_ratio(weights=weights)
            sortino = port_opt.calc_annualized_sortino_ratio(weights=weights)
            print("%20s (%02d%%): ret %.3f, std %.3f, Sharpe %.3f, Sortino %.3f" % (
                obj_fun_str, target_std, ret, std, sharpe, sortino))
