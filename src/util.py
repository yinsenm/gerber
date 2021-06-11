import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, date
from portfolio_optimizer import portfolio_optimizer
from gerber import gerber_cov_stat1, gerber_cov_stat2

DEBUG = 0  # turn on debug mode or not

def calc_assets_moments(returns_df: pd.DataFrame,
                        weights=None,
                        cov_function: str = None,
                        freq: str = "monthly") -> tuple:
    """
    calcultate the annualized return and volatility (std) given returns, weights and return data frequency
    :param returns_df:
    :param weights:
    :param cov_function: covariance function, can be one of the HC, GS1, GS2, or None
    :param freq:
    :return: (ret, std) tuple
    """
    assert freq in ['daily', 'monthly'], \
        "The return series can only be either daily or monthly"
    assert cov_function in ['HC', 'GS1', 'GS2', None], \
        "The covariance function must be one from HC, GS1, and GS2 or None"
    factor = 252 if freq == "daily" else 12
    if weights is None:
        std = returns_df.std() * np.sqrt(factor)
        ret = ((1 + returns_df.mean()) ** factor) - 1.0
    else:
        ret = ((1 + returns_df.mul(weights).sum(axis=1).mean()) ** factor) - 1.0
        if cov_function is None:
            std = returns_df.mul(weights).sum(axis=1).std() * np.sqrt(factor)
        elif cov_function == "HS":
            cov_mat = returns_df.cov()  # covariance matrix
        elif cov_function == "GS1":
            cov_mat, _ = gerber_cov_stat1(returns_df.values)  # covariance matrix
        else:
            cov_mat, _ = gerber_cov_stat2(returns_df.values)  # covariance matrix

        if cov_function is not None:
            std = np.sqrt(np.dot(weights.T, np.dot(cov_mat * factor, weights)))
    return ret, std


def calc_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    # calculate the monthly returns from a dataframe of daily prices (indexed by dates with assets on the columns)
    # group by year and month and take first value of the month
    df_start_month = (df.groupby([df.index.year, df.index.month])).apply(lambda x: x.head(1))
    df_start_month = df_start_month.droplevel([0, 1])  # drop the first 2 indexes
    # df_start_month = df_start_month.append(df.iloc[-1].to_frame().transpose())
    df_monthly_returns = df.pct_change().shift(-1).dropna()
    return df_monthly_returns


def get_frontier_limits(returns_df: pd.DataFrame,
                        cov_function: str = "HC",
                        freq: str = "monthly",
                        gs_threshold: float = 0.5) -> dict:
    """
    Estimate optimal portfolios at the endpoints of the efficient frontier.
    :param returns_df: pd.Data.Frame of the assets' return
    :param cov_function: covariance function can be one of HC, GS1, GS2
    :param freq: compounding frequency in returns_df
    :param gs_threshold: threshold of Gerber statistics between 0 and 1
    :return: dict of mimVariance and maxReturn portfolio
    """
    port_opt = portfolio_optimizer(min_weight=0, max_weight=1,
                                   cov_function=cov_function,
                                   freq=freq,
                                   gs_threshold=gs_threshold)
    port_opt.set_returns(returns_df)
    _, p = returns_df.shape

    result_dict = {}
    # get minVariance and maxReturn Portfolio
    for obj_fun_str in ['minVariance', 'maxReturn']:
        weights = port_opt.optimize(obj_fun_str)
        ret, std = port_opt.calc_annualized_portfolio_moments(weights=weights)
        result_dict[obj_fun_str] = {}
        result_dict[obj_fun_str]["ret_std"] = (ret, std)
        result_dict[obj_fun_str]["weights"] = weights

    # loop through the tickers and calculate its returns and volatility pair
    # find the asset with maximum return and use its (ret, std) as portfolio boundary
    max_ret, max_std, max_idx, idx = -1, 0, -1, -1
    for ticker in returns_df:
        _dat = returns_df[ticker]
        ret, std = calc_assets_moments(_dat, freq=freq)
        idx += 1
        if ret > max_ret:
            max_ret = ret
            max_std = std
            max_idx = idx

    if result_dict["maxReturn"]["ret_std"][1] < max_std:
        # allocate 100% on a signal asset
        result_dict["maxReturn"] = {}
        result_dict["maxReturn"]["ret_std"] = (max_ret, max_std)
        result_dict["maxReturn"]["weights"] = np.array([0] * p)
        result_dict["maxReturn"]["weights"][max_idx] = 1
    return result_dict


def get_frontier_by_return(returns_df: pd.DataFrame,
                           target_returns_array: np.array,
                           cov_function: str = "HC",
                           freq: str = "monthly") -> tuple:
    """
        calculate the pairs of volatility / return coordinates for the efficient frontier
            given the targeted annualized returns
    :param returns_df:
    :param target_returns_array:  np.array of target returns
    :param cov_function:
    :param freq:
    :return: a tuple of (rets_list, stds_list, weights_list) pair
    """
    port_opt = portfolio_optimizer(min_weight=0, max_weight=1, cov_function=cov_function, freq=freq)
    port_opt.set_returns(returns_df)
    rets_list, stds_list, weights_list = [], [], []

    init_weights = None  # use init_weights to hot start the MVO optimization later
    for target_return in target_returns_array:
        weights = port_opt.optimize('meanVariance', target_return=target_return, init_weights=init_weights)
        ret, std = port_opt.calc_annualized_portfolio_moments(weights)
        rets_list.append(ret)
        stds_list.append(std)
        weights_list.append(weights)
        init_weights = weights
    return rets_list, stds_list, weights_list


def get_frontier_by_risk(returns_df: pd.DataFrame,
                         target_risks_array: np.array,
                         cov_function: str = "HC",
                         freq: str = "monthly",
                         prev_port_weights: dict = None,
                         gs_threshold: float = 0.5,
                         cost: float = None) -> tuple:
    """
        calculate the pairs of volatility / return coordinates for the efficient frontier
            given the targeted annualized volatilities
    :param returns_df: pandas dataframe of return
    :param target_risks_array: np.array of target risks
    :param cov_function: covariance function either in HC (historical covariance), GS1 (Geber 1) and GS2 (Geber2)
    :param freq:
    :param gs_threshold: threshold of Gerber statistics between 0 and 1
    :return: a tuple of (rets_list, stds_list, weights_list) pair
    """

    # get range of stds for efficient portfolio
    _port_limits = get_frontier_limits(returns_df, cov_function, freq, gs_threshold=gs_threshold)
    max_ret, max_std = _port_limits['maxReturn']['ret_std']
    max_wgt = _port_limits['maxReturn']['weights']
    min_ret, min_std = _port_limits['minVariance']['ret_std']
    min_wgt = _port_limits['minVariance']['weights']

    # solve for mean variance portfolio given targeted risks
    port_opt = portfolio_optimizer(min_weight=0, max_weight=1,
                                   cov_function=cov_function,
                                   freq=freq,
                                   gs_threshold=gs_threshold)
    port_opt.set_returns(returns_df)
    rets_list, stds_list, weights_list = [], [], []

    init_weights = None  # use init_weights to hot start the MVO optimization later
    for target_risk in target_risks_array:
        if target_risk <= min_std:
            rets_list.append(min_ret)
            stds_list.append(min_std)
            weights_list.append(min_wgt)
            init_weights = None
            if DEBUG:
                warnings.warn("Target risk level %.3f is lower than the feasible range [%.3f, %.3f]." % (target_risk, min_std, max_std))
        elif min_std < target_risk < max_std:
            if prev_port_weights is not None and cost is not None:
                weights = port_opt.optimize('meanVariance', target_std=target_risk, init_weights=init_weights,
                                            prev_weights=prev_port_weights["%02dpct" % int(target_risk * 100)]["weights"],
                                            cost=cost)  # add 10pct as penalty for transaction
            else:
                weights = port_opt.optimize('meanVariance', target_std=target_risk, init_weights=init_weights)

            _ret, _std = port_opt.calc_annualized_portfolio_moments(weights)
            rets_list.append(_ret)
            stds_list.append(_std)
            weights_list.append(weights)
            init_weights = weights
        elif target_risk >= max_std:
            rets_list.append(max_ret)
            stds_list.append(max_std)
            weights_list.append(max_wgt)
            init_weights = None
            if DEBUG:
                warnings.warn("Target risk level %.3f is higher than the feasible range [%.3f, %.3f]." % (target_risk, min_std, max_std))

    return rets_list, stds_list, weights_list


def get_mean_variance_space(returns_df: pd.DataFrame,
                            target_risks_array,
                            obj_function_list: list,
                            cov_function: str = "HC",
                            freq: str = "monthly",
                            prev_port_weights: dict=None,
                            simulations: int = 0,
                            gs_threshold: float = 0.5,
                            cost: float = None) -> dict:
    """
    Plot the mean-variance space (and efficient frontier) with simulations of portfolios, individual assets and optimal portfolios
    :param freq:
    :param cov_function:
    :param returns_df:
    :param obj_function_list:
    :param target_volatilities: list
    :param simulations:
    :param cost: cost of transaction fee and slippage in bps or 0.01%
    :return: result_dict
    """

    # initialize portfolio constructor
    port_opt = portfolio_optimizer(min_weight=0,
                                   max_weight=1,
                                   cov_function=cov_function,
                                   freq=freq,
                                   gs_threshold=gs_threshold)
    port_opt.set_returns(returns_df)

    # store the tuple of volatility and return pair for each objective functions
    result_dict = {}
    result_dict['port_opt'] = {}  # optimal portfolio (ret, std)
    result_dict['asset'] = {}  # asset (ret, std)
    result_dict['mvo'] = {}  # mvo (ret, std)

    # loop through the objective functions to generate optimal weights
    for obj_fun_str in obj_function_list:
        if prev_port_weights is not None and cost is not None:
            weights = port_opt.optimize(obj_fun_str, prev_weights=prev_port_weights[obj_fun_str]["weights"], cost=cost)
        else:
            weights = port_opt.optimize(obj_fun_str)
        ret, std = port_opt.calc_annualized_portfolio_moments(weights=weights)
        result_dict['port_opt'][obj_fun_str] = {}
        result_dict['port_opt'][obj_fun_str]["ret_std"] = (ret, std)
        result_dict['port_opt'][obj_fun_str]["weights"] = weights

    # # loop through the tickers and calculate its returns and volatility pair
    max_ret, max_std = -1, 0
    for ticker in returns_df:
        _dat = returns_df[ticker]
        ret, std = calc_assets_moments(_dat, freq=freq)
        if ret > max_ret:
            max_ret = ret
            max_std = std
        result_dict['asset'][ticker] = (ret, std)

    # min_std = round(result_dict['port_opt']['minVariance']["ret_std"][1], 2)
    # max_std = round(max(max_std, result_dict['port_opt']['maxReturn']["ret_std"][1]), 2)
    #
    # # calculate the upper part of the efficient frontier with a step of 1%
    # mvo_port_risks_array = np.linspace(min_std, max_std, int((max_std - min_std) / 0.01 + 1))
    #
    # result_dict['mvo']['rets'], result_dict['mvo']['stds'], result_dict['mvo']['weights'] = \
    #     get_frontier_by_risk(returns_df=returns_df,
    #                          target_risks_array=mvo_port_risks_array,
    #                          cov_function=cov_function, freq=freq,
    #                          prev_port_weights=None)

    # compute the range of volatility on the efficient frontier
    _rets, _stds, _wgts = get_frontier_by_risk(returns_df=returns_df,
                                               target_risks_array=target_risks_array,
                                               cov_function=cov_function, freq=freq,
                                               prev_port_weights=prev_port_weights,
                                               gs_threshold=gs_threshold,
                                               cost=cost)
    result_dict['mvo']['rets'], result_dict['mvo']['stds'], result_dict['mvo']['weights'] = _rets, _stds, _wgts

    # append targeted risk portfolio
    for (_trg_rsk, _ret, _std, _wgt) in zip(target_risks_array, _rets, _stds, _wgts):
        result_dict['port_opt']["%02dpct" % int(_trg_rsk * 100)] = {}
        result_dict['port_opt']["%02dpct" % int(_trg_rsk * 100)]["ret_std"] = (_ret, _std)
        result_dict['port_opt']["%02dpct" % int(_trg_rsk * 100)]["weights"] = _wgt

    # run simulations of portfolios and extract their pairs of volatility v.s. return
    _, nassets = returns_df.shape
    rets, stds = [], []

    for i in range(simulations):
        weights = np.random.dirichlet(np.ones(nassets), size=1)[0]
        ret, std = calc_assets_moments(returns_df, weights=weights,
                                       cov_function=cov_function, freq=freq)
        rets.append(ret)
        stds.append(std)
    result_dict['simulations_ret_std'] = [rets, stds]  # simulated return / volatility pair

    # add time stamp to the results
    # get the start and end of the returns_df timestamp
    bgn_date_str = returns_df.index.min().strftime("%Y-%m-%d")
    end_date_str = returns_df.index.max().strftime("%Y-%m-%d")
    result_dict['bgn_date_str'] = bgn_date_str
    result_dict['end_date_str'] = end_date_str
    result_dict['tickers'] = returns_df.columns.format()
    result_dict['cov_function'] = cov_function
    result_dict['freq'] = freq
    return result_dict

# plotting
def ApplyPlotStyle(x, title):
    x.set_title(title, fontsize=15)
    x.set(xlabel='Annualized Volatility (Standard Deviation)', ylabel='Annualized Return')
    ticks = mticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y))
    x.xaxis.set_major_formatter(ticks)
    x.yaxis.set_major_formatter(ticks)
    x.xaxis.label.set_size(13)
    x.yaxis.label.set_size(13)
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.xaxis.set_ticks_position('none')
    x.yaxis.set_ticks_position('none')

def plot_efficient_frontier(result_dict, prefix="results", plotSize=(10, 8)):
    bgn_date_str = result_dict['bgn_date_str']
    end_date_str = result_dict['end_date_str']
    cov_function = result_dict['cov_function']
    freq = result_dict['freq']

    os.makedirs("../%s" % prefix, exist_ok=True)

    # initialize figure size
    plt.figure(figsize=plotSize)

    if len(result_dict['simulations_ret_std']):
        plt.scatter(result_dict['simulations_ret_std'][1], result_dict['simulations_ret_std'][0],
                    c=[y / x for x, y in
                       zip(result_dict['simulations_ret_std'][1], result_dict['simulations_ret_std'][0])],
                    marker='o', alpha=0.1, s=15)

    # plot efficient frontier
    plt.plot(result_dict['mvo']['stds'], result_dict['mvo']['rets'], '-', c='red')

    # plot optimal portfolios
    for obj_function, rets in result_dict['port_opt'].items():
        ret_std = rets["ret_std"]
        if obj_function == "maxReturn":
            off_set = -0.005
        else:
            off_set = 0
        plt.plot(ret_std[1], ret_std[0], '-o', c='black')
        plt.text(ret_std[1], ret_std[0] + off_set, obj_function, verticalalignment='top',
                 horizontalalignment='left', fontsize=12, c='black')

    # plot assets
    for obj_function, ret_std in result_dict['asset'].items():
        plt.scatter(ret_std[1], ret_std[0], marker='o', c='green', s=20, alpha=0.5)
        plt.text(ret_std[1], ret_std[0], obj_function, verticalalignment='top',
                 horizontalalignment='left', fontsize=8, c='green', alpha=0.5)
    ApplyPlotStyle(plt.gca(), title='Mean-Variance (%s, %s) from %s to %s' %
                                    (cov_function, freq, bgn_date_str, end_date_str))
    plt.savefig("../%s/%s_%s_%s.pdf" % (prefix, cov_function, bgn_date_str, end_date_str))


def plot_efficient_frontiers(results_dict, prefix="results", plotSize=(10, 8), permno_to_ticker: dict=None):
    bgn_date_str = results_dict['HC']['bgn_date_str']
    end_date_str = results_dict['HC']['end_date_str']
    freq = results_dict['HC']['freq']
    colors = {"HC": "blue", "SM": "green", "GS1": "red", "GS2": "cyan"}

    # initialize figure size
    plt.figure(figsize=plotSize)

    # plot efficient frontiers
    for cov_function, value in results_dict.items():
        plt.plot(value['mvo']['stds'], value['mvo']['rets'], '-', c=colors[cov_function], label=cov_function)
        plt.scatter(value['port_opt']['maxSharpe']['ret_std'][1], value['port_opt']['maxSharpe']['ret_std'][0],
                 c=colors[cov_function], label="%s: Sharpe" % cov_function, marker='*', s=12 ** 2)

        # port target variance portfolios
        for _key, _value in value['port_opt'].items():
            if "pct" in _key:
                plt.plot(_value['ret_std'][1], _value['ret_std'][0],
                            c=colors[cov_function], marker="o", markersize=6, alpha=0.5,
                            mec=colors[cov_function], mew=0.)

    # plot assets
    for obj_function, ret_std in results_dict['HC']['asset'].items():
        plt.plot(ret_std[1], ret_std[0], marker='o', c='green', markersize=12, alpha=0.5, mec='green', mew=0.)
        if permno_to_ticker is None:
            plt.text(ret_std[1], ret_std[0], obj_function, verticalalignment='top',
                     horizontalalignment='left', fontsize=10, c='green', alpha=0.5)
        else:
            plt.text(ret_std[1], ret_std[0], permno_to_ticker[obj_function], verticalalignment='top',
                     horizontalalignment='left', fontsize=10, c='green', alpha=0.5)
    ApplyPlotStyle(plt.gca(), title='Mean-Variance (%s) from %s to %s' % \
                                    (freq, bgn_date_str, end_date_str))
    plt.legend(loc="best", ncol=2)
    plt.savefig("%s/%s_%s.pdf" % (prefix, bgn_date_str, end_date_str))
    plt.close()


# test util function
if __name__ == "__main__":
    cov_function = "HC"
    freq = "monthly"
    bgn_date = "2016-01-01"
    end_date = "2020-01-01"
    file_path = "../data/bloomberg/rets_v2.csv"
    rets_df = pd.read_csv(file_path, parse_dates=['Date']).query("Date >= '%s' and Date <= '%s'" % \
        (bgn_date, end_date)).set_index(["Date"])
    port_limits = get_frontier_limits(rets_df, cov_function, freq)

    port_limits.keys()

    # get maximum portfolio
    print("Return std range [%.3f, %.3f]" % (port_limits['minVariance']['ret_std'][1], port_limits['maxReturn']['ret_std'][1]))

    target_risks_array = np.array([3, 6, 9, 12, 15, 18, 21]) / 100.
    ret_list, std_list, wgt_list = get_frontier_by_risk(returns_df=rets_df,
                                                        target_risks_array=target_risks_array,
                                                        cov_function=cov_function,
                                                        freq=freq)

    obj_function_list = ['equalWeighting', 'minVariance', 'maxReturn',  # optimization target
                         'maxSharpe', 'maxSortino', 'riskParity']

    result_dict = get_mean_variance_space(returns_df=rets_df,
                            target_risks_array=target_risks_array,
                            obj_function_list=obj_function_list,
                            cov_function="HC", freq="monthly")
    print(result_dict)
    plot_efficient_frontier(result_dict, prefix="fig_test")