"""
Name     : run_mvo.py
Author   : Yinsen Miao
 Contact : yinsenm@gmail.com
Time     : 7/1/2021
Desc     : run mean-variance optimization
"""

from util import get_mean_variance_space, plot_efficient_frontiers, get_frontier_limits
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import argparse
DEBUG = 0

# define global variables
target_volatilities_array = np.arange(2, 16) / 100.  # target volatility level from 2% to 16%
obj_function_list = ['minVariance', 'maxSharpe']
cov_function_list = ["HC", "SM", "GS1"]  # list of covariance function

# portfolio setting
cash_start = 100000.
risk_free_rate = 0.
gs_threshold = 0.9  # threshold for gerber statistics
lookback_win_in_year = 2
lookback_win_size = 12 * lookback_win_in_year


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse parameter for Bloomberg 9")
    parser.add_argument("-s", "--gs_threshold", type=float, default=0.5)
    parser.add_argument("-o", "--optimization_cost", type=float, default=10)
    parser.add_argument("-t", "--transaction_cost", type=float, default=10)
    args = parser.parse_args()
    gs_threshold = args.gs_threshold  # threshold for gerber statistics
    optimization_cost = args.optimization_cost  # penalty for excessive transaction
    transaction_cost = args.transaction_cost  # actual transaction fee in trading simulation
    savepath = "../results/%dyr_c%02dbps_t%02dbps/gs_threshold%.2f" % \
        (lookback_win_in_year, optimization_cost, transaction_cost, gs_threshold)


    prcs = pd.read_csv("../data/prcs.csv", parse_dates=['Date']).\
        set_index(['Date'])
    rets = prcs.pct_change().dropna(axis=0)
    prcs = prcs.iloc[1:]  # drop first row
    nT, p = prcs.shape
    symbols = prcs.columns.to_list()

    # create folder to save results
    os.makedirs("%s" % savepath, exist_ok=True)
    os.makedirs("%s/plots" % savepath, exist_ok=True)


    port_names = obj_function_list + ['%02dpct' % int(tgt * 100) for tgt in target_volatilities_array]

    """
    initialize the portfolio below:
    - equalWeighting
    - minVariance
    - maxReturn
    - maxSharpe
    - maxSortino
    - riskParity
    - meanVariance with risk constraints 3pct, 6pct, 9pct, 12pct, 15pct
    """
    account_dict = {}
    for cov_function in cov_function_list:
        account_dict[cov_function] = {}
        for port_name in port_names:
            account_dict[cov_function][port_name] = []
            account_dict[cov_function][port_name].append(
                {
                    "date": prcs.index[lookback_win_size - 1].strftime("%Y-%m-%d"),
                    "weights": np.array([0] * p),  # portfolio weight for each asset
                    "shares": np.array([0] * p),   # portfolio shares for each asset
                    "values": np.array([0] * p),   # portfolio dollar value for each asset
                    "portReturn": 0,
                    "transCost": 0,
                    "weightDelta": 0,  # compute portfolio turnover for current rebalancing period
                    "portValue": cash_start,
                }
            )

    # # save the trajectory of efficient frontier
    # efficient_frontiers = {}
    # for cov_function in cov_function_list:
    #     efficient_frontiers[cov_function] = []


    t = lookback_win_size
    # keep track of previous optimal weights to penalize extensive turnover
    prev_port_weights_dict = {key: None for key in cov_function_list}
    for t in tqdm(range(lookback_win_size, nT)):
        bgn_date = rets.index[t - lookback_win_size]
        end_date = rets.index[t - 1]
        end_date_p1 = rets.index[t]

        bgn_date_str = bgn_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        end_date_p1_str = end_date_p1.strftime("%Y-%m-%d")

        # subset the data accordingly
        sub_rets = rets.iloc[t - lookback_win_size: t]
        _nT, _ = sub_rets.shape
        prcs_t = prcs.iloc[t-1: t].values[0]  # price at time t
        prcs_tp1 = prcs.iloc[t: t + 1].values[0]  # price at time t + 1
        rets_tp1 = rets.iloc[t: t + 1].values[0]  # return at time t + 1

        if DEBUG:
            print("MVO optimimize from [%s, %s] (n=%d) and applied to rets at %s" % \
                  (bgn_date_str, end_date_str, _nT, end_date_p1_str))

        # get portfolio weight for a given cov_function
        opt_ports_dict = {}
        for cov_function in cov_function_list:
            if DEBUG:
                print("Processing %s ..." % cov_function)
            opt_ports_dict[cov_function] = get_mean_variance_space(sub_rets,
                                                                   target_volatilities_array,
                                                                   obj_function_list, cov_function,
                                                                   prev_port_weights=prev_port_weights_dict[cov_function],
                                                                   gs_threshold=gs_threshold,
                                                                   cost=optimization_cost)
            prev_port_weights_dict[cov_function] = opt_ports_dict[cov_function]["port_opt"]
            for port_name in port_names:
                port_tm1 = account_dict[cov_function][port_name][-1]

                # updated portfolio
                port_t = {
                    "date": end_date_p1_str,
                    "weights": opt_ports_dict[cov_function]['port_opt'][port_name]['weights'],
                    "shares": None,
                    "values": None,
                    "portReturn": None,
                    "transCost": None,
                    "weightDelta": None,  # compute portfolio turnover for current rebalancing period
                    "portValue": None
                }

                # calculate shares given new weight
                port_t["shares"] = port_tm1['portValue'] * port_t["weights"] / prcs_t
                port_t["portReturn"] = (port_t["weights"] * rets_tp1).sum()
                port_t['values'] = port_t['weights'] * port_tm1['portValue']

                # compute transaction by trading volume
                # redistribute money according to the new weight
                volume_buy  = np.maximum(port_t["values"] - port_tm1["values"], 0)
                volume_sell = np.maximum(port_tm1["values"] - port_t["values"], 0)
                port_t["transCost"] = (volume_buy + volume_sell).sum() * transaction_cost / 10000 # pay 10bps transaction cost


                # calculate portfolio turnover for this period
                port_t["weightDelta"] = np.sum(np.abs(port_t["weights"] - port_tm1["weights"]))

                # calculate updated portfolio at time t
                port_t["portValue"] = (port_tm1['portValue'] - port_t["transCost"]) * (1 + port_t['portReturn'])
                account_dict[cov_function][port_name].append(port_t)

            # # save efficient frontier for both ex-ante and ex-post
            # efficient_frontier = {
            #     "date": end_date_p1_str,
            #     "rets": [round(ret, 3) for ret in opt_ports_dict[cov_function]["mvo"]["rets"]],  # ex-ante annual return
            #     "stds": [round(std, 3) for std in opt_ports_dict[cov_function]["mvo"]["stds"]],  # ex-ante annual risk (std)
            #     "post_rets": [(wgt * rets_tp1).sum() for wgt in opt_ports_dict[cov_function]["mvo"]["weights"]]  # ex-post monthly return
            # }
            # efficient_frontiers[cov_function].append(efficient_frontier)

        # plot efficient frontiers among HC, GS1, and GS2
        plot_efficient_frontiers(opt_ports_dict, prefix="%s/plots" % savepath)



    # save the port  result as a pickle file
    with open("%s/result.pickle" % savepath, "wb") as f:
        pickle.dump(account_dict, f)

    # # load saved pickle file
    # with open("%s/result.pickle" % savepath, "rb") as f:
    #     account_dict = pickle.load(f)

    for cov_func in cov_function_list:
        portAccountDF = pd.DataFrame.from_dict({
            (port_name, account['date']): {
                "value": account['portValue'],
                "return": account['portReturn'],
                "trans": account['transCost'],
                "turnover": account['weightDelta']
            }
            for port_name in account_dict[cov_func].keys()
            for account in account_dict[cov_func][port_name]
        }, orient='index')


        portAccountDF.reset_index(inplace=True)
        portAccountDF.columns = ['port', 'date', 'value', 'return', 'trans', 'turnover']

        portAccountDF.pivot(index="date", columns='port', values='value'). \
            to_csv("%s/%s_value.csv" % (savepath, cov_func))
        portAccountDF.pivot(index="date", columns='port', values='return'). \
            to_csv("%s/%s_return.csv" % (savepath, cov_func))
        portAccountDF.pivot(index="date", columns='port', values='trans'). \
            to_csv("%s/%s_trans.csv" % (savepath, cov_func))
        portAccountDF.pivot(index="date", columns='port', values='turnover'). \
            to_csv("%s/%s_turnover.csv" % (savepath, cov_func))