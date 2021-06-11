from util import get_mean_variance_space, plot_efficient_frontiers, get_frontier_limits
import pandas as pd
import numpy as np
import pickle
import os
from datetime import timedelta
from tqdm import tqdm
import argparse
DEBUG = 0

# define global variables
target_volatilities_array = np.arange(8, 31) / 100.  # target volatility level from 8% to 30%
obj_function_list = ['equalWeighting', 'minVariance', 'maxReturn',  # optimization target
                     'maxSharpe', 'maxSortino', 'riskParity']
cov_function_list = ["HC", "SM", "GS1", "GS2"]  # list of covariance function

# trading setting
cash_start = 100000.
lookback_win_in_year = 3
lookback_win_size = 12 * lookback_win_in_year
start_date = "1990-01-01"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse parameter for DJIA index")
    parser.add_argument("-s", "--gs_threshold", type=float, default=0.5)
    parser.add_argument("-o", "--optimization_cost", type=float, default=20)
    parser.add_argument("-t", "--transaction_cost", type=float, default=10)
    args = parser.parse_args()

    gs_threshold = args.gs_threshold  # threshold for gerber statistics
    optimization_cost = args.optimization_cost  # penalty for excessive transaction
    transaction_cost = args.transaction_cost  # actual transaction fee in trading simulation
    print("GS threshold %.2f" % gs_threshold)
    print("Optimization cost %02d\nTransaction cost %02d" % (optimization_cost, transaction_cost))

    savepath = "./results/DJIA/%dyr_c%02dbps_t%02dbps/gs_threshold%.2f" % \
               (lookback_win_in_year, optimization_cost, transaction_cost, gs_threshold)  # small threshold

    # create folder to save results
    os.makedirs("%s" % savepath, exist_ok=True)
    os.makedirs("%s/plots" % savepath, exist_ok=True)  # save plots of efficient frontier
    os.makedirs("%s/caches" % savepath, exist_ok=True)  # save cached weights

    # transaction scheme in basis point (bs)
    # Consider transaction cost as a fraction of traded value
    # Based on chart from Figure 4 on Page 43 of Charles M. Jones' paper entitled
    # “A Century of Stock Market Liquidity and Transaction Costs”, we can simplify and approximate the trend over the decades with
    # a linear model with transaction cost of about 1.2% around early 1960s that goes down to 0.2% around 2000.
    # Considering that the Jones paper doesn't cover the 2000 to 2020 period,
    # this would further imply trading cost was assumed 0.2% after 2000.
    # transact_scheme_dict = {
    #     year: 120 - 2.5 * (year - 1960) if year < 2000 else 20 for year in range(1960, 2021)
    # }

    transact_scheme_dict = {
        year: transaction_cost for year in range(1960, 2021)
    }

    # load DJIA data
    with open("./data/dow30/djia.pickle", 'rb') as f:
        djia_dict = pickle.load(f)
    rets = djia_dict["rets"]  # DJIA total returns
    djia = djia_dict["djia"]  # DJIA indicators
    djia = djia[start_date:]
    nT, p = djia.shape
    djia_permnos = list(set(djia.columns))  # stock permno id

    # generate portfolio name
    port_names = obj_function_list + ['%02dpct' % int(tgt * 100) for tgt in target_volatilities_array]
    djia_rebalacing_dates = djia.index
    port_start_date = djia_rebalacing_dates[0]

    # initialize portfolio
    account_dict = {}
    for cov_function in cov_function_list:
        account_dict[cov_function] = {}
        for port_name in port_names:
            account_dict[cov_function][port_name] = []
            account_dict[cov_function][port_name].append(
                {
                    "date": port_start_date,
                    # portfolio weight
                    "weights": {permno: 0. for permno in djia_permnos},
                    "returns": {permno: 0. for permno in djia_permnos},
                    # track dollar value for each stock seperately
                    "values": {permno: 0. for permno in djia_permnos},  # dollar value per permno
                    "portReturn": 0,
                    "transCost": 0,
                    "weightDelta": 0,  # compute portfolio turnover for current rebalancing period
                    "portValue": cash_start,
                }
            )

    # from datetime import datetime
    # port_rebalance_date = datetime.strptime("20170929", "%Y%m%d")
    # simulate MVO portfolio

    # keep track of previous optimal weights to penalize extensive turnover
    prev_port_weights_dict = {key: None for key in cov_function_list}
    for port_rebalance_date in tqdm(djia_rebalacing_dates[:-1]):
        djia_component = djia.loc[port_rebalance_date].dropna().index.to_list()

        # filter out monthly return of the active assets
        hist_ret_df = rets.loc[: port_rebalance_date, djia_component].tail(lookback_win_size).fillna(0.)
        futr_ret_df = rets.loc[port_rebalance_date + timedelta(weeks=4):].head(1).fillna(0.) # realized total rets
        dict_futr_ret = futr_ret_df.to_dict("records")[0]

        port_bgn_date, port_end_date = hist_ret_df.index[0], hist_ret_df.index[-1]
        port_cur_date = futr_ret_df.index[0]

        if DEBUG:
            _nT, _ = hist_ret_df.shape
            print("MVO optimization from date [%s, %s] (n=%d) to apply weights on %s" % \
                  (port_bgn_date.strftime("%Y%m%d"),
                   port_end_date.strftime("%Y%m%d"),
                   _nT,
                   port_cur_date.strftime("%Y%m%d")))

        # save optimum MVO portfolio
        opt_ports_dict = {}

        # loop through each covariance function
        for cov_function in cov_function_list:
            try:
                # load saved weights
                with open("%s/caches/%s.pickle" % (savepath, port_end_date.strftime("%Y%m%d")), "rb") as f:
                    opt_ports_dict[cov_function] = pickle.load(f)[cov_function]

            except FileNotFoundError:
                # if caches not found then run MVO optimizer
                opt_ports_dict[cov_function] = get_mean_variance_space(hist_ret_df, target_volatilities_array,
                                                                       obj_function_list, cov_function,
                                                                       prev_port_weights=prev_port_weights_dict[cov_function],
                                                                       gs_threshold=gs_threshold,
                                                                       cost=optimization_cost)

            prev_port_weights_dict[cov_function] = opt_ports_dict[cov_function]["port_opt"]
            # loop through each portfolio under a covariance function
            for port_name in port_names:
                # get complete portfolio weight from each portfolio type
                _opt_port_wgt = {
                    permno: weight for (permno, weight) in zip(
                        opt_ports_dict[cov_function]['tickers'],
                        opt_ports_dict[cov_function]['port_opt'][port_name]['weights']
                    )
                }

                # complete weight for all assets
                opt_port_wgt = {
                    permno: _opt_port_wgt[permno] if permno in _opt_port_wgt.keys() else 0. for permno in djia_permnos
                }

                # get old portfolio
                port_tm1 = account_dict[cov_function][port_name][-1]

                # rebalance our portfolio
                port_t = {
                    "date": port_cur_date,
                    "weights": opt_port_wgt,
                    "returns": None,
                    "values": None,
                    "portReturn": None,
                    "transCost": None,
                    "portValue": None,
                }

                # redistribute money according to the new weight
                port_t['values'] = {
                    permno: port_t['weights'][permno] * port_tm1['portValue'] for permno in djia_permnos
                }

                # calculate transaction cost for rebalancing portfolio
                value_buy = {permno: max(port_t["values"][permno] - port_tm1["values"][permno], 0) for permno in djia_permnos}
                value_sell = {permno: max(port_tm1["values"][permno] - port_t["values"][permno], 0) for permno in djia_permnos}

                # update value for each permno after discounting by the transaction cost
                temp = {
                    permno: port_t['values'][permno] - (value_buy[permno] + value_sell[permno]) * transact_scheme_dict[port_cur_date.year] / 10000.
                    for permno in djia_permnos
                }
                port_t['values'] = temp

                # update total transaction cost for rebalancing portfolio
                port_t["transCost"] = (sum(value_buy.values()) + sum(value_sell.values())) * \
                                      transact_scheme_dict[port_cur_date.year] / 10000.  # divide by bps

                # calculate return for each permno
                port_t['returns'] = dict_futr_ret

                # calculate value for each permno after accumulating in $ values
                temp = {
                    permno: port_t['values'][permno] * (1 + port_t['returns'][permno])
                    for permno in djia_permnos
                }
                port_t['values'] = temp

                port_t["portReturn"] = sum([value * port_t["weights"][permno] for permno, value in port_t['returns'].items()])

                # calculate portfolio value by summing up value per permno
                port_t['portValue'] = sum(port_t['values'].values())

                # calculate portfolio turnover for this period
                port_t["weightDelta"] = np.sum(np.abs([port_t["weights"][permno] - port_tm1["weights"][permno] for permno in djia_permnos]))

                # append update account
                account_dict[cov_function][port_name].append(port_t)

        # plot efficient frontiers among HC, GS1, and GS2
        plot_efficient_frontiers(opt_ports_dict, prefix="%s/plots" % savepath)

        # save MVO into cache
        with open("%s/caches/%s.pickle" % (savepath, port_end_date.strftime("%Y%m%d")), 'wb') as f:
            pickle.dump(opt_ports_dict, f)

    # save the portfolio as a pickle file
    with open("%s/port_result.pickle" % savepath, "wb") as f:
        pickle.dump(account_dict, f)

    # save the results as csv file
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