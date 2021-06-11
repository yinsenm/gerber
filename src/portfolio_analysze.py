import pandas as pd
import numpy as np
from pyfinance import TSeries, TFrame
from pyfinance.datasets import load_rf, load_shiller
from scipy.stats import norm
from get_data import get_stock_data
from datetime import date

"""
Reference:
- Computer Value at Risk
    - a. https://www.interviewqs.com/blog/value-at-risk
    - b. https://blog.quantinsti.com/calculating-value-at-risk-in-excel-python/
    - c. https://www.investopedia.com/articles/04/101304.asp
- Other performance metrics
    - https://github.com/bsolomon1124/pyfinance
    - https://mp.weixin.qq.com/s/IguEpW9bT8NdnPIV_IyINg
"""


def evaluate_port_performance(filename: str = "./results/bloomberg_2yr/GS1_value.csv"):
    """
    :param filename: historical portfolio values
    :return: pd.DataFrame of assessed portfolio performance
    """
    prcs = pd.read_csv(filename, parse_dates=['date'], index_col=['date']).\
        resample("M", label="right").last()
    lrets = np.log(prcs).diff().dropna()

    # # download sp500 index as benchmark
    # sp500_pre_2020 = pd.read_csv("./data/sp500/dsp500.csv", parse_dates=['caldt'], index_col=['caldt']).\
    #     dropna()["vwretd"].resample("M", label="right").apply(lambda x: np.prod(x + 1.0) - 1.0)
    #
    # sp500_post_2020 = get_stock_data("SPY",
    #                        start_date="2019-12-01",
    #                        end_date=date.today().strftime("%Y-%m-%d"),
    #                        time_interval='monthly').\
    #     resample("M", label="right").last()['Close'].pct_change().dropna()
    rf = load_rf(freq="M")  # load 3-month treasury bill as risk free rate

    # create an empty dataframe to save performance of each portfolio
    df_port_performances = pd.DataFrame(index=[
        'Arithmetic Return (%)',
        'Geometric Return (%)',
        'Cumulative Return (%)',
        # 'Alpha',
        # 'Beta',
        'Annualized STD (%)',
        'Annualized Skewness',
        'Annualized Kurtosis',
        # 'Downsided Annualized STD (%)',
        'Maximum Drawdown (%)',
        'Monthly 95% VaR (%)',
        'Shape Ratio',
        # 'Calmar Ratio',
        # 'Sortino Ratio',
        # 'Information Ratio',
        # 'Treynor Ratio'
    ])


    for port_name in prcs.columns:
        prc = prcs[port_name]
        ret = prc.pct_change().dropna()
        ts = TSeries(ret, freq='M')  # calculate monthly return

        # # subset relevant period of monthly SP500 return as benchmark
        # sp500 = pd.concat([sp500_pre_2020, sp500_post_2020]).loc[ts.index]

        annualized_ret = ts.anlzd_ret()  # annualized return
        annualized_std = ts.anlzd_stdev()  # annualized standard deviation
        annualized_semi_std = ts.semi_stdev()  # stdev of downside returns
        # https://quant.stackexchange.com/questions/3667/skewness-and-kurtosis-under-aggregation

        annualized_skewness = (12 * lrets[port_name]).skew()
        annualized_kurtosis = (12 * lrets[port_name]).kurtosis()

        cumulative_ret = ts.cuml_ret()   # cumulative return
        annual_rets = ts.rollup('A')     # return by years

        n = annual_rets.shape[0] # extract number of years
        annual_arithmetic_ret = annual_rets.mean()
        annual_geometric_ret = ((annual_rets + 1).prod()) ** (1.0 / n) - 1.0

        # Calculating VaR using Variance-Covariance approach
        VaR1 = norm.ppf(1 - 0.95, ret.mean(), ret.std()) # 95% monthly VaR
        # Historical Simulation approach
        VaR2 = ret.quantile(0.05) # 95% monthly VaR

        sharpe_ratio = ts.sharpe_ratio(rf=rf)  # annualized Sharpe ratio
        sortino_ratio = ts.sortino_ratio()
        calmar_ratio = ts.calmar_ratio()
        maximum_drawdown = ts.max_drawdown()  # maximum drawdown

        # # CAPM model to compute alpha and beta
        # alpha = ts.alpha(sp500)
        # beta = ts.beta(sp500)
        # rsq = ts.rsq(sp500)
        # bat = ts.batting_avg(sp500)
        # pct_neg = ts.pct_negative()
        # pct_pos = ts.pct_positive()  # price change positively
        # ir = ts.info_ratio(sp500)
        # tr = ts.treynor_ratio(sp500)

        """
        'Arithmetic Return (%)',
        'Geometric Return (%)',
        'Cumulative Return (%)',
        'Annualized STD (%)',
        'Annualized Skewness',
        'Annualized Kurtosis',
        'Maximum Drawdown (%)',
        'Monthly 95% VaR (%)',
        'Shape Ratio',
        'Calmar Ratio',
        'Sortino Ratio',
        """

        df_port_performances[port_name] = [
            annual_arithmetic_ret * 100,
            annual_geometric_ret * 100,
            cumulative_ret * 100,
            # alpha, beta,
            annualized_std * 100,
            annualized_skewness,
            annualized_kurtosis,
            # annualized_semi_std * 100,
            maximum_drawdown * 100,
            VaR2 * 100,
            sharpe_ratio,
            # calmar_ratio,
            # sortino_ratio,
            # ir, tr,
        ]
    return df_port_performances.round(2)


if __name__ == "__main__":
    filename = "./results/djia_3yr/HC_value.csv"
    df_port_performances = evaluate_port_performance(filename=filename)
    print(df_port_performances)