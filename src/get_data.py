"""
Name    : download_yahoo.py
Author  : Yinsen Miao
Contact : yinsenm@gmail.com
Time    : 7/1/2021
Desc    : Download stock data from Yahoo.com
"""
import pandas as pd
from yahoofinancials import YahooFinancials
from datetime import datetime


def get_stock_data(ticker: str, start_date: str, end_date: str, time_interval: str) -> pd.DataFrame:
    '''
    :param ticker: stock ticker
    :param start_date: start_end in %Y-%m-%d format
    :param end_date: end_date in %Y-%m-%d format
    :param time_interval: data frequency in either daily or weekly
    '''
    stock = YahooFinancials(ticker)
    data = stock.get_historical_price_data(start_date=start_date,
                                           end_date=end_date,
                                           time_interval=time_interval)

    dat = pd.DataFrame(data[ticker]['prices']).drop(['date', 'adjclose'], axis=1)
    dat.dropna(inplace=True)  # drop row with na to remove non trading days
    dat['formatted_date'] = pd.to_datetime(dat['formatted_date'], format="%Y-%m-%d")
    dat.set_index('formatted_date', inplace=True)
    dat.index.name = ""
    dat.columns = ['High', "Low", "Open", "Close", "Volume"]
    dat = dat[["Open", 'High', "Low", "Close", "Volume"]]
    return dat


# test the download function
if __name__ == "__main__":
    ticker = "TSLA"
    start_date = "2018-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    time_interval = "Daily"
    stock_df = get_stock_data(ticker, start_date=start_date, end_date=end_date, time_interval=time_interval)
    print(stock_df)