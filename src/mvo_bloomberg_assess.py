from portfolio_analysze import evaluate_port_performance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns.set_style("white")
sns.set_context("talk", font_scale=0.8)

color_maps = {
    "HC":  "tab:blue",
    "SM":  "tab:brown",
    "GS1": "tab:red",
    "GS2": "tab:green",
}

markers_maps = {
    "HC":  "o",
    "SM":  "p",
    "GS1": "s",
    "GS2": "D"
}

prefix = "results/bloomberg_1990_2020/bloomberg_2yr_ts_0.70"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assess Bloomberg Performance")
    parser.add_argument("--prefix", type=str, default=None)
    args = parser.parse_args()
    if args.prefix is not None:
        prefix = args.prefix

    ports_dict = {
        "HC":  "%s/HC_value.csv" % prefix,
        "SM": "%s/SM_value.csv" % prefix,
        "GS1": "%s/GS1_value.csv" % prefix,
        "GS2": "%s/GS2_value.csv" % prefix,
    }

    turnovers_dict = {
        "HC":  "%s/HC_turnover.csv" % prefix,
        "SM": "%s/SM_turnover.csv" % prefix,
        "GS1": "%s/GS1_turnover.csv" % prefix,
        "GS2": "%s/GS2_turnover.csv" % prefix,
    }

    port_names_dict = {
        "03pct": "Ultra Conservative\n3%",
        "06pct": "Conservative\n6%",
        "09pct": "Moderate\n9%",
        "12pct": "Aggressive\n12%",
        "15pct": "Ultra Aggressive\n15%",
        #"maxSharpe": "max Sharpe"
    }

    # compute portfolio performance
    df_ports_list = []
    for key, value in ports_dict.items():
        print("Evaluating %s performance ..." % key)
        df_port_perform = evaluate_port_performance(value)
        df_port_perform.columns = pd.MultiIndex.from_tuples(
            map(lambda x: (x, key), df_port_perform.columns)
        )
        df_ports_list.append(df_port_perform)

    # compute portfolio turnover
    for i, (key, value) in enumerate(turnovers_dict.items()):
        df_turnover = pd.read_csv(value, parse_dates=["date"]).set_index("date")
        df_ports_list[i].loc["Turnover"] = [round(val, 2) for val in df_turnover.mean() * 12]

    # evaluate the performance for each portfolio
    df_port_performance = pd.concat(df_ports_list, axis=1)

    # plot the ex-post efficient frontier
    plt.figure(figsize=(8, 6))
    for cov in color_maps.keys():
        risk_columns = [("%02dpct" % rsk, cov) for rsk in range(3, 16, 2)]
        rets = df_port_performance.loc["Geometric Return (%)", risk_columns].values
        stds = df_port_performance.loc["Annualized STD (%)", risk_columns].values
        plt.plot(stds, rets, "-%s" % markers_maps[cov], color=color_maps[cov], label=cov)
        if cov == "HC":
            for idx, rsk in enumerate(risk_columns):
                plt.text(stds[idx], rets[idx]-0.3, range(3, 16, 2)[idx],  verticalalignment='bottom', fontsize=12)
        plt.legend(loc="best")
        plt.xlabel("Annualized Volatility (%)")
        plt.ylabel("Annualized Return (%)")
        plt.ylim((5, 10.5))
    sns.despine()
    plt.savefig("./%s/ex_post.pdf" % prefix)
    plt.close()

    # export performance as a *.tex file
    risks_columns = [("%02dpct" % rsk, cov) for rsk in [3, 6, 9, 12, 15] for cov in color_maps.keys()]
    #ports_columns = [(port, cov) for port in ['maxSharpe'] for cov in color_maps.keys()] # remove maxSharpe
    selected_columns = risks_columns #+ ports_columns
    df_port_performance[selected_columns].to_latex("./%s/performance.tex" % prefix)

    # plot comparision between cumulative return among portfolios
    df_cumu_port = df_port_performance.loc["Cumulative Return (%)", selected_columns].to_frame().reset_index()
    df_cumu_port.columns = ["Portfolio", "Method", "Cumulative Return (%)"]
    df_cumu_port.Portfolio = [port_names_dict[port] for port in df_cumu_port.Portfolio] # map to labels

    g = sns.catplot(x="Portfolio", y="Cumulative Return (%)", hue="Method", data=df_cumu_port, kind="bar",
                      palette=sns.color_palette(list(color_maps.values())), height=4, aspect=9.5/4)
    g._legend.set_title(None)
    g.set_xlabels("")
    g.set(ylim=(0, 2000))
    g.savefig("./%s/cumulative_return.pdf" % prefix)
    plt.close()


    # export cumulative value in dollars
    df_values_list = []
    for key, value in ports_dict.items():
        df_value_perform = pd.read_csv(value, parse_dates=["date"]).set_index("date").tail(1)
        df_value_perform.columns = pd.MultiIndex.from_tuples(
            map(lambda x: (x, key), df_value_perform.columns)
        )
        df_values_list.append(df_value_perform)
    df_values = pd.concat(df_values_list, axis=1)[selected_columns].loc["2020-12-31"].to_frame().reset_index()
    df_values.columns = ["Portfolio", "Method", "Value"]
    df_values.Portfolio = [port_names_dict[port] for port in df_values.Portfolio] # map to labels
    df_values = df_values.pivot(index="Portfolio", columns="Method").round(2)
    df_values.columns = df_values.columns.droplevel()
    df_values.loc[port_names_dict.values(), [key for key in ports_dict.keys()]].\
        to_latex("./%s/cumulative_value.tex" % prefix)