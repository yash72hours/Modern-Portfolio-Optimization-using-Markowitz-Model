import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stocks = ['RELIANCE', 'TCS', 'JSWSTEEL', 'INFY', 'ICICIBANK', 'ITC', 'PAYTM', 'HSCL']
stocks = [stock + '.NS' for stock in stocks]

stock_data = yf.download(stocks, period=f'{int(2*252)}d')['Adj Close']
returns = stock_data.pct_change().dropna()

mean_returns = returns.mean()
cov_matrix = returns.cov()

num_portfolios = 100000

np.random.seed(42)

results = []
max_weight_per_stock = 0.25

risk_free_rate = 0.0

for i in range(num_portfolios):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)

    if np.max(weights) > max_weight_per_stock:
        continue
    
    portfolio_return = np.sum(weights * mean_returns) * 252
    downside_deviation = np.std(np.minimum(0, returns @ weights)) * np.sqrt(252)  # Downside deviation
    sortino_ratio = (portfolio_return - risk_free_rate) / downside_deviation
    
    results.append([portfolio_return, downside_deviation, sortino_ratio, weights])

results_df = pd.DataFrame(results, columns=['Return', 'Downside Deviation', 'Sortino Ratio', 'Weights'])

optimal_portfolio = results_df.loc[results_df['Sortino Ratio'].idxmax()]

results_df.dropna(inplace=True)

optimal_portfolio = results_df.loc[results_df['Sortino Ratio'].idxmax()]

print("Optimal Portfolio:")
print(optimal_portfolio)

plt.scatter(results_df['Downside Deviation'], results_df['Return'], c=results_df['Sortino Ratio'], cmap='viridis')
plt.colorbar(label='Sortino Ratio')
plt.title('Efficient Frontier (Sortino Ratio)')
plt.xlabel('Downside Deviation')
plt.ylabel('Return')
plt.scatter(optimal_portfolio['Downside Deviation'], optimal_portfolio['Return'], color='red', marker='*', s=100)
plt.show()

optimal_weights_df = pd.DataFrame({'Stock': stocks, 'Optimal Weight': optimal_portfolio['Weights']})
optimal_weights_df.to_excel('optimal_weights_final_sortino.xlsx', index=False)

print("Optimal Weights:")
print(optimal_weights_df)
