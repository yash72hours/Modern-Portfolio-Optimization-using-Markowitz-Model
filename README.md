
# Detailed Explanation of Modern Portfolio Theory (MPT) and the Code Implementation

The code is used on NIFTY MIDCAP SELECT stocks in this code using 2021 to 2023 data as train set.

## Overview of Modern Portfolio Theory (MPT)
Modern Portfolio Theory (MPT) is a financial theory developed by Harry Markowitz in the 1950s,
which aims to maximize portfolio returns for a given amount of risk or minimize risk for a
given level of expected return. The key concepts behind MPT include:

1. **Diversification**: MPT emphasizes the importance of diversifying investments across 
   different assets to reduce risk. By holding a mix of assets that are not perfectly correlated,
   investors can decrease the overall volatility of their portfolio.

2. **Efficient Frontier**: This is a graphical representation of the optimal portfolios that offer
   the highest expected return for a defined level of risk. The efficient frontier is formed by
   plotting the risk (standard deviation) against the expected return of various portfolios.

3. **Risk and Return**: MPT categorizes risk into two types:
   - **Systematic Risk**: The inherent risk associated with the entire market or market segment
     (e.g., economic changes, political instability).
   - **Unsystematic Risk**: The risk specific to an individual asset, which can be mitigated 
     through diversification.

4. **Sharpe Ratio**: The Sharpe ratio measures the risk-adjusted return of a portfolio. It is 
   calculated as the excess return of the portfolio over the risk-free rate divided by the 
   portfolio's standard deviation (risk). A higher Sharpe ratio indicates a better risk-adjusted 
   performance.

## Functioning of the Code
The provided Python code implements MPT using stock price data for NIFTY MIDCAP SELECT stocks.
Here’s a breakdown of how the code works step-by-step:

### 1. Import Necessary Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
- **NumPy**: Used for numerical calculations, especially with arrays.
- **Pandas**: Utilized for data manipulation and analysis.
- **Matplotlib**: A plotting library used to create visualizations.

### 2. Load the Data
```python
file_path = 'your_file.csv'  # Replace with the actual path to your file
df = pd.read_csv(file_path, index_col='date', parse_dates=True)
```
- The code reads the CSV file containing daily close prices for NIFTY MIDCAP SELECT stocks.
- The `index_col='date'` argument sets the date column as the index, and `parse_dates=True` 
  ensures that the date column is interpreted as date objects.

### 3. Calculate Daily Returns
```python
returns = df.pct_change().dropna()
```
- The `.pct_change()` method computes the percentage change between the current and previous 
  row, effectively calculating daily returns for each stock.
- The `.dropna()` function removes any rows with missing values, ensuring the dataset is clean.

### 4. Calculate Mean Returns and Covariance Matrix
```python
mean_returns = returns.mean()
cov_matrix = returns.cov()
```
- **Mean Returns**: The average daily return for each stock is calculated using `.mean()`.
- **Covariance Matrix**: The covariance matrix captures the relationships between the returns 
  of different stocks, indicating how the returns move together.

### 5. Portfolio Simulation
```python
num_portfolios = 10000
risk_free_rate = 0.01  # Assume a risk-free rate of 1%
```
- The number of portfolios to simulate is set to 10,000. This number can be adjusted based on 
  computational capacity and desired accuracy.
- The `risk_free_rate` is assumed to be 1%, which represents the return of a risk-free asset 
  (e.g., government bonds).

### 6. Generate Random Portfolios
```python
for _ in range(num_portfolios):
    weights = np.random.random(len(df.columns))
    weights /= np.sum(weights)  # Normalize weights to sum to 1

    portfolio_return = np.sum(weights * mean_returns) * 252  # Annualized return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * 
    np.sqrt(252)  # Annualized volatility
    
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Store results
    port_returns.append(portfolio_return)
    port_volatility.append(portfolio_volatility)
    port_sharpe_ratio.append(sharpe_ratio)
    stock_weights.append(weights)
```
- For each simulated portfolio:
  - **Random Weights**: Random weights are generated for each stock, ensuring that the sum 
    of weights equals 1 (i.e., the entire capital is allocated).
  - **Portfolio Return**: The expected annualized return is calculated by taking the weighted 
    sum of the mean returns and multiplying by 252 (the typical number of trading days in a year).
  - **Portfolio Volatility**: The annualized volatility is computed using the covariance matrix 
    and the weights, incorporating the square root of 252 to annualize the standard deviation.
  - **Sharpe Ratio**: The Sharpe ratio is calculated to evaluate the portfolio’s performance 
    relative to its risk.
  - The results (returns, volatility, Sharpe ratios, and weights) are stored for later analysis.

### 7. Identify the Best Portfolio
```python
max_sharpe_idx = np.argmax(port_sharpe_ratio)
max_sharpe_return = port_returns[max_sharpe_idx]
max_sharpe_volatility = port_volatility[max_sharpe_idx]
max_sharpe_weights = stock_weights[max_sharpe_idx]
```
- The portfolio with the highest Sharpe ratio is identified using `np.argmax()`, which returns 
  the index of the maximum value.
- The corresponding return, volatility, and weights for the best portfolio are extracted.

### 8. Print Best Portfolio Weights
```python
print("Best Portfolio Weights (Max Sharpe Ratio):")
for stock, weight in zip(df.columns, max_sharpe_weights):
    print(f"{stock}: {weight:.4f}")
```
- The weights of the best portfolio are printed, providing insight into how capital is allocated 
  among the stocks.

### 9. Plotting the Efficient Frontier
```python
plt.figure(figsize=(10, 6))
plt.scatter(port_volatility, port_returns, c=port_sharpe_ratio, cmap='viridis', marker='o', alpha=0.5, label='All Portfolios')
plt.colorbar(label='Sharpe Ratio')

plt.scatter(max_sharpe_volatility, max_sharpe_return, marker='*', color='r', s=200, label='Max Sharpe Ratio Portfolio')

plt.title('Efficient Frontier with Best Portfolio Highlighted')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.legend()
plt.show()
```
- A scatter plot is created to visualize all simulated portfolios, with color indicating the 
  Sharpe ratio.
- The best portfolio (with the highest Sharpe ratio) is highlighted using a large red star marker.
- Labels and legends are added to improve the clarity of the plot.

## Conclusion
This code effectively implements the core principles of Modern Portfolio Theory (MPT) to analyze
and visualize the performance of a portfolio of NIFTY MIDCAP SELECT stocks. It allows investors to 
make informed decisions based on risk-return trade-offs and highlights the importance of 
diversification in portfolio construction. By identifying the optimal portfolio with the highest 
Sharpe ratio, investors can aim for the best risk-adjusted returns in their investment strategies.
