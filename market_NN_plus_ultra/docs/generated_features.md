# Market NN Plus Ultra Feature Registry

| Name | Depends On | Tags | Description |
| --- | --- | --- | --- |
| bollinger_band_width | close | volatility | Relative Bollinger band width as a volatility proxy |
| fft_energy_ratio | close | spectral | Ratio of high-frequency FFT energy to total energy over 128 steps |
| log_return_1 | close | returns | One-step log return |
| log_return_5 | close | returns, multi_horizon | Five-step log return |
| macd_hist | close | trend | MACD histogram capturing trend accelerations |
| realised_vol_20 | close | volatility | Annualised realised volatility over 20 periods |
| regime_score | close | regime | Soft bull/bear regime score |
| rolling_kurtosis_30 | close | higher_moment | 30-step kurtosis of returns |
| rolling_skew_30 | close | higher_moment | 30-step skewness of returns |
| rsi_14 | close | momentum | Relative Strength Index over 14 periods |
| volume_zscore | volume | volume, regime | Rolling z-score of volume anomalies |
| vwap_ratio | high, low, close, volume | volume, intraday | Price distance from VWAP |
