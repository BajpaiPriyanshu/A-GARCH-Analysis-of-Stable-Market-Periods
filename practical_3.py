import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

TICKER = "RELIANCE.NS"
DATA_START = "2018-01-01"
DATA_END = "2019-01-15"
TARGET_DATE = "2019-01-10"
TRAINING_DAYS = 250
VAR_CONFIDENCE = 0.99
ES_CONFIDENCE = 0.975

def main():
    print("\n" + "="*80)
    print("GARCH VALIDATION STUDY: STABLE MARKET PERIOD")
    print("Normal vs Student-t Distribution Comparison")
    print("="*80)
    
    print(f"\n{'='*80}")
    print(f"FETCHING DATA FOR {TICKER}")
    print(f"{'='*80}")
    
    data = yf.download(TICKER, start=DATA_START, end=DATA_END, progress=False)
    
    if data.empty:
        raise ValueError(f"No data retrieved for {TICKER}")
    
    print(f"Columns returned: {data.columns.tolist()}")
    
    adj_close_col = None
    possible_names = ['Adj Close', 'Adj Close', 'adj close', 'AdjClose', 'adjclose']
    
    for col_name in possible_names:
        if col_name in data.columns:
            adj_close_col = col_name
            break
    
    if adj_close_col is None and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        for col_name in possible_names:
            if col_name in data.columns:
                adj_close_col = col_name
                break
    
    if adj_close_col is None:
        for col in data.columns:
            if 'adj' in str(col).lower() and 'close' in str(col).lower():
                adj_close_col = col
                break
    
    if adj_close_col is None:
        if 'Close' in data.columns:
            adj_close_col = 'Close'
            print("Warning: Using 'Close' instead of 'Adj Close'")
        else:
            raise ValueError(f"Could not find price column. Available columns: {data.columns.tolist()}")
    
    print(f"Using column: {adj_close_col}")
    
    returns = 100 * np.log(data[adj_close_col] / data[adj_close_col].shift(1))
    returns = returns.dropna()
    
    print(f"Data points retrieved: {len(data)}")
    print(f"Returns calculated: {len(returns)}")
    print(f"Date range: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING WINDOW SELECTION")
    print(f"{'='*80}")
    
    target_dt = pd.to_datetime(TARGET_DATE)
    prior_dates = returns.index[returns.index < target_dt]
    
    if len(prior_dates) < TRAINING_DAYS:
        raise ValueError(f"Insufficient data: need {TRAINING_DAYS} days, have {len(prior_dates)}")
    
    training_returns = returns.loc[prior_dates[-TRAINING_DAYS:]]
    
    print(f"Target Date: {TARGET_DATE}")
    print(f"Training Days: {TRAINING_DAYS}")
    print(f"Training Start: {training_returns.index[0].strftime('%Y-%m-%d')}")
    print(f"Training End: {training_returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"Actual training days: {len(training_returns)}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING DATA SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Mean Return: {training_returns.mean():.4f}%")
    print(f"Std Deviation: {training_returns.std():.4f}%")
    print(f"Skewness: {training_returns.skew():.4f}")
    print(f"Kurtosis: {training_returns.kurtosis():.4f}")
    print(f"Min Return: {training_returns.min():.4f}%")
    print(f"Max Return: {training_returns.max():.4f}%")
    
    print(f"\n{'='*80}")
    print(f"FITTING MODELS AND CALCULATING RISK METRICS")
    print(f"{'='*80}")
    
    variance_models = ['GARCH', 'EGARCH', 'GJR-GARCH']
    distributions = ['normal', 't']
    results = []
    
    for vol_model in variance_models:
        print(f"\n{'-'*80}")
        print(f"Variance Model: {vol_model}")
        print(f"{'-'*80}")
        
        for dist in distributions:
            print(f"\nFitting {vol_model} with {dist.upper()} distribution...")
            
            if vol_model == 'GJR-GARCH':
                model = arch_model(training_returns, 
                                  mean='Constant',
                                  vol='GARCH',
                                  p=1, o=1, q=1,
                                  dist=dist)
            elif vol_model == 'EGARCH':
                model = arch_model(training_returns,
                                  mean='Constant', 
                                  vol='EGARCH',
                                  p=1, q=1,
                                  dist=dist)
            else:
                model = arch_model(training_returns,
                                  mean='Constant', 
                                  vol='GARCH',
                                  p=1, q=1,
                                  dist=dist)
            
            fitted = model.fit(disp='off', show_warning=False)
            
            forecast = fitted.forecast(horizon=1)
            forecasted_variance = forecast.variance.values[-1, 0]
            forecasted_volatility = np.sqrt(forecasted_variance)
            forecasted_mean = forecast.mean.values[-1, 0]
            
            if dist == 'normal':
                z_var = stats.norm.ppf(1 - VAR_CONFIDENCE)
                z_es = stats.norm.ppf(1 - ES_CONFIDENCE)
                var = -(forecasted_mean + z_var * forecasted_volatility)
                es = -(forecasted_mean + z_es * forecasted_volatility)
            else:
                nu = fitted.params['nu']
                t_var = stats.t.ppf(1 - VAR_CONFIDENCE, nu)
                t_es = stats.t.ppf(1 - ES_CONFIDENCE, nu)
                scale_factor = np.sqrt((nu - 2) / nu)
                var = -(forecasted_mean + t_var * forecasted_volatility * scale_factor)
                es = -(forecasted_mean + t_es * forecasted_volatility * scale_factor)
            
            results.append({
                'Variance Model': vol_model,
                'Distribution': dist.capitalize(),
                'VaR (99%)': var,
                'ES (97.5%)': es,
                'Forecast Mean': forecasted_mean,
                'Forecast Vol': forecasted_volatility,
                'AIC': fitted.aic,
                'BIC': fitted.bic
            })
            
            print(f"  VaR (99%): {var:.4f}%")
            print(f"  ES (97.5%): {es:.4f}%")
            print(f"  AIC: {fitted.aic:.2f}")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print(f"COMPLETE RESULTS TABLE")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"SIDE-BY-SIDE COMPARISON: NORMAL vs STUDENT-T")
    print(f"{'='*80}")
    
    comparison_data = []
    
    for vol_model in variance_models:
        normal_row = results_df[(results_df['Variance Model'] == vol_model) & 
                                (results_df['Distribution'] == 'Normal')].iloc[0]
        t_row = results_df[(results_df['Variance Model'] == vol_model) & 
                          (results_df['Distribution'] == 'T')].iloc[0]
        
        var_diff = abs(normal_row['VaR (99%)'] - t_row['VaR (99%)'])
        var_pct_diff = (var_diff / normal_row['VaR (99%)']) * 100
        
        es_diff = abs(normal_row['ES (97.5%)'] - t_row['ES (97.5%)'])
        es_pct_diff = (es_diff / normal_row['ES (97.5%)']) * 100
        
        comparison_data.append({
            'Model': vol_model,
            'Normal VaR': f"{normal_row['VaR (99%)']:.4f}%",
            'Student-t VaR': f"{t_row['VaR (99%)']:.4f}%",
            'VaR Diff': f"{var_diff:.4f}%",
            'VaR % Diff': f"{var_pct_diff:.2f}%",
            'Normal ES': f"{normal_row['ES (97.5%)']:.4f}%",
            'Student-t ES': f"{t_row['ES (97.5%)']:.4f}%",
            'ES Diff': f"{es_diff:.4f}%",
            'ES % Diff': f"{es_pct_diff:.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"HYPOTHESIS TEST RESULTS")
    print(f"{'='*80}")
    
    avg_var_normal = results_df[results_df['Distribution'] == 'Normal']['VaR (99%)'].mean()
    avg_var_t = results_df[results_df['Distribution'] == 'T']['VaR (99%)'].mean()
    overall_var_pct_diff = abs(avg_var_t - avg_var_normal) / avg_var_normal * 100
    
    avg_es_normal = results_df[results_df['Distribution'] == 'Normal']['ES (97.5%)'].mean()
    avg_es_t = results_df[results_df['Distribution'] == 'T']['ES (97.5%)'].mean()
    overall_es_pct_diff = abs(avg_es_t - avg_es_normal) / avg_es_normal * 100
    
    print(f"\nAverage VaR (Normal): {avg_var_normal:.4f}%")
    print(f"Average VaR (Student-t): {avg_var_t:.4f}%")
    print(f"Overall VaR % Difference: {overall_var_pct_diff:.2f}%")
    
    print(f"\nAverage ES (Normal): {avg_es_normal:.4f}%")
    print(f"Average ES (Student-t): {avg_es_t:.4f}%")
    print(f"Overall ES % Difference: {overall_es_pct_diff:.2f}%")
    
    threshold = 0.2
    
    print(f"\n{'='*80}")
    print(f"CONCLUSION")
    print(f"{'='*80}")
    
    if overall_var_pct_diff < threshold and overall_es_pct_diff < threshold:
        conclusion = f"""
HYPOTHESIS CONFIRMED:
During the stable market period (Jan 2019), the difference between Normal 
and Student-t distributions is minimal (VaR: {overall_var_pct_diff:.2f}%, ES: {overall_es_pct_diff:.2f}%).

This supports the hypothesis that complex non-normal distributions offer 
little practical advantage over Normal distributions during calm market periods.
The added complexity of Student-t distribution does not materially improve 
risk estimates when markets are stable.
        """
    else:
        conclusion = f"""
HYPOTHESIS REJECTED:
The difference between Normal and Student-t distributions is substantial 
(VaR: {overall_var_pct_diff:.2f}%, ES: {overall_es_pct_diff:.2f}%), exceeding the {threshold}% threshold.

Even during stable market periods, the choice of distribution materially 
affects risk estimates. Student-t distribution may still provide value 
in capturing tail risk even in calm markets.
        """
    
    print(conclusion)
    
    print(f"\n{'='*80}")
    print(f"MODEL SELECTION (Information Criteria)")
    print(f"{'='*80}")
    
    best_aic = results_df.loc[results_df['AIC'].idxmin()]
    best_bic = results_df.loc[results_df['BIC'].idxmin()]
    
    print(f"\nBest Model (AIC): {best_aic['Variance Model']} with {best_aic['Distribution']} distribution")
    print(f"  AIC: {best_aic['AIC']:.2f}")
    print(f"  VaR (99%): {best_aic['VaR (99%)']:.4f}%")
    
    print(f"\nBest Model (BIC): {best_bic['Variance Model']} with {best_bic['Distribution']} distribution")
    print(f"  BIC: {best_bic['BIC']:.2f}")
    print(f"  VaR (99%): {best_bic['VaR (99%)']:.4f}%")
    
    print(f"\n{'='*80}")
    print(f"EXPORTING RESULTS")
    print(f"{'='*80}")
    
    results_df.to_csv('garch_full_results.csv', index=False)
    comparison_df.to_csv('normal_vs_studentt_comparison.csv', index=False)
    
    print("\n[SUCCESS] Results exported to:")
    print("  - garch_full_results.csv")
    print("  - normal_vs_studentt_comparison.csv")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return results_df, comparison_df


if __name__ == "__main__":
    try:
        results_df, comparison_df = main()
        print("\n[SUCCESS] Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n{'!'*80}")
        print(f"ERROR OCCURRED")
        print(f"{'!'*80}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()