import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split

Merged_DIR = 'data/merged'
Model_DIR = 'models'
Output_DIR = 'outputs/evaluation'
os.makedirs(Output_DIR, exist_ok=True)

Buoy_stations = ['46041', '46087', '46211']
Land_stations = ['DESW1', 'LAPW1', 'WPTW1']

Buoy_features = ['swh', 'mwp', 'mwd_sin', 'mwd_cos', 'u10', 'v10', 'sp', 't2m', 'tp']
Land_features = ['u10', 'v10', 'sp', 't2m', 'tp']

Buoy_targets = ['WVHT', 'DPD', 'MWD']
Land_targets = ['WSPD', 'WDIR', 'PRES']

def load_station(station, targets, features):
    path = os.path.join(Merged_DIR, f"{station}_merged.csv")
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    raw_features = [f for f in features if f not in ['mwd_sin', 'mwd_cos']]
    df = df.dropna(subset=raw_features)
    df['mwd_sin'] = np.sin(np.radians(df['mwd']))
    df['mwd_cos'] = np.cos(np.radians(df['mwd']))
    X = df[features]
    y = df[targets]
    return X, y, df.index

def evaluate_station(station, targets, features):
    print(f"\nEvaluating: {station}")
    X, y, timestamps = load_station(station, targets, features)
    results = {}

    for target in targets:
        if target in ['MWD', 'WDIR']:
            mask = y[target].notna()
            X_t = X[mask]
            y_t = y[target][mask]
            ts_t = timestamps[mask]

            X_train, X_test, _, y_test, _, ts_test = train_test_split(
                X_t, y_t, ts_t, test_size=0.2, shuffle=False
            )

            model_sin = joblib.load(os.path.join(Model_DIR, f"{station}_{target}_sin.pkl"))
            model_cos = joblib.load(os.path.join(Model_DIR, f"{station}_{target}_cos.pkl"))

            pred_sin = model_sin.predict(X_test)
            pred_cos = model_cos.predict(X_test)
            y_pred = np.degrees(np.arctan2(pred_sin, pred_cos)) % 360
            y_actual = np.degrees(np.arctan2(
                np.sin(np.radians(y_test)),
                np.cos(np.radians(y_test))
            )) % 360

            diff = np.abs(y_pred - y_actual)
            diff = np.minimum(diff, 360 - diff)
            mae = diff.mean()
            rmse = np.sqrt((diff**2).mean())

        else:
            mask = y[target].notna()
            X_t = X[mask]
            y_t = y[target][mask]
            ts_t = timestamps[mask]

            X_train, X_test, _, y_test, _, ts_test = train_test_split(
                X_t, y_t, ts_t, test_size=0.2, shuffle=False
            )

            model = joblib.load(os.path.join(Model_DIR, f"{station}_{target}.pkl"))
            y_pred = model.predict(X_test)
            mae = np.abs(y_pred - y_test).mean()
            rmse = np.sqrt(((y_pred - y_test)**2).mean())

        print(f"  {target}: MAE={mae:.3f}  RMSE={rmse:.3f}")
        results[target] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'ts_test': ts_test,
            'mae': mae,
            'rmse': rmse
        }

    return results

def plot_scatter(station, targets, results):
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        r = results[target]
        y_test = r['y_test']
        y_pred = r['y_pred']

        ax.scatter(y_test, y_pred, alpha=0.3, s=5, color='steelblue')

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect fit')

        ax.set_xlabel(f'Observed {target}')
        ax.set_ylabel(f'Predicted {target}')
        ax.set_title(f'{station} — {target}\nMAE={r["mae"]:.3f}  RMSE={r["rmse"]:.3f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(Output_DIR, f"{station}_scatter.png"), dpi=150)
    plt.show()
    print(f"  Scatter plot saved: {Output_DIR}/{station}_scatter.png")

def plot_timeseries(station, targets, results, window_days=30):
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        r = results[target]
        ts = r['ts_test']
        y_test = pd.Series(r['y_test'].values, index=ts)
        y_pred = pd.Series(r['y_pred'], index=ts)

        # Take first window_days of test set
        cutoff = ts[0] + pd.Timedelta(days=window_days)
        y_test_window = y_test[y_test.index <= cutoff]
        y_pred_window = y_pred[y_pred.index <= cutoff]

        ax.plot(y_test_window.index, y_test_window.values, label='Observed', color='steelblue', linewidth=1)
        ax.plot(y_pred_window.index, y_pred_window.values, label='Predicted', color='tomato', linewidth=1, alpha=0.8)
        ax.set_ylabel(target)
        ax.set_title(f'{station} — {target} (first {window_days} days of test set)')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(Output_DIR, f"{station}_timeseries.png"), dpi=150)
    plt.show()
    print(f"  Time series plot saved: {Output_DIR}/{station}_timeseries.png")

def plot_metrics_table(all_results):
    rows = []
    for station, targets, results in all_results:
        for target in targets:
            r = results[target]
            rows.append({
                'Station': station,
                'Variable': target,
                'MAE': round(r['mae'], 3),
                'RMSE': round(r['rmse'], 3)
            })

    df = pd.DataFrame(rows)
    print("\nMetrics Summary:")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(Output_DIR, 'metrics_summary.csv'), index=False)
    print(f"\n  Metrics table saved: {Output_DIR}/metrics_summary.csv")
    
    fig, ax = plt.subplots(figsize=(8, len(rows) * 0.5 + 1))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Model Performance Summary', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(Output_DIR, 'metrics_summary.png'), dpi=150)
    plt.show()
    print(f"  Metrics table figure saved: {Output_DIR}/metrics_summary.png")

def plot_seasonal(station, targets, results):
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall',   10: 'Fall',  11: 'Fall'}

    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        r = results[target]
        ts = r['ts_test']
        y_test = pd.Series(r['y_test'].values, index=ts)
        y_pred = pd.Series(r['y_pred'], index=ts)

        seasonal_mae = {}
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            mask = ts.month.map(season_map) == season
            if mask.sum() < 10:
                continue
            if target in ['MWD', 'WDIR']:
                diff = np.abs(y_pred[mask].values - y_test[mask].values)
                diff = np.minimum(diff, 360 - diff)
                seasonal_mae[season] = diff.mean()
            else:
                seasonal_mae[season] = np.abs(y_pred[mask].values - y_test[mask].values).mean()

        colors = ['steelblue', 'mediumseagreen', 'tomato', 'goldenrod']
        ax.bar(seasonal_mae.keys(), seasonal_mae.values(), color=colors[:len(seasonal_mae)])
        ax.set_title(f'{station} — {target}\nSeasonal MAE')
        ax.set_ylabel('MAE')

    plt.tight_layout()
    plt.savefig(os.path.join(Output_DIR, f"{station}_seasonal.png"), dpi=150)
    plt.show()
    print(f"  Seasonal plot saved: {Output_DIR}/{station}_seasonal.png")

if __name__ == '__main__':
    print("Starting evaluation...")
    print("-" * 40)

    all_results = []

    print("\n BUOY STATIONS")
    for station in Buoy_stations:
        results = evaluate_station(station, Buoy_targets, Buoy_features)
        plot_scatter(station, Buoy_targets, results)
        plot_timeseries(station, Buoy_targets, results)
        plot_seasonal(station, Buoy_targets, results)
        all_results.append((station, Buoy_targets, results))

    print("\n LAND STATIONS")
    for station in Land_stations:
        results = evaluate_station(station, Land_targets, Land_features)
        plot_scatter(station, Land_targets, results)
        plot_timeseries(station, Land_targets, results)
        plot_seasonal(station, Land_targets, results)
        all_results.append((station, Land_targets, results))

    print("\n METRICS SUMMARY")
    plot_metrics_table(all_results)

    print("\nDone! All outputs saved to:", Output_DIR)