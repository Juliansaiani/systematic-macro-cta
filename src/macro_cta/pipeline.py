"""Evaluate fixed-rule ETF trend signals with matched dates and explicit accounting."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_market_data
from .models import compute_ewma_tsmom_signals
from .portfolio import (
    compute_positions,
    simulate_strategy,
    rolling_positions,
    score_performance,
)


def verify_reference(actual: pd.DataFrame, expected: pd.DataFrame, tolerance: float = 1e-6) -> pd.DataFrame:
    """Verify computed metrics against reference baseline."""
    keys = ["variant", "cost_bps"]
    metrics = ["sharpe", "cagr", "ann_vol", "max_drawdown"]
    for frame in (actual, expected):
        missing = set(keys + metrics) - set(frame.columns)
        if missing or frame.empty:
            raise ValueError(f"Missing required reference metrics or empty comparison: {missing}")
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("Reference comparison keys must be unique.")

    compared = actual.merge(expected, on=keys, how="outer", suffixes=("_new", "_reference"), indicator=True)
    if not compared["_merge"].eq("both").all():
        raise AssertionError("Reference and computed result rows differ.")

    report = compared[keys].copy()
    for metric in metrics:
        err = (compared[f"{metric}_new"] - compared[f"{metric}_reference"]).abs()
        report[f"{metric}_error"] = err
        if not np.isfinite(err).all() or err.max() > tolerance:
            raise AssertionError(f"Reference mismatch: {metric}, largest error={err.max()}")
    return report


def generate_figures(series_dict: dict, crisis_df: pd.DataFrame, output_dir: Path) -> None:
    """Export 2-panel Crisis Alpha summary figure and 3-panel crisis trajectory figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter, MaxNLocator

    plt.style.use("default")
    cta_s = series_dict.get("Daily expanding (5 bps)", series_dict.get("Systematic Trend (Net 5 bps)"))
    b6040_s = series_dict.get("60/40 Portfolio", series_dict.get("Global 60/40 Benchmark"))
    eq_s = series_dict.get("Equity basket", series_dict.get("Global Equity Basket"))

    def get_dd(s):
        if s is None or len(s) == 0:
            return 0.0
        c = np.cumprod(1.0 + s.values)
        p = np.maximum.accumulate(c)
        return float(((c - p) / p).min()) if len(c) > 0 else 0.0

    crises = {
        "China/Oil Shock\n(2015-16)": ("2015-08-03", "2016-02-29"),
        "COVID-19 Crash\n(2020)": ("2020-02-03", "2020-03-31"),
        "Rate Shock\n(2022)": ("2022-01-03", "2022-12-30"),
    }
    labels = list(crises.keys())
    ret_trend, ret_6040, ret_eq = [], [], []
    dd_trend, dd_6040, dd_eq = [], [], []

    for lbl, (s_dt, e_dt) in crises.items():
        st = cta_s.loc[s_dt:e_dt] if cta_s is not None and len(cta_s) > 0 else pd.Series(dtype=float)
        ret_trend.append(float((1.0 + st).prod() - 1.0) * 100 if len(st) > 0 else 0.0)
        dd_trend.append(get_dd(st) * 100)
        if b6040_s is not None:
            s6 = b6040_s.loc[s_dt:e_dt]
            ret_6040.append(float((1.0 + s6).prod() - 1.0) * 100 if len(s6) > 0 else 0.0)
            dd_6040.append(get_dd(s6) * 100)
        else:
            ret_6040.append(0.0)
            dd_6040.append(0.0)
        if eq_s is not None:
            se = eq_s.loc[s_dt:e_dt]
            ret_eq.append(float((1.0 + se).prod() - 1.0) * 100 if len(se) > 0 else 0.0)
            dd_eq.append(get_dd(se) * 100)
        else:
            ret_eq.append(0.0)
            dd_eq.append(0.0)

    # Figure 1: Side-by-side Crisis Returns & Crisis Max Drawdowns (overview.png)
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(14.5, 5.5))
    x = np.arange(len(labels))
    width = 0.25

    b1 = ax2.bar(x - width, ret_trend, width, label="Systematic Trend (Net 5 bps)", color="#2ca02c")
    b2 = ax2.bar(x, ret_6040, width, label="Global 60/40", color="#7f7f7f")
    b3 = ax2.bar(x + width, ret_eq, width, label="Global Equities", color="#d62728")
    ax2.axhline(0, color="black", lw=0.9)
    ax2.set_title("Crisis Period Total Return (%)", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_ylabel("Total Return (%)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylim(-26, 36)
    ax2.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax2.grid(axis="y", alpha=0.25, linestyle=":")
    ax2.spines[["top", "right"]].set_visible(False)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 1.2 if h >= 0 else -1.6
            ax2.text(bar.get_x() + bar.get_width() / 2., h + offset, f"{h:+.2f}%", ha="center", va=va, fontsize=8.2, fontweight="bold", color=bar.get_facecolor())

    b1_dd = ax3.bar(x - width, dd_trend, width, label="Systematic Trend (Net 5 bps)", color="#2ca02c")
    b2_dd = ax3.bar(x, dd_6040, width, label="Global 60/40", color="#7f7f7f")
    b3_dd = ax3.bar(x + width, dd_eq, width, label="Global Equities", color="#d62728")
    ax3.axhline(0, color="black", lw=0.9)
    ax3.set_title("Crisis Maximum Drawdown (%)", fontsize=11.5, fontweight="bold", pad=10)
    ax3.set_ylabel("Max Drawdown (%)", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9.5)
    ax3.set_ylim(-40, 6)
    ax3.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax3.grid(axis="y", alpha=0.25, linestyle=":")
    ax3.spines[["top", "right"]].set_visible(False)
    for bars in [b1_dd, b2_dd, b3_dd]:
        for bar in bars:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., h - 1.6, f"{h:.2f}%", ha="center", va="top", fontsize=8.2, fontweight="bold", color=bar.get_facecolor())

    handles, names = ax2.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Systematic Macro CTA Trend: Convex Crisis Alpha & Tail-Risk Protection", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Export dedicated 3-panel Crisis Trajectories figure
    fig_crises, ax_arr = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for i, (title, (s, e)) in enumerate([
        ("China / Oil Shock", ("2015-08-03", "2016-02-29")),
        ("COVID-19 Crash", ("2020-02-03", "2020-03-31")),
        ("2022 Global Rate Shock", ("2022-01-03", "2022-12-30")),
    ]):
        ax_c = ax_arr[i]
        st = cta_s.loc[s:e] if cta_s is not None else pd.Series(dtype=float)
        s6 = b6040_s.loc[s:e] if b6040_s is not None else pd.Series(dtype=float)
        se = eq_s.loc[s:e] if eq_s is not None else pd.Series(dtype=float)
        t_ret = f"{(1.0 + st).prod() - 1.0:+.2%}" if len(st) > 0 else "N/A"
        b_ret = f"{(1.0 + s6).prod() - 1.0:+.2%}" if len(s6) > 0 else "N/A"
        eq_r = f"{(1.0 + se).prod() - 1.0:+.2%}" if len(se) > 0 else "N/A"

        if len(st) > 0:
            d = st.index
            p_dates = pd.DatetimeIndex([d[0] - pd.offsets.BDay(1)] + list(d))
            cum_t = np.r_[100.0, np.cumprod(1.0 + st.values) * 100.0]
            ax_c.plot(p_dates, cum_t, color="#2ca02c", lw=2.2, label=f"Systematic Trend ({t_ret})")
        if len(s6) > 0:
            cum_6 = np.r_[100.0, np.cumprod(1.0 + s6.values) * 100.0]
            ax_c.plot(p_dates, cum_6, color="#7f7f7f", lw=1.6, ls="--", label=f"Global 60/40 ({b_ret})")
        if len(se) > 0:
            cum_e = np.r_[100.0, np.cumprod(1.0 + se.values) * 100.0]
            ax_c.plot(p_dates, cum_e, color="#d62728", lw=1.5, ls=":", label=f"Global Equities ({eq_r})")

        ax_c.axhline(100.0, color="black", lw=0.8, alpha=0.6)
        ax_c.set_title(f"{title}\n({s} to {e})", fontsize=11, fontweight="bold", pad=10)
        ax_c.set_ylabel("Indexed Cumulative Return (Base 100)", fontsize=9.5)
        handles_c, _ = ax_c.get_legend_handles_labels()
        if handles_c:
            ax_c.legend(loc="lower left" if i != 1 else "upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)
        ax_c.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%b %y" if i == 2 else "%b %d"))
        ax_c.grid(alpha=0.25, linestyle=":")
        ax_c.spines[["top", "right"]].set_visible(False)

    fig_crises.suptitle("Crisis Alpha Dynamics: Daily Cumulative Performance During Market Drawdowns", fontsize=13, fontweight="bold", y=1.02)
    fig_crises.tight_layout()
    fig_crises.savefig(output_dir / "crisis_episodes.png", dpi=160, bbox_inches="tight")
    plt.close(fig_crises)

    # Export Full-Sample Cumulative Wealth figure (full_sample.png)
    fig_full, ax_f = plt.subplots(figsize=(13.5, 5.2))
    monthly_s = series_dict.get("Monthly rolling (5 bps)")
    for series, color, ls, lw, lbl in [
        (cta_s, "#1f77b4", "-", 2.0, "Daily expanding (Net 5 bps)"),
        (monthly_s, "#2ca02c", "-", 1.8, "Monthly rolling 21d (Net 5 bps)"),
        (b6040_s, "#7f7f7f", "--", 1.5, "Global 60/40 (Gross)"),
        (eq_s, "#d62728", ":", 1.4, "Equity basket (Gross)"),
    ]:
        if series is not None and len(series) > 0:
            cum = np.r_[100.0, np.cumprod(1.0 + series.values) * 100.0]
            dates_full = pd.DatetimeIndex([series.index[0] - pd.offsets.BDay(1)] + list(series.index))
            ax_f.plot(dates_full, cum, color=color, ls=ls, lw=lw, label=f"{lbl} ({cum[-1] / 100.0 - 1.0:+.1%})")
    ax_f.axhline(100.0, color="black", lw=0.7, alpha=0.5)
    ax_f.set_title("Full-Sample Indexed Cumulative Performance (Base = 100)", fontsize=12.5, fontweight="bold", pad=10)
    ax_f.set_ylabel("Indexed Wealth (Base 100)", fontsize=10)
    ax_f.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    ax_f.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_f.grid(alpha=0.25, linestyle=":")
    ax_f.spines[["top", "right"]].set_visible(False)
    handles_f, _ = ax_f.get_legend_handles_labels()
    if handles_f:
        ax_f.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.2)
    fig_full.tight_layout()
    fig_full.savefig(output_dir / "full_sample.png", dpi=160, bbox_inches="tight")
    plt.close(fig_full)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, verify: bool = False, write_reference: bool = False, refresh: bool = False) -> None:
    """Run fixed-rule trend signals and portfolio accounting; verify before exporting."""
    root = Path(root).resolve()
    if verify and write_reference:
        raise ValueError("Verification and baseline writing are separate operations.")
    if verify and refresh:
        raise ValueError("Verification and live Yahoo refresh are separate operations.")
    settings_path = root / "config.json"
    config = json.loads(settings_path.read_text(encoding="utf-8"))
    ref_path = root / config["verification"]["reference"]
    if verify and not ref_path.exists():
        raise FileNotFoundError("Reference missing; --verify never creates its own baseline.")

    prices, log_returns = load_market_data(root, config["data"], use_cache=True, refresh=refresh)
    simple_returns = np.expm1(log_returns)

    _, composite = compute_ewma_tsmom_signals(log_returns, config["model"])
    daily_targets = compute_positions(composite, log_returns, config["portfolio"])
    monthly_targets = rolling_positions(log_returns, config["portfolio"], config["model"])

    valid_daily = daily_targets.shift(1).notna().all(axis=1)
    valid_monthly = monthly_targets.shift(1).notna().all(axis=1)
    common_valid = valid_daily & valid_monthly
    if not common_valid.any():
        raise ValueError("Insufficient history for a common evaluation window.")
    evaluation_start = common_valid[common_valid].index[0]
    if not common_valid.loc[evaluation_start:].all():
        raise ValueError("Missing signals inside the common evaluation window.")

    portfolio = config["portfolio"]
    metrics_records, daily_series, diagnostics = [], {}, {}

    for name, targets in [("Daily expanding", daily_targets), ("Monthly rolling", monthly_targets)]:
        for cost in config["cost_bps"]:
            sim = simulate_strategy(
                simple_returns, targets, cost_bps=cost,
                target_vol_portfolio=portfolio["target_vol_portfolio"],
                max_leverage=portfolio["max_leverage"],
                position_clip=portfolio["position_clip"],
                evaluation_start=evaluation_start,
            )
            gross_leverage = sim.positions.abs().sum(axis=1)
            perf = score_performance(sim.net_return)
            annual_turnover = float(sim.turnover.mean() * 252)

            metrics_records.append({
                "variant": name, "cost_bps": cost,
                "annual_turnover": annual_turnover,
                "max_gross_leverage": float(gross_leverage.max()),
                **perf,
            })

            if name == "Daily expanding" and cost == 0:
                daily_series["Daily expanding (Gross)"] = sim.net_return
            if cost == 5:
                daily_series[f"{name} (5 bps)"] = sim.net_return
                diagnostics[f"{name}:gross_leverage"] = gross_leverage
                diagnostics[f"{name}:turnover"] = sim.turnover
                diagnostics[f"{name}:cost_fraction"] = sim.costs

    eq_cols = [k for k, v in config["data"]["universe"].items() if v["class"] == "equity"]
    bd_cols = [k for k, v in config["data"]["universe"].items() if v["class"] == "bond"]
    eq_ret = simple_returns[eq_cols].mean(axis=1).loc[evaluation_start:]
    bd_ret = simple_returns[bd_cols].mean(axis=1).loc[evaluation_start:]
    daily_series["60/40 Portfolio"] = 0.6 * eq_ret + 0.4 * bd_ret
    daily_series["Equity basket"] = eq_ret

    metrics_df = pd.DataFrame(metrics_records)
    returns_df = pd.DataFrame(daily_series)

    # Verification compares metrics against regression baseline
    report = None
    if verify:
        expected = pd.read_csv(ref_path)
        report = verify_reference(metrics_df, expected, config["verification"]["absolute_tolerance"])
        for field in ("days", "annual_turnover", "max_gross_leverage", "total_return"):
            a = metrics_df.set_index(["variant", "cost_bps"])[field].sort_index()
            b = expected.set_index(["variant", "cost_bps"])[field].sort_index()
            if not np.allclose(a, b, atol=config["verification"]["absolute_tolerance"], rtol=0):
                raise AssertionError(f"Reference mismatch: {field}")

    crisis_periods = {
        "China / Oil Shock": ("2015-08-01", "2016-02-29"),
        "COVID-19 Crash": ("2020-02-03", "2020-03-31"),
        "Rate Shock": ("2022-01-03", "2022-12-31"),
    }
    crisis_rows = []
    for name, (start, end) in crisis_periods.items():
        subset = returns_df.loc[start:end]
        if subset.empty or returns_df.index[0] > pd.Timestamp(start) or returns_df.index[-1] < pd.Timestamp(end):
            continue
        crisis_rows.append({
            "period": name, "start": str(subset.index[0].date()), "end": str(subset.index[-1].date()),
            "days": len(subset),
            **{col: float((1.0 + subset[col]).prod() - 1) for col in subset.columns},
        })
    crisis_df = pd.DataFrame(crisis_rows, columns=["period", "start", "end", "days", *returns_df.columns] if crisis_rows else None)

    snapshot_dir = root / config["data"].get("snapshot_dir", "data/snapshot")
    cache_dir = root / config["data"].get("cache_dir", "data/cache")
    using_snapshot = (not refresh) and (snapshot_dir / "prices.csv").exists() and (snapshot_dir / "returns.csv").exists()
    active_input_dir = snapshot_dir if using_snapshot else cache_dir
    input_provenance = (
        "Tracked adjusted-price snapshot (data/snapshot); configured tickers"
        if using_snapshot
        else "Live Yahoo Finance adjusted-close refresh via yfinance (data/cache)"
    )
    metadata = {
        "accounting_version": 2,
        "platform": platform.platform(), "python_version": sys.version,
        "packages": {name: __import__(name).__version__ for name in ("numpy", "pandas", "scipy", "matplotlib", "yfinance")},
        "n_instruments": len(log_returns.columns),
        "source_tickers": {name: meta["ticker"] for name, meta in config["data"]["universe"].items() if "ticker" in meta},
        "history_start": str(log_returns.index[0].date()), "history_end": str(log_returns.index[-1].date()),
        "evaluation_start": str(returns_df.index[0].date()), "evaluation_end": str(returns_df.index[-1].date()),
        "evaluation_days": len(returns_df),
        "config_sha256": _file_hash(settings_path),
        "input_sha256": {name: _file_hash(active_input_dir / name) for name in ("prices.csv", "returns.csv") if (active_input_dir / name).exists()},
        "input_provenance": input_provenance,
        "code_sha256": {name: _file_hash(Path(__file__).with_name(name)) for name in ("data.py", "models.py", "portfolio.py", "pipeline.py")},
        "cash_interest": 0.0, "sharpe_reference_rate_annual": 0.0,
        "benchmark_cost_bps": 0, "strategy_cost_bps": config["cost_bps"],
        "leverage_cap_basis": "after-fee NAV at daily rebalance; may drift intraday",
        "signal_history": "Log returns; numeric spans unchanged; no parameter search",
        "reference_is": "numerical regression baseline, not independent financial validation",
    }
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(artifacts / "metrics.csv", index=False)
    returns_df.to_csv(artifacts / "daily_returns.csv", index_label="Date")
    pd.DataFrame(diagnostics).to_csv(artifacts / "execution_checks.csv", index_label="Date")
    if not crisis_df.empty:
        crisis_df.to_csv(artifacts / "crisis_returns.csv", index=False)
    else:
        pd.DataFrame(columns=["period", "start", "end", "days", *returns_df.columns]).to_csv(artifacts / "crisis_returns.csv", index=False)
    generate_figures(daily_series, crisis_df, artifacts)
    (artifacts / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if write_reference:
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(ref_path, index=False)
        print(f"[+] Wrote explicit numerical regression baseline: {ref_path}")
    if report is not None:
        report.to_csv(artifacts / "verification.csv", index=False)
        print(f"[+] Numerical regression verification passed: {ref_path}")
    elif (artifacts / "verification.csv").exists():
        (artifacts / "verification.csv").unlink()
    print(f"[+] Study artifacts exported: {artifacts}")


def main():
    parser = argparse.ArgumentParser(description="Macro CTA fixed-rule research runner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify", action="store_true", help="Compare against tracked snapshot baseline before export")
    group.add_argument("--write-reference", action="store_true", help="Explicitly replace the regression baseline after reviewing a model change")
    group.add_argument("--refresh", action="store_true", help="Download a fresh Yahoo Finance series into data/cache without modifying the tracked snapshot")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[2], verify=args.verify, write_reference=args.write_reference, refresh=args.refresh)


if __name__ == "__main__":
    main()
