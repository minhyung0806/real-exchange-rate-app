"""Real Exchange Rate Explorer — IMF SDMX 3.0, Python 3.12."""
from __future__ import annotations

from datetime import date
from io import StringIO
import calendar
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pycountry
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://api.imf.org/external/sdmx/3.0"
FIRST = pd.Period("1976-01", freq="M")
COUNTRIES = {country.name: country.alpha_3 for country in pycountry.countries}
COUNTRIES["Euro area"] = "G163"
COUNTRIES = dict(sorted(COUNTRIES.items()))
EXPORT_COLUMNS = [
    "Month", "Country 1", "Country 2", "CPI Country 1", "CPI Country 2",
    "Exchange Rate Country 1 (Currency 1 per USD)",
    "Exchange Rate Country 2 (Currency 2 per USD)",
    "Nominal S(t) (Currency 2 per Currency 1)", "RER(t)",
    "RER* (selected-period average)", "Real Index(t)", "Valid Observation",
]


class IMFError(Exception):
    """User-displayable IMF retrieval error."""


def http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=0.7, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"Accept": "csvfile, text/csv;q=0.9", "User-Agent": "RERExplorer/1.0"})
    return session


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_series(code: str, kind: str) -> pd.Series:
    """Download IMF monthly observations, indexed by calendar month.

    IMF SDMX 3.0: /data/dataflow/{agency}/{flow}/{version}/{key}
    """
    if kind not in ("CPI", "ER"):
        raise ValueError("Invalid data kind")
    flow = "CPI" if kind == "CPI" else "ER"
    key = f"{code}.CPI._T.IX.M" if kind == "CPI" else f"{code}.XDC_USD.PA_RT.M"
    url = f"{BASE}/data/dataflow/IMF.STA/{flow}/~/{key}"
    try:
        with http_session() as session:
            response = session.get(
                url,
                params={"startPeriod": "1976-01", "attributes": "none"},
                timeout=(8, 35),
            )
    except requests.RequestException as exc:
        raise IMFError(f"Could not contact IMF for {kind} ({code}): {exc}") from exc
    if response.status_code == 204:
        return pd.Series(dtype="float64", name=f"{code}_{kind}")
    if not response.ok:
        raise IMFError(f"IMF returned HTTP {response.status_code} for {kind} ({code}). URL: {response.url}")
    if not response.content.strip():
        return pd.Series(dtype="float64", name=f"{code}_{kind}")
    try:
        frame = pd.read_csv(StringIO(response.text), low_memory=False)
    except Exception as exc:
        raise IMFError(f"IMF returned unreadable CSV for {kind} ({code}).") from exc
    frame.columns = frame.columns.str.strip()
    if not {"TIME_PERIOD", "OBS_VALUE"}.issubset(frame.columns):
        raise IMFError(f"Unexpected IMF CSV columns for {kind} ({code}): {', '.join(frame.columns[:12])}")
    # Guard against a broad/mismatched response silently mixing multiple series.
    expected = {"COUNTRY": code, "FREQUENCY": "M"}
    for column, value in expected.items():
        if column in frame:
            frame = frame.loc[frame[column].astype(str) == value]
    frame = frame.loc[frame["TIME_PERIOD"].astype(str).str.fullmatch(r"\d{4}-(?:M)?(?:0[1-9]|1[0-2])", na=False)].copy()
    if frame.empty:
        return pd.Series(dtype="float64", name=f"{code}_{kind}")
    months = pd.PeriodIndex(frame["TIME_PERIOD"].astype(str).str.replace("-M", "-", regex=False), freq="M")
    values = pd.to_numeric(frame["OBS_VALUE"], errors="coerce").to_numpy(dtype=float)
    series = pd.Series(values, index=months, name=f"{code}_{kind}")
    series = series.replace([np.inf, -np.inf], np.nan)
    # Duplicate month entries indicate a response with multiple distinct series.
    if series.index.has_duplicates:
        raise IMFError(f"IMF returned multiple observations per month for {kind} ({code}); cannot select one safely.")
    return series.sort_index().loc[lambda s: s.index >= FIRST]


def calculate(country1: str, country2: str, start: pd.Period, end: pd.Period,
              cpi1: pd.Series, cpi2: pd.Series, er1: pd.Series, er2: pd.Series) -> tuple[pd.DataFrame, float]:
    months = pd.period_range(start, end, freq="M")
    df = pd.DataFrame(index=months)
    df["Month"] = months.astype(str)
    df["Country 1"] = country1
    df["Country 2"] = country2
    c1, c2, e1, e2 = EXPORT_COLUMNS[3:7]
    df[c1] = cpi1.reindex(months)
    df[c2] = cpi2.reindex(months)
    df[e1] = er1.reindex(months)
    df[e2] = er2.reindex(months)
    positive = (df[[c1, c2, e1, e2]] > 0).all(axis=1)
    finite = np.isfinite(df[[c1, c2, e1, e2]].to_numpy(dtype=float)).all(axis=1)
    valid = positive & finite
    nominal_col = EXPORT_COLUMNS[7]
    df[nominal_col] = np.where(
        (df[e1] > 0) & (df[e2] > 0) & np.isfinite(df[[e1, e2]]).all(axis=1),
        df[e2] / df[e1], np.nan,
    )
    df["RER(t)"] = np.where(valid, df[nominal_col] * df[c1] / df[c2], np.nan)
    df["RER(t)"] = df["RER(t)"].replace([np.inf, -np.inf], np.nan)
    valid = valid & df["RER(t)"].notna() & (df["RER(t)"] > 0)
    df.loc[~valid, "RER(t)"] = np.nan
    benchmark = float(df.loc[valid, "RER(t)"].mean()) if valid.any() else float("nan")
    df[EXPORT_COLUMNS[9]] = benchmark
    df["Real Index(t)"] = 100 * df["RER(t)"] / benchmark if np.isfinite(benchmark) and benchmark > 0 else np.nan
    df["Valid Observation"] = valid
    return df[EXPORT_COLUMNS].reset_index(drop=True), benchmark


def chart(df: pd.DataFrame, mode: str, c1: str, c2: str) -> go.Figure:
    both = mode == "Both"
    fig = make_subplots(specs=[[{"secondary_y": True}]]) if both else go.Figure()
    x = pd.to_datetime(df["Month"])
    real = go.Scatter(x=x, y=df["Real Index(t)"], mode="lines", name="Real Index", connectgaps=False,
                      line=dict(color="#2563eb", width=2.5), hovertemplate="%{x|%b %Y}<br>Real Index: %{y:.2f}<extra></extra>")
    nominal = go.Scatter(x=x, y=df[EXPORT_COLUMNS[7]], mode="lines", name="Nominal S(t)", connectgaps=False,
                         line=dict(color="#f59e0b", width=2), hovertemplate="%{x|%b %Y}<br>S(t): %{y:,.5f}<extra></extra>")
    if mode in ("Real", "Both"):
        fig.add_trace(real, secondary_y=False) if both else fig.add_trace(real)
        fig.add_hline(y=100, line_dash="dash", line_color="#64748b", annotation_text="Sample average = 100")
    if mode in ("Nominal", "Both"):
        fig.add_trace(nominal, secondary_y=True) if both else fig.add_trace(nominal)
    real_label = "Real Exchange Rate Index (Sample Average = 100)"
    nominal_label = f"{c2} currency per {c1} currency (S(t))"
    if both:
        fig.update_yaxes(title_text=real_label, secondary_y=False)
        fig.update_yaxes(title_text=nominal_label, secondary_y=True)
    else:
        fig.update_yaxes(title_text=real_label if mode == "Real" else nominal_label)
    fig.update_layout(template="plotly_white", height=510, hovermode="x unified", margin=dict(l=15, r=20, t=35, b=20),
                      legend=dict(orientation="h", y=1.12), xaxis_title="Month")
    return fig


def main() -> None:
    st.set_page_config(page_title="Real Exchange Rate Explorer", page_icon="🌍", layout="wide")
    st.title("Real Exchange Rate Explorer")
    st.caption("Compare two economies' CPI-adjusted exchange rates using monthly IMF data.")
    names = list(COUNTRIES)
    today = pd.Period(date.today(), freq="M")
    with st.sidebar:
        st.header("Settings")
        first = st.selectbox("Country 1", names, index=names.index("United States"))
        second = st.selectbox("Country 2", names, index=names.index("Euro area"))
        start_date = st.date_input("Start month", date(2000, 1, 1), min_value=date(1976, 1, 1), max_value=date.today())
        end_date = st.date_input("End month", date.today(), min_value=date(1976, 1, 1), max_value=date.today())
        mode = st.radio("Chart mode", ["Real", "Nominal", "Both"], horizontal=True)
        st.caption("Only the year and month of each date are used.")
    start, end = pd.Period(start_date, freq="M"), pd.Period(end_date, freq="M")
    if start > end:
        st.error("Start month must not be later than end month.")
        st.stop()
    code1, code2 = COUNTRIES[first], COUNTRIES[second]
    with st.spinner("Loading IMF CPI and exchange-rate observations..."):
        try:
            data = {(code, kind): get_series(code, kind) for code in set([code1, code2]) for kind in ("CPI", "ER")}
        except IMFError as exc:
            st.error(str(exc))
            st.info("Try another country pair or refresh later. No synthetic data are substituted.")
            st.stop()
    for name, code in ((first, code1), (second, code2)):
        for kind in ("CPI", "ER"):
            if data[(code, kind)].dropna().empty:
                st.warning(f"No monthly {kind} observations returned for {name} ({code}).")
    df, benchmark = calculate(first, second, start, end, data[(code1, "CPI")], data[(code2, "CPI")],
                              data[(code1, "ER")], data[(code2, "ER")])
    n = int(df["Valid Observation"].sum())
    last = df.loc[df["Valid Observation"]].tail(1)
    latest = float(last["Real Index(t)"].iloc[0]) if not last.empty else float("nan")
    a, b, c = st.columns(3)
    a.metric("RER* (sample average)", f"{benchmark:,.5f}" if np.isfinite(benchmark) else "N/A")
    b.metric("Latest valid Real Index", f"{latest:,.2f}" if np.isfinite(latest) else "N/A",
             help="Latest month with all four required IMF observations, not necessarily the end month.")
    c.metric("Valid months / total", f"{n} / {len(df)}")
    if n == 0:
        st.error("Insufficient data: no months have both CPIs and both exchange rates. The real index cannot be calculated.")
    elif n < 12:
        st.warning("Fewer than 12 valid months: the historical benchmark may be unstable.")
    if n < len(df):
        st.info(f"{len(df)-n} month(s) lack valid CPI and/or exchange-rate data. Missing values are not filled.")
    if first == second:
        st.info("Identical countries: nominal S(t) = 1 and Real Index = 100 in valid months.")
    st.plotly_chart(chart(df, mode, first, second), use_container_width=True)
    st.download_button("Download monthly CSV", data=df.to_csv(index=False, na_rep=""),
                       file_name=f"rer_{code1}_{code2}_{start}_{end}.csv", mime="text/csv")
    with st.expander("Methodology and interpretation", expanded=True):
        st.markdown(r"""
**Nominal direction:** IMF reports E₁ and E₂ in local currency per US dollar.
Therefore **S(t) = E₂(t) / E₁(t)** is Currency 2 per one unit of Currency 1.

**Real exchange rate:** RER(t) = S(t) × CPI₁(t) / CPI₂(t).

**Historical benchmark:** RER* is the arithmetic mean of valid RER observations **within your selected months**.
**Real Index(t) = 100 × RER(t) / RER***.

Above 100 indicates relative overvaluation of Currency 1 against its selected-period historical average;
below 100 indicates relative undervaluation. This average is **not necessarily economic equilibrium**.
Different CPI base years can change the RER level, though a consistent rescaling cancels in the indexed series.
The nominal exchange rate is never normalized. All missing values remain missing.
""")
    st.caption("Source: IMF SDMX 3.0, IMF.STA CPI and ER dataflows. Coverage and latest months vary by series. "
               "The Euro area uses G163. Country names are mapped to ISO3 codes; some territories may lack IMF data.")


if __name__ == "__main__":
    main()
