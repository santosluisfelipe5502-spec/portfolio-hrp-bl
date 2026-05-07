import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date
import io, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="HRP + Black-Litterman | Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilo ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#f8f7f4}
[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid #e8e6e0}
[data-testid="stSidebar"] *{color:#1a1a18 !important}
[data-testid="stSidebar"] .stMarkdown p{color:#444441 !important}
[data-testid="stSidebar"] label{color:#1a1a18 !important;font-size:13px !important}
[data-testid="stSidebar"] small{color:#888780 !important}
[data-testid="stSidebar"] h3{color:#1a1a18 !important;font-size:16px !important}
[data-testid="stSidebar"] hr{border-color:#e8e6e0 !important}
.stCheckbox label{color:#1a1a18 !important;font-size:13px !important}
.stCheckbox span{color:#1a1a18 !important}
.stTabs [data-baseweb="tab"]{color:#444441 !important}
.stTabs [aria-selected="true"]{color:#1a1a18 !important;font-weight:500 !important}
.metric-card{background:#fff;border:1px solid #e8e6e0;border-radius:10px;padding:16px 20px;margin-bottom:8px}
.metric-label{font-size:11px;color:#888780;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.metric-value{font-size:26px;font-weight:500;color:#1a1a18;line-height:1.1}
.metric-sub{font-size:12px;color:#b4b2a9;margin-top:4px}
.metric-value.pos{color:#0F6E56}
.metric-value.neg{color:#A32D2D}
.metric-value.warn{color:#854F0B}
.section-title{font-size:11px;font-weight:500;letter-spacing:.08em;text-transform:uppercase;color:#888780;margin:1.5rem 0 .5rem}
.badge{display:inline-block;font-size:11px;padding:2px 10px;border-radius:12px;font-weight:500}
.badge-green{background:#EAF3DE;color:#27500A}
.badge-blue{background:#E6F1FB;color:#185FA5}
.badge-amber{background:#FAEEDA;color:#633806}
.badge-red{background:#FCEBEB;color:#A32D2D}
div[data-testid="stHorizontalBlock"]{gap:12px}
</style>
""", unsafe_allow_html=True)

# ── Constantes ───────────────────────────────────────────────────────────────
ASSET_CFG = [
    {"name": "IDA Pré",    "key": "IDKAPRE5A", "color": "#E24B4A", "cluster": "Renda fixa", "w": 0.268, "vol": 0.05},
    {"name": "IMA",        "key": "IMA",        "color": "#1D9E75", "cluster": "Renda fixa", "w": 0.188, "vol": 0.07},
    {"name": "IHFA",       "key": "IHFA",       "color": "#378ADD", "cluster": "Âncora",     "w": 0.172, "vol": 0.06},
    {"name": "IDA-Geral",  "key": "IDAGERAL",   "color": "#888780", "cluster": "Âncora",     "w": 0.144, "vol": 0.03},
    {"name": "Ibovespa",   "key": "IBOV",       "color": "#BA7517", "cluster": "Equity",     "w": 0.145, "vol": 0.24},
    {"name": "Internac.",  "key": "INTL",       "color": "#7F77DD", "cluster": "Equity",     "w": 0.083, "vol": 0.184},
]
WEIGHTS = {a["name"]: a["w"] for a in ASSET_CFG}
COLORS  = {a["name"]: a["color"] for a in ASSET_CFG}

# ── Funções utilitárias ──────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_cdi():
    """Busca CDI mensal via API do Banco Central (SGS 4391)."""
    try:
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados?formato=json"
        r = requests.get(url, timeout=15)
        df = pd.DataFrame(r.json())
        df["data"]  = pd.to_datetime(df["data"], dayfirst=True) + pd.offsets.MonthEnd(0)
        df["valor"] = df["valor"].astype(float) / 100
        return df.set_index("data").sort_index()
    except Exception as e:
        st.warning(f"Não foi possível buscar o CDI automaticamente: {e}")
        return None

def read_uploaded(file):
    """Lê XLS/XLSX/CSV em DataFrame com duas colunas: data, valor."""
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file, sep=None, engine="python")
        else:
            df = pd.read_excel(file, engine="openpyxl")
        # Tentar detectar coluna de data e valor
        df.columns = [str(c).strip() for c in df.columns]
        date_col = next((c for c in df.columns if "data" in c.lower() or "date" in c.lower()), df.columns[0])
        val_col  = next((c for c in df.columns if c != date_col), df.columns[-1])
        # Tentar col[1] e col[2] (formato ANBIMA: código, data, valor)
        if df.shape[1] >= 3:
            date_col = df.columns[1]
            val_col  = df.columns[2]
        df["_data"]  = pd.to_datetime(df[date_col].astype(str).str.split(" ").str[0], dayfirst=True, errors="coerce")
        df["_valor"] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=["_data","_valor"])[["_data","_valor"]]
        df.columns = ["data","valor"]
        return df.set_index("data").sort_index()
    except Exception as e:
        st.error(f"Erro ao ler {file.name}: {e}")
        return None

def to_monthly(df):
    return df.resample("ME").last()

def calc_intl(spy_df, tlt_df, w_spy=0.40, w_tlt=0.60):
    df = pd.concat([spy_df.rename(columns={"valor":"SPY"}),
                    tlt_df.rename(columns={"valor":"TLT"})], axis=1).dropna()
    ret = w_spy * df["SPY"].pct_change() + w_tlt * df["TLT"].pct_change()
    idx = (1 + ret).cumprod() * 100
    idx.iloc[0] = 100
    return idx.rename("valor").to_frame()

def align_and_compute(series_dict, cdi_df, start="2009-01-31", end=None):
    if end is None:
        end = pd.Timestamp.today() + pd.offsets.MonthEnd(0)
    end = pd.Timestamp(end)
    rets = {}
    for k, s in series_dict.items():
        s = s[s.index >= start]
        s = s[s.index <= end]
        rets[k] = s["valor"].pct_change().dropna()

    common = rets[list(rets.keys())[0]].index
    for r in rets.values():
        common = common.intersection(r.index)

    port_ret = sum(WEIGHTS[k] * rets[k][common] for k in WEIGHTS if k in rets)
    cdi_aligned = cdi_df["valor"].reindex(common).ffill()

    return port_ret, cdi_aligned, common

def metrics(r, rf_s):
    ann_ret = (1 + r.mean()) ** 12 - 1
    ann_vol = r.std() * np.sqrt(12)
    rf_ann  = (1 + rf_s.mean()) ** 12 - 1
    excess  = r.values - rf_s.values
    sharpe  = (excess.mean() / r.std()) * np.sqrt(12)
    neg     = r[r < rf_s.values]
    downside = neg.std() * np.sqrt(12) if len(neg) > 1 else np.nan
    sortino = (ann_ret - rf_ann) / downside if downside and not np.isnan(downside) else np.nan
    cum     = (1 + r).cumprod()
    dd      = (cum - cum.cummax()) / cum.cummax()
    max_dd  = dd.min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    var95   = r.quantile(0.05)
    return dict(
        ann_ret=ann_ret, ann_vol=ann_vol, rf_ann=rf_ann,
        sharpe=sharpe, sortino=sortino, max_dd=max_dd,
        calmar=calmar, var95=var95, cum=cum, dd=dd,
    )

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 HRP + Black-Litterman")
    st.markdown('<span class="badge badge-blue">Fase 5 — Dashboard</span>', unsafe_allow_html=True)
    st.divider()

    st.markdown("#### CDI (taxa livre de risco)")
    use_auto_cdi = st.toggle("Buscar CDI automaticamente (BCB)", value=True)
    cdi_file = None
    if not use_auto_cdi:
        cdi_file = st.file_uploader("Upload CDI (CSV/Excel)", type=["csv","xls","xlsx"], key="cdi")

    st.divider()
    st.markdown("#### Índices de mercado")
    st.caption("Faça upload dos arquivos ANBIMA/B3. Deixe em branco para usar dados simulados.")

    uploads = {}
    for cfg in ASSET_CFG:
        if cfg["key"] == "INTL":
            continue
        uploads[cfg["name"]] = st.file_uploader(
            cfg["name"], type=["csv","xls","xlsx"], key=cfg["key"]
        )

    st.markdown("**Internacional** (40% SPY + 60% TLT)")
    spy_file = st.file_uploader("SPY",     type=["csv","xls","xlsx"], key="SPY")
    tlt_file = st.file_uploader("TLT",     type=["csv","xls","xlsx"], key="TLT")

    st.divider()
    st.markdown("#### Período de análise")
    start_date = st.date_input("Início", value=date(2009,1,1))
    end_date   = st.date_input("Fim",    value=date.today())

    st.divider()
    st.markdown("#### Parâmetros HRP + BL")
    spy_w = st.slider("Peso SPY na carteira internacional", 0.0, 1.0, 0.40, 0.05)
    tlt_w = 1.0 - spy_w
    st.caption(f"TLT = {tlt_w:.0%}")
    banda = st.slider("Banda de tolerância (%)", 1.0, 10.0, 3.0, 0.5)
    patrimonio = st.number_input("Patrimônio (R$ mil)", min_value=100, value=10000, step=500)

# ── Carga de dados ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_demo_series():
    """Gera séries demo quando não há upload."""
    def gen(ret, vol, n, s):
        rng = np.random.default_rng(s)
        daily = ret/12; dvol = vol/np.sqrt(12)
        v, arr = 100.0, [100.0]
        for _ in range(n-1):
            v *= (1 + daily + dvol * rng.standard_normal())
            arr.append(round(v, 4))
        idx = pd.date_range("2009-01-31", periods=n, freq="ME")
        return pd.DataFrame({"valor": arr}, index=idx)
    params = [(0.085,0.06,208,1),(0.095,0.07,208,2),(0.085,0.06,208,3),
              (0.132,0.03,208,4),(0.092,0.238,208,5),(0.118,0.184,208,6)]
    keys = ["IDA Pré","IMA","IHFA","IDA-Geral","Ibovespa","Internac."]
    return {k: gen(*p) for k,p in zip(keys, params)}

with st.spinner("Carregando dados…"):
    # CDI
    if use_auto_cdi:
        cdi_raw = fetch_cdi()
        if cdi_raw is None and cdi_file:
            cdi_raw = read_uploaded(cdi_file)
    elif cdi_file:
        cdi_raw = read_uploaded(cdi_file)
    else:
        cdi_raw = None

    if cdi_raw is None:
        # CDI simulado
        idx = pd.date_range("2009-01-31", periods=208, freq="ME")
        cdi_vals = np.where(idx.year < 2017, 0.011,
                   np.where(idx.year < 2020, 0.005,
                   np.where(idx.year < 2022, 0.003, 0.011)))
        cdi_raw = pd.DataFrame({"valor": cdi_vals}, index=idx)
        st.sidebar.warning("CDI simulado (sem dados reais)")

    # Séries de ativos
    series = {}
    demo = load_demo_series()
    has_real = False

    for cfg in ASSET_CFG:
        if cfg["key"] == "INTL":
            continue
        f = uploads.get(cfg["name"])
        if f:
            raw = read_uploaded(f)
            if raw is not None:
                series[cfg["name"]] = to_monthly(raw)
                has_real = True
                continue
        series[cfg["name"]] = demo[cfg["name"]]

    # Internacional
    if spy_file and tlt_file:
        spy_df = read_uploaded(spy_file)
        tlt_df = read_uploaded(tlt_file)
        if spy_df is not None and tlt_df is not None:
            spy_m = to_monthly(spy_df)
            tlt_m = to_monthly(tlt_df)
            series["Internac."] = calc_intl(spy_m, tlt_m, spy_w, tlt_w)
            has_real = True
    else:
        series["Internac."] = demo["Internac."]

    # Computar portfólio
    start_ts = pd.Timestamp(start_date) + pd.offsets.MonthEnd(0)
    end_ts   = pd.Timestamp(end_date)   + pd.offsets.MonthEnd(0)
    port_ret, cdi_aligned, common_idx = align_and_compute(series, cdi_raw, str(start_ts)[:10], str(end_ts)[:10])
    ibov_ret = series["Ibovespa"]["valor"].pct_change().dropna().reindex(common_idx).ffill()

    m_port = metrics(port_ret, cdi_aligned)
    m_ibov = metrics(ibov_ret, cdi_aligned)

    port_cum = m_port["cum"] * 100
    ibov_cum = (1 + ibov_ret).cumprod() * 100
    cdi_cum  = (1 + cdi_aligned).cumprod() * 100

    rf_ann = m_port["rf_ann"]

# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    data_src = "📡 CDI via BCB" if (use_auto_cdi and fetch_cdi() is not None) else "📂 CDI local"
    real_tag = "📂 dados reais" if has_real else "🔬 dados simulados"
    st.markdown(f"""
    <h2 style='margin:0;font-size:24px;font-weight:500;color:#1a1a18'>
        HRP + Black-Litterman
    </h2>
    <p style='margin:4px 0 0;font-size:13px;color:#888780'>
        {common_idx[0].strftime('%b/%Y')} → {common_idx[-1].strftime('%b/%Y')} &nbsp;·&nbsp;
        {len(common_idx)} meses &nbsp;·&nbsp; {data_src} &nbsp;·&nbsp; {real_tag}
    </p>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style='text-align:right;padding-top:4px'>
        <span class='badge badge-blue'>rf = CDI médio {rf_ann*100:.2f}% a.a.</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
acum_port = (port_cum.iloc[-1] / 100 - 1) * 100
acum_cdi  = (cdi_cum.iloc[-1]  / 100 - 1) * 100
acum_ibov = (ibov_cum.iloc[-1] / 100 - 1) * 100

def kpi(label, value, sub="", cls=""):
    return f"""<div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value {cls}'>{value}</div>
        <div class='metric-sub'>{sub}</div>
    </div>"""

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.markdown(kpi("Acumulado HRP+BL", f"+{acum_port:.1f}%", f"CDI +{acum_cdi:.1f}%", "pos"), unsafe_allow_html=True)
k2.markdown(kpi("Retorno a.a.", f"{m_port['ann_ret']*100:.2f}%", f"IBOV {m_port['ann_ret']*100 - m_ibov['ann_ret']*100:+.1f}%"), unsafe_allow_html=True)
cls_sharpe = "pos" if m_port["sharpe"] > 0.2 else "warn" if m_port["sharpe"] > 0 else "neg"
k3.markdown(kpi("Sharpe (rf=CDI)", f"{m_port['sharpe']:.3f}", f"IBOV {m_ibov['sharpe']:.3f}", cls_sharpe), unsafe_allow_html=True)
cls_sort = "pos" if m_port["sortino"] and m_port["sortino"] > 0.3 else "warn"
k4.markdown(kpi("Sortino", f"{m_port['sortino']:.3f}" if m_port["sortino"] else "—", "penaliza só quedas", cls_sort), unsafe_allow_html=True)
k5.markdown(kpi("Max Drawdown", f"{m_port['max_dd']*100:.2f}%", f"IBOV {m_ibov['max_dd']*100:.1f}%", "warn"), unsafe_allow_html=True)
k6.markdown(kpi("Calmar ratio", f"{m_port['calmar']:.3f}", f"IBOV {m_ibov['calmar']:.3f}", "pos"), unsafe_allow_html=True)

st.divider()

# ── Tabs principais ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Retorno acumulado",
    "📉 Drawdown",
    "📊 Métricas",
    "🔄 Rebalanceamento",
    "🌐 Cenários",
])

# ── Tab 1: Retorno acumulado ──────────────────────────────────────────────────
with tab1:
    show_ibov = st.checkbox("Mostrar Ibovespa", value=True)
    show_cdi  = st.checkbox("Mostrar CDI",      value=True)
    show_igual= st.checkbox("Mostrar 1/N igual",value=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values.round(2),
        name="HRP+BL", line=dict(color="#378ADD", width=2.5)))
    if show_cdi:
        fig.add_trace(go.Scatter(x=cdi_cum.index, y=cdi_cum.values.round(2),
            name="CDI", line=dict(color="#1D9E75", width=1.5, dash="dot")))
    if show_ibov:
        fig.add_trace(go.Scatter(x=ibov_cum.index, y=ibov_cum.values.round(2),
            name="Ibovespa", line=dict(color="#BA7517", width=1.5, dash="dash")))
    if show_igual:
        eq_ret = sum((1/6) * series[a["name"]]["valor"].pct_change().dropna().reindex(common_idx).ffill()
                     for a in ASSET_CFG)
        eq_cum = (1 + eq_ret).cumprod() * 100
        fig.add_trace(go.Scatter(x=eq_cum.index, y=eq_cum.values.round(2),
            name="1/N igual", line=dict(color="#7F77DD", width=1.2, dash="longdash")))

    fig.update_layout(
        height=340, margin=dict(l=0,r=0,t=8,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        yaxis=dict(tickprefix="R$", gridcolor="#e8e6e0"),
        xaxis=dict(gridcolor="#e8e6e0"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Retorno anual
    st.markdown("<div class='section-title'>Retorno anual</div>", unsafe_allow_html=True)
    ann_port = port_cum.resample("YE").last().pct_change().dropna() * 100
    ann_cdi  = cdi_cum.resample("YE").last().pct_change().dropna() * 100
    ann_ibov = ibov_cum.resample("YE").last().pct_change().dropna() * 100
    years = sorted(set(ann_port.index.year) & set(ann_cdi.index.year) & set(ann_ibov.index.year))

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=[str(y) for y in years],
        y=[ann_port[ann_port.index.year==y].values[0] for y in years if len(ann_port[ann_port.index.year==y])],
        name="HRP+BL", marker_color="#378ADD"))
    fig2.add_trace(go.Bar(x=[str(y) for y in years],
        y=[ann_cdi[ann_cdi.index.year==y].values[0] for y in years if len(ann_cdi[ann_cdi.index.year==y])],
        name="CDI", marker_color="rgba(29,158,117,0.4)"))
    fig2.add_trace(go.Bar(x=[str(y) for y in years],
        y=[ann_ibov[ann_ibov.index.year==y].values[0] for y in years if len(ann_ibov[ann_ibov.index.year==y])],
        name="Ibovespa", marker_color="rgba(186,117,23,0.3)"))
    fig2.update_layout(
        height=280, barmode="group", margin=dict(l=0,r=0,t=8,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0"),
        xaxis=dict(gridcolor="#e8e6e0"),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2: Drawdown ───────────────────────────────────────────────────────────
with tab2:
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=m_port["dd"].index, y=(m_port["dd"]*100).round(2),
        name="HRP+BL", fill="tozeroy",
        line=dict(color="#378ADD", width=1.5),
        fillcolor="rgba(55,138,221,0.15)"))
    fig_dd.add_trace(go.Scatter(
        x=m_ibov["dd"].index, y=(m_ibov["dd"]*100).round(2),
        name="Ibovespa", fill="tozeroy",
        line=dict(color="#E24B4A", width=1.2, dash="dot"),
        fillcolor="rgba(226,75,74,0.08)"))
    fig_dd.update_layout(
        height=320, margin=dict(l=0,r=0,t=8,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0"),
        xaxis=dict(gridcolor="#e8e6e0"),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    dd_max_port = m_port["dd"].min() * 100
    dd_max_ibov = m_ibov["dd"].min() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Max DD — HRP+BL",  f"{dd_max_port:.2f}%")
    c2.metric("Max DD — Ibovespa",f"{dd_max_ibov:.2f}%",  delta=f"{dd_max_port-dd_max_ibov:.1f}%")
    c3.metric("Redução de risco",  f"{(dd_max_port-dd_max_ibov):.1f}%", delta="proteção HRP+BL")

# ── Tab 3: Métricas ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-title'>Comparativo de métricas</div>", unsafe_allow_html=True)

    rows = {
        "Retorno a.a.":      (f"{m_port['ann_ret']*100:.2f}%", f"{rf_ann*100:.2f}%",        f"{m_ibov['ann_ret']*100:.2f}%"),
        "Volatilidade a.a.": (f"{m_port['ann_vol']*100:.2f}%", "~0%",                        f"{m_ibov['ann_vol']*100:.2f}%"),
        "Sharpe (rf=CDI)":   (f"{m_port['sharpe']:.3f}",       "—",                          f"{m_ibov['sharpe']:.3f}"),
        "Sortino":           (f"{m_port['sortino']:.3f}" if m_port["sortino"] else "—", "—", f"{m_ibov['sortino']:.3f}" if m_ibov["sortino"] else "—"),
        "Calmar ratio":      (f"{m_port['calmar']:.3f}",        "—",                          f"{m_ibov['calmar']:.3f}"),
        "Max drawdown":      (f"{m_port['max_dd']*100:.2f}%",   "0%",                         f"{m_ibov['max_dd']*100:.2f}%"),
        "VaR 95% (mensal)":  (f"{m_port['var95']*100:.2f}%",    "positivo",                   f"{m_ibov['var95']*100:.2f}%"),
        "Acumulado":         (f"+{acum_port:.1f}%",             f"+{acum_cdi:.1f}%",           f"+{acum_ibov:.1f}%"),
    }
    df_metrics = pd.DataFrame(rows, index=["HRP+BL","CDI","Ibovespa"]).T
    df_metrics.index.name = "Métrica"
    st.dataframe(df_metrics, use_container_width=True)

    st.divider()
    st.markdown("<div class='section-title'>Alocação por ativo</div>", unsafe_allow_html=True)
    alloc_df = pd.DataFrame([{
        "Ativo": a["name"], "Cluster": a["cluster"],
        "Peso HRP+BL": f"{a['w']*100:.1f}%",
        "Vol. hist.": f"{a['vol']*100:.1f}%",
        "Contrib. risco": f"{a['w']*a['vol'] / sum(x['w']*x['vol'] for x in ASSET_CFG)*100:.1f}%",
    } for a in ASSET_CFG])
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

# ── Tab 4: Rebalanceamento ────────────────────────────────────────────────────
with tab4:
    st.markdown("Ajuste os pesos atuais do portfólio e veja o drift em relação ao target HRP+BL.")

    atual = {}
    col_s1, col_s2 = st.columns(2)
    for i, cfg in enumerate(ASSET_CFG):
        col = col_s1 if i < 3 else col_s2
        with col:
            atual[cfg["name"]] = st.slider(
                cfg["name"],
                min_value=0.0, max_value=60.0,
                value=round(cfg["w"]*100, 1), step=0.5,
                key=f"rebal_{cfg['name']}"
            )

    total_atual = sum(atual.values())
    delta_cor = "normal" if abs(total_atual - 100) < 0.5 else "inverse"
    st.metric("Total alocado", f"{total_atual:.1f}%", delta=f"{total_atual-100:.1f}% vs 100%")
    if abs(total_atual - 100) > 0.5:
        st.warning("⚠️ Total diferente de 100%. Ajuste os pesos antes de analisar.")

    st.divider()
    st.markdown("<div class='section-title'>Drift vs target</div>", unsafe_allow_html=True)

    drift_data = []
    for cfg in ASSET_CFG:
        d = atual[cfg["name"]] - cfg["w"] * 100
        drift_data.append({"Ativo": cfg["name"], "Target": cfg["w"]*100,
                           "Atual": atual[cfg["name"]], "Drift": d,
                           "Fora da banda": abs(d) > banda, "color": cfg["color"]})

    fig_drift = go.Figure()
    for row in drift_data:
        bar_color = "#E24B4A" if row["Drift"] < -banda else "#BA7517" if row["Drift"] > banda else "#378ADD"
        fig_drift.add_trace(go.Bar(
            x=[row["Ativo"]], y=[row["Drift"]],
            name=row["Ativo"], marker_color=bar_color,
            showlegend=False,
        ))
    fig_drift.add_hline(y=banda,  line_dash="dot", line_color="#888780", annotation_text=f"+{banda:.1f}% banda")
    fig_drift.add_hline(y=-banda, line_dash="dot", line_color="#888780", annotation_text=f"-{banda:.1f}% banda")
    fig_drift.update_layout(
        height=260, margin=dict(l=0,r=0,t=8,b=0),
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0", zeroline=True, zerolinecolor="#B4B2A9"),
    )
    st.plotly_chart(fig_drift, use_container_width=True)

    fora = [d for d in drift_data if d["Fora da banda"]]
    if fora:
        st.error(f"🔔 {len(fora)} ativo(s) fora da banda de ±{banda:.1f}%")
        pat_reais = patrimonio * 1000
        custo_op = 0.001  # 0.1% por padrão

        st.markdown("<div class='section-title'>Trades sugeridos</div>", unsafe_allow_html=True)
        trade_rows = []
        for d in fora:
            vol_fin = abs(d["Drift"] / 100) * pat_reais
            custo   = vol_fin * custo_op
            trade_rows.append({
                "Ativo": d["Ativo"],
                "Ação": "Reduzir" if d["Drift"] > 0 else "Aumentar",
                "Drift": f"{d['Drift']:+.1f}%",
                "Volume (R$)": f"R$ {vol_fin:,.0f}",
                "Custo est. (R$)": f"R$ {custo:,.0f}",
            })
        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
        custo_total = sum(abs(d["Drift"]/100)*pat_reais*custo_op for d in fora)
        st.info(f"💡 Custo total estimado de rebalanceamento: **R$ {custo_total:,.0f}** (0.10% sobre volume negociado)")
    else:
        st.success(f"✅ Portfólio dentro da banda de tolerância ±{banda:.1f}%. Nenhum rebalanceamento necessário.")

# ── Tab 5: Cenários ───────────────────────────────────────────────────────────
with tab5:
    st.markdown("Simule o impacto de cenários macroeconômicos nos retornos esperados do portfólio.")

    preset = st.selectbox("Cenário pré-definido", [
        "Base", "Juro alto", "Recessão", "Rali de risco", "Customizado"
    ])
    presets = {
        "Base":           dict(selic=13.75, ipca=4.5, pib=2.0, fx=5.1),
        "Juro alto":      dict(selic=16.5,  ipca=6.5, pib=0.5, fx=5.6),
        "Recessão":       dict(selic=10.0,  ipca=3.0, pib=-2.0,fx=6.2),
        "Rali de risco":  dict(selic=11.0,  ipca=4.0, pib=3.5, fx=4.8),
        "Customizado":    dict(selic=13.75, ipca=4.5, pib=2.0, fx=5.1),
    }
    p = presets[preset]

    c1, c2, c3, c4 = st.columns(4)
    selic = c1.slider("Selic (% a.a.)", 5.0, 20.0, p["selic"], 0.25)
    ipca  = c2.slider("IPCA (% a.a.)",  2.0, 15.0, p["ipca"],  0.25)
    pib   = c3.slider("PIB (% a.a.)",  -5.0,  8.0, p["pib"],   0.25)
    fx    = c4.slider("USD/BRL",         4.0,  8.0, p["fx"],    0.1)

    asset_rets = {
        "IDA Pré":   max(0, selic * 0.85 - max(0, (selic-12)*0.6)),
        "IMA":       selic*0.6 + ipca*0.4 + (1.5 if selic > 13 else 0),
        "IHFA":      selic*0.7 + pib*0.3 + 2.0,
        "IDA-Geral": selic * 0.95,
        "Ibovespa":  8 + pib*2.5 - max(0,(selic-12)*0.8) + (-2 if fx > 5.5 else 1),
        "Internac.": 10 + (3 if fx > 5.5 else -2) + (1 if pib > 2 else -1),
    }
    port_ret_sc = sum(WEIGHTS[k] * v for k, v in asset_rets.items())
    premium_sc  = port_ret_sc - selic
    sharpe_sc   = premium_sc / (m_port["ann_vol"] * 100)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Retorno esperado HRP+BL", f"{port_ret_sc:.2f}%", delta=f"{premium_sc:+.2f}% vs Selic")
    m2.metric("Prêmio sobre CDI",        f"{premium_sc:+.2f}%",  delta="positivo" if premium_sc > 0 else "negativo")
    m3.metric("Vol. estimada",           f"{m_port['ann_vol']*100:.1f}%")
    m4.metric("Sharpe projetado",        f"{sharpe_sc:.3f}")

    st.markdown("<div class='section-title'>Retorno esperado por ativo neste cenário</div>", unsafe_allow_html=True)
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Bar(
        x=list(asset_rets.keys()),
        y=[round(v,2) for v in asset_rets.values()],
        marker_color=[COLORS[k] for k in asset_rets],
        showlegend=False,
    ))
    fig_sc.add_hline(y=selic, line_dash="dot", line_color="#1D9E75",
                     annotation_text=f"Selic {selic:.2f}%", annotation_font_color="#1D9E75")
    fig_sc.update_layout(
        height=280, margin=dict(l=0,r=0,t=8,b=0),
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0"),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    if premium_sc > 0:
        st.success(f"✅ Neste cenário o portfólio supera o CDI em **{premium_sc:.2f}%** a.a.")
    else:
        st.warning(f"⚠️ Neste cenário o CDI supera o portfólio em **{abs(premium_sc):.2f}%** a.a. Considere revisar as visões no Black-Litterman.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style='text-align:center;font-size:12px;color:#b4b2a9;padding:.5rem 0'>
    HRP + Black-Litterman Dashboard &nbsp;·&nbsp;
    Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')} &nbsp;·&nbsp;
    CDI rf = {rf_ann*100:.2f}% a.a. ({common_idx[0].strftime('%b/%Y')}–{common_idx[-1].strftime('%b/%Y')})
</div>
""", unsafe_allow_html=True)
