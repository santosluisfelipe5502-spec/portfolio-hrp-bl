import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date
import io, warnings
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# ── Layout padrão Plotly (não usado diretamente — inline em cada gráfico) ──
_UNUSED = dict(
    plot_bgcolor="#f8f7f4",
    paper_bgcolor="#f8f7f4",
    font=dict(color="#1a1a18", size=12),
    xaxis=dict(
        color="#1a1a18",
        tickfont=dict(color="#444441", size=11),
        titlefont=dict(color="#444441"),
        gridcolor="#e8e6e0",
        linecolor="#e8e6e0",
        zerolinecolor="#c8c6c0",
    ),
    yaxis=dict(
        color="#1a1a18",
        tickfont=dict(color="#444441", size=11),
        titlefont=dict(color="#444441"),
        gridcolor="#e8e6e0",
        linecolor="#e8e6e0",
        zerolinecolor="#c8c6c0",
    ),
    legend=dict(
        font=dict(color="#1a1a18", size=12),
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
    ),
    hoverlabel=dict(
        font=dict(color="#1a1a18"),
        bgcolor="#ffffff",
        bordercolor="#e8e6e0",
    ),
    margin=dict(l=0, r=0, t=8, b=0),
)
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
[data-testid="stSidebar"]{background:#ffffff !important;border-right:1px solid #e8e6e0}
[data-testid="stSidebar"] *{color:#1a1a18 !important;background-color:transparent !important}
[data-testid="stSidebar"] small{color:#888780 !important}
[data-testid="stSidebar"] hr{border-color:#e8e6e0 !important}
[data-testid="stSidebar"] section{background:#ffffff !important}
[data-testid="stSidebar"] .stFileUploader{background:#f8f7f4 !important;border-radius:8px}
[data-testid="stSidebar"] .stFileUploader *{color:#1a1a18 !important}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:#f8f7f4 !important;border:1.5px dashed #c8c6c0 !important}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *{color:#444441 !important}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button{background:#378ADD !important;color:#ffffff !important;border:none !important}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *{color:#ffffff !important}
[data-testid="stSidebar"] .stMarkdown p{color:#444441 !important}
[data-testid="stSidebar"] h3{color:#1a1a18 !important}
[data-testid="stSidebar"] .stToggle label span{color:#1a1a18 !important}
[data-testid="stSidebar"] .stAlert{background:#EAF3DE !important}
[data-testid="stSidebar"] .stAlert *{color:#27500A !important}
[data-testid="stSidebar"] .stInfo{background:#E6F1FB !important}
[data-testid="stSidebar"] .stInfo *{color:#185FA5 !important}
body,p,span,div,label{color:#1a1a18}
label,.stSlider label,.stCheckbox label,.stSelectbox label,
.stFileUploader label,.stDateInput label,.stNumberInput label,
.stToggle label,[data-testid="stWidgetLabel"] p{color:#1a1a18 !important;font-size:13px !important}
.stCheckbox span p{color:#1a1a18 !important}
.stSlider [data-testid="stTickBarMin"],[data-testid="stTickBarMax"]{color:#1a1a18 !important}
.stTabs [data-baseweb="tab"]{color:#444441 !important}
.stTabs [aria-selected="true"]{color:#1a1a18 !important;font-weight:500 !important}
[data-testid="stSelectbox"]>div>div{background:#ffffff !important;color:#1a1a18 !important;border:1px solid #e8e6e0 !important}
[data-testid="stSelectbox"] svg{fill:#1a1a18 !important}
[data-baseweb="popover"],[data-baseweb="menu"]{background:#ffffff !important}
[data-baseweb="popover"] *,[data-baseweb="menu"] *{color:#1a1a18 !important;background:#ffffff !important}
[data-baseweb="option"]{background:#ffffff !important;color:#1a1a18 !important}
[data-baseweb="option"]:hover{background:#f8f7f4 !important;color:#1a1a18 !important}
li[role="option"]{background:#ffffff !important;color:#1a1a18 !important}
li[role="option"]:hover{background:#f8f7f4 !important}
ul[role="listbox"]{background:#ffffff !important}
ul[role="listbox"] li{color:#1a1a18 !important}
[data-testid="stMetricLabel"] p{color:#888780 !important;font-size:13px !important}
[data-testid="stMetricValue"]{color:#1a1a18 !important}
[data-testid="stMarkdownContainer"] p{color:#1a1a18 !important}
.stAlert p{color:#1a1a18 !important}
[data-testid="stDataFrame"] *{color:#1a1a18 !important}
.metric-card{background:#fff;border:1px solid #e8e6e0;border-radius:10px;padding:16px 20px;margin-bottom:8px}
.metric-label{font-size:11px;color:#888780 !important;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.metric-value{font-size:26px;font-weight:500;color:#1a1a18 !important;line-height:1.1}
.metric-sub{font-size:12px;color:#888780 !important;margin-top:4px}
.metric-value.pos{color:#0F6E56 !important}
.metric-value.neg{color:#A32D2D !important}
.metric-value.warn{color:#854F0B !important}
.section-title{font-size:11px;font-weight:500;letter-spacing:.08em;text-transform:uppercase;color:#888780 !important;margin:1.5rem 0 .5rem}
.badge{display:inline-block;font-size:11px;padding:2px 10px;border-radius:12px;font-weight:500}
.badge-green{background:#EAF3DE;color:#27500A !important}
.badge-blue{background:#E6F1FB;color:#185FA5 !important}
.badge-amber{background:#FAEEDA;color:#633806 !important}
.badge-red{background:#FCEBEB;color:#A32D2D !important}
div[data-testid="stHorizontalBlock"]{gap:12px}
</style>
""", unsafe_allow_html=True)


# ── Constantes ───────────────────────────────────────────────────────────────
ASSET_CFG = [
    {"name": "IRF-M",      "key": "IRFM",       "color": "#E24B4A", "cluster": "Renda fixa", "w": 0.268, "vol": 0.04},
    {"name": "IMA",        "key": "IMA",        "color": "#1D9E75", "cluster": "Renda fixa", "w": 0.188, "vol": 0.07},
    {"name": "IHFA",       "key": "IHFA",       "color": "#378ADD", "cluster": "Âncora",     "w": 0.172, "vol": 0.06},
    {"name": "IDA-DI",     "key": "IDADI",      "color": "#888780", "cluster": "Âncora",     "w": 0.144, "vol": 0.02},
    {"name": "Ibovespa",   "key": "IBOV",       "color": "#BA7517", "cluster": "Equity",     "w": 0.145, "vol": 0.24},
    {"name": "Internac.",  "key": "INTL",       "color": "#7F77DD", "cluster": "Equity",     "w": 0.083, "vol": 0.184},
]
WEIGHTS = {a["name"]: a["w"] for a in ASSET_CFG}

# ── Algoritmo HRP (López de Prado, 2016) ─────────────────────────────────────
def compute_hrp(returns_df):
    """Calcula pesos HRP dado um DataFrame de retornos mensais.
    Etapas: correlação → distância → clusterização → quasi-diag → alocação recursiva.
    """
    # Remover colunas com NaN excessivo
    returns_df = returns_df.dropna(axis=1, thresh=int(len(returns_df)*0.7))
    returns_df = returns_df.fillna(0)

    cov  = returns_df.cov()
    corr = returns_df.corr()

    # Etapa 1: Matriz de distância
    dist = np.sqrt((1 - corr) / 2)
    np.fill_diagonal(dist.values, 0)
    dist_sq = squareform(dist.values)

    # Etapa 2: Clusterização hierárquica
    link = linkage(dist_sq, method="single")
    ordem = leaves_list(link)
    ativos_ord = [corr.columns[i] for i in ordem]

    # Etapa 3: Quasi-diagonalização — reordenar covariância
    cov_ord = cov.loc[ativos_ord, ativos_ord]

    # Etapa 4: Alocação recursiva pelo inverso da variância
    def get_cluster_var(cov_m, items):
        sub = cov_m.loc[items, items]
        ivp = 1 / np.diag(sub.values)
        ivp /= ivp.sum()
        return float(ivp @ sub.values @ ivp)

    def hrp_alloc(cov_m, items):
        weights = pd.Series(1.0, index=items)
        clusters = [items]
        while clusters:
            clusters = [c for c in clusters if len(c) > 1]
            if not clusters:
                break
            new_clusters = []
            for cluster in clusters:
                mid = len(cluster) // 2
                c1, c2 = cluster[:mid], cluster[mid:]
                var1 = get_cluster_var(cov_m, c1)
                var2 = get_cluster_var(cov_m, c2)
                alpha = 1 - var1 / (var1 + var2)
                weights[c1] *= alpha
                weights[c2] *= 1 - alpha
                new_clusters += [c1, c2]
            clusters = new_clusters
        return weights / weights.sum()

    w = hrp_alloc(cov_ord, ativos_ord)
    return w.reindex([a["name"] for a in ASSET_CFG]).fillna(0)
COLORS  = {a["name"]: a["color"] for a in ASSET_CFG}

# ── Funções utilitárias ──────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_bcb_serie(serie_id, divisor=1.0, max_retries=3, timeout=30):
    """Busca qualquer série do SGS/BCB com retry automático e timeout generoso."""
    urls = [
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie_id}/dados?formato=json",
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie_id}/dados/ultimos/1000?formato=json",
    ]
    for url in urls:
        for attempt in range(max_retries):
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 200:
                    data = r.json()
                    if data:
                        df = pd.DataFrame(data)
                        df["valor"] = (df["valor"].astype(str)
                                       .str.replace(",",".")
                                       .astype(float)) / divisor
                        df["data"] = pd.to_datetime(df["data"], dayfirst=True,
                                                     errors="coerce")
                        df = df.dropna(subset=["data","valor"])
                        df = df.set_index("data").sort_index()
                        return df
            except Exception:
                if attempt < max_retries - 1:
                    import time; time.sleep(2)
                continue
    return None

def fetch_cdi():
    """Busca CDI mensal via API do Banco Central (SGS 4391)."""
    df = fetch_bcb_serie(4391, divisor=100.0)
    if df is not None:
        df.index = df.index + pd.offsets.MonthEnd(0)
        return df
    return None

def fetch_ipca():
    """Busca IPCA mensal via API do Banco Central (SGS 433)."""
    df = fetch_bcb_serie(433, divisor=100.0)
    if df is not None:
        df.index = df.index + pd.offsets.MonthEnd(0)
        return df
    return None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ptax():
    """Busca PTAX mensal (fechamento USD/BRL) via API do Banco Central.
    Tenta múltiplas séries e endpoints para maior robustez.
    """
    # Estratégia 1: SGS 3698 — USD/BRL mensal (média)
    # Estratégia 2: SGS 1 — USD/BRL diário, agrega para mensal
    # Estratégia 3: PTAX Olinda API
    # Estratégia 1: SGS 3698 — USD/BRL mensal
    df = fetch_bcb_serie(3698)
    if df is not None and len(df) > 12:
        df.index = df.index + pd.offsets.MonthEnd(0)
        df["retorno"] = df["valor"].pct_change()
        return df

    # Estratégia 2: SGS 1 — USD/BRL diário, agrega para mensal
    df = fetch_bcb_serie(1)
    if df is not None and len(df) > 12:
        df = df.resample("ME").last()
        df["retorno"] = df["valor"].pct_change()
        return df

    # Estratégia 3: Olinda PTAX API (endpoint diferente)
    try:
        import json
        url_olinda = (
            "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
            "CotacaoMoedaMensal(codigoMoeda=@codigoMoeda,competenciaInicio=@competenciaInicio,"
            "competenciaFim=@competenciaFim)?@codigoMoeda='USD'"
            "&@competenciaInicio='01-2005'&@competenciaFim='12-2026'"
            "&$format=json&$select=cotacaoCompra,dataHoraCotacao"
        )
        r = requests.get(url_olinda, timeout=20)
        if r.status_code == 200:
            data = r.json().get("value", [])
            if data:
                df = pd.DataFrame(data)
                df["data"]  = pd.to_datetime(df["dataHoraCotacao"], errors="coerce") + pd.offsets.MonthEnd(0)
                df["valor"] = pd.to_numeric(df["cotacaoCompra"], errors="coerce")
                df = df.dropna().set_index("data").sort_index()[["valor"]]
                df["retorno"] = df["valor"].pct_change()
                if len(df) > 12:
                    return df
    except Exception:
        pass

    return None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_yfinance(ticker, start="2002-01-01"):
    """Busca série mensal de qualquer ticker via yfinance.
    Funciona para SPY, TLT, ^BVSP e outros.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, interval="1mo",
                         auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            return None
        close = df["Close"].squeeze()
        close.index = close.index + pd.offsets.MonthEnd(0)
        close.index = close.index.tz_localize(None)
        result = pd.DataFrame({"valor": close.values}, index=close.index)
        return result.dropna().sort_index()
    except Exception:
        return None

def read_uploaded(file):
    """Lê XLS/XLSX/CSV em DataFrame com duas colunas: data, valor.
    Suporta formato ANBIMA (código, data, valor), Investing.com (Data, Último, ...)
    e outros formatos com detecção automática.
    """
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            # Tentar separadores comuns — SEM thousands para não corromper datas
            for sep in [",", ";", "\t", None]:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, sep=sep, engine="python")
                    if df.shape[1] >= 2:
                        break
                except Exception:
                    continue
        else:
            df = pd.read_excel(file, engine="openpyxl")

        df.columns = [str(c).strip() for c in df.columns]

        # ── Detectar coluna de data ──
        date_col = next(
            (c for c in df.columns if any(k in c.lower()
             for k in ["data","date","dt","período","periodo"])),
            df.columns[0]
        )

        # ── Detectar coluna de valor ──
        # Prioridade: "Último"/"Ultimo" (Investing.com) > "valor"/"value" > col[2] (ANBIMA)
        val_col = None
        for pref in ["último", "ultimo", "fechamento", "close", "valor", "value", "cota"]:
            match = next((c for c in df.columns
                          if pref in c.lower() and c != date_col), None)
            if match:
                val_col = match
                break
        if val_col is None:
            # Fallback ANBIMA: col[2] se tiver 3+ colunas
            val_col = df.columns[2] if df.shape[1] >= 3 else df.columns[1]

        # ── Parsear data ──
        # Suporta DD.MM.YYYY (Investing.com) e DD/MM/YYYY (ANBIMA/BCB)
        raw_dates = df[date_col].astype(str).str.strip().str.split(" ").str[0]
        parsed_dates = None
        for fmt in ["%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"]:
            try:
                parsed_dates = pd.to_datetime(raw_dates, format=fmt, errors="coerce")
                if parsed_dates.notna().sum() > len(raw_dates) * 0.8:
                    break
            except Exception:
                continue
        if parsed_dates is None:
            parsed_dates = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")

        # ── Parsear valor ──
        # Se já for numérico (Investing.com), usar direto
        if pd.api.types.is_numeric_dtype(df[val_col]):
            parsed_vals = pd.to_numeric(df[val_col], errors="coerce")
        else:
            # Tratar formato BR: 1.234,56 → 1234.56
            raw_vals = df[val_col].astype(str).str.strip()
            raw_vals = (raw_vals
                        .str.replace(r"[^0-9,.-]", "", regex=True)
                        .str.replace(".", "", regex=False)
                        .str.replace(",", ".", regex=False))
            parsed_vals = pd.to_numeric(raw_vals, errors="coerce")

        result = pd.DataFrame({"data": parsed_dates, "valor": parsed_vals})
        result = result.dropna(subset=["data","valor"])
        result = result.set_index("data").sort_index()
        return result

    except Exception as e:
        st.error(f"Erro ao ler {file.name}: {e}")
        return None

def to_monthly(df):
    return df.resample("ME").last()

def calc_intl(spy_df, tlt_df, w_spy=0.40, w_tlt=0.60, ptax_df=None):
    """Calcula índice Internacional combinado.
    Se ptax_df for fornecido, converte retornos USD → BRL via PTAX.
    """
    df = pd.concat([spy_df.rename(columns={"valor":"SPY"}),
                    tlt_df.rename(columns={"valor":"TLT"})], axis=1).dropna()
    ret_usd = w_spy * df["SPY"].pct_change() + w_tlt * df["TLT"].pct_change()

    if ptax_df is not None:
        # Alinhar PTAX ao mesmo índice e converter USD → BRL
        ptax_ret = ptax_df["retorno"].reindex(ret_usd.index).ffill().fillna(0)
        ret_brl  = (1 + ret_usd) * (1 + ptax_ret) - 1
        ret = ret_brl
    else:
        ret = ret_usd

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
        label = (f"{cfg['name']} (CSV Investing.com ou Excel B3)"
                 if cfg["name"] == "Ibovespa" else cfg["name"])
        uploads[cfg["name"]] = st.file_uploader(
            label, type=["csv","xls","xlsx"], key=cfg["key"]
        )

    st.markdown("**Internacional** (40% SPY + 60% TLT)")
    spy_file = st.file_uploader("SPY",     type=["csv","xls","xlsx"], key="SPY")
    tlt_file = st.file_uploader("TLT",     type=["csv","xls","xlsx"], key="TLT")
    st.markdown("**PTAX (USD/BRL)** — conversão cambial")
    use_auto_ptax = st.toggle("Buscar PTAX automaticamente (BCB)", value=True)
    ptax_file = None
    if not use_auto_ptax:
        ptax_file = st.file_uploader("Upload PTAX (CSV/Excel)", type=["csv","xls","xlsx"], key="ptax")

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
def load_from_repo(filename):
    """Tenta carregar arquivo da pasta dados/ do repositório.
    Suporta múltiplas extensões automaticamente.
    """
    import os
    bases = ["dados", "data", "."]
    exts  = ["", ".csv", ".xls", ".xlsx"]
    for base in bases:
        for ext in exts:
            path = os.path.join(base, filename + ext) if ext else os.path.join(base, filename)
            if os.path.exists(path):
                try:
                    if path.endswith(".csv"):
                        # Detectar separador
                        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
                            sample = f.read(2048)
                        sep = ";" if sample.count(";") > sample.count(",") else ","
                        df = pd.read_csv(path, sep=sep, engine="python")
                    else:
                        df = pd.read_excel(path, engine="openpyxl")
                    return df, path
                except Exception:
                    continue
    return None, None

def read_from_df(df, path):
    """Converte DataFrame bruto para série (data, valor) usando o mesmo
    parser do read_uploaded, mas a partir de um DataFrame já carregado."""
    try:
        df.columns = [str(c).strip() for c in df.columns]
        date_col = next(
            (c for c in df.columns if any(k in c.lower()
             for k in ["data","date","dt","período","periodo"])),
            df.columns[0]
        )
        val_col = None
        for pref in ["último","ultimo","fechamento","close","valor","value","cota"]:
            match = next((c for c in df.columns
                          if pref in c.lower() and c != date_col), None)
            if match:
                val_col = match
                break
        if val_col is None:
            val_col = df.columns[2] if df.shape[1] >= 3 else df.columns[1]

        raw_dates = df[date_col].astype(str).str.strip().str.split(" ").str[0]
        parsed_dates = None
        for fmt in ["%d.%m.%Y","%d/%m/%Y","%Y-%m-%d","%m/%d/%Y"]:
            try:
                parsed_dates = pd.to_datetime(raw_dates, format=fmt, errors="coerce")
                if parsed_dates.notna().sum() > len(raw_dates)*0.8:
                    break
            except Exception:
                continue
        if parsed_dates is None:
            parsed_dates = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")

        if pd.api.types.is_numeric_dtype(df[val_col]):
            parsed_vals = pd.to_numeric(df[val_col], errors="coerce")
        else:
            raw_vals = (df[val_col].astype(str).str.strip()
                        .str.replace(r"[^0-9,.-]","",regex=True)
                        .str.replace(".","",regex=False)
                        .str.replace(",",".",regex=False))
            parsed_vals = pd.to_numeric(raw_vals, errors="coerce")

        result = pd.DataFrame({"data":parsed_dates,"valor":parsed_vals})
        result = result.dropna(subset=["data","valor"]).set_index("data").sort_index()
        return result if len(result) > 0 else None
    except Exception:
        return None

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
    params = [(0.085,0.04,208,1),(0.095,0.07,208,2),(0.085,0.06,208,3),
              (0.125,0.02,208,4),(0.092,0.238,208,5),(0.118,0.184,208,6)]
    keys = ["IRF-M","IMA","IHFA","IDA-DI","Ibovespa","Internac."]
    return {k: gen(*p) for k,p in zip(keys, params)}

with st.spinner("Carregando dados e conectando ao Banco Central…"):
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
        # CDI simulado — fallback silencioso na sidebar
        idx = pd.date_range("2009-01-31", periods=208, freq="ME")
        cdi_vals = np.where(idx.year < 2017, 0.011,
                   np.where(idx.year < 2020, 0.005,
                   np.where(idx.year < 2022, 0.003, 0.011)))
        cdi_raw = pd.DataFrame({"valor": cdi_vals}, index=idx)
        st.sidebar.info("ℹ️ CDI simulado — API BCB lenta, tente recarregar em instantes")

    # Séries de ativos
    # Ordem de prioridade: 1) upload manual  2) pasta dados/ do repo  3) demo simulado
    series = {}
    demo = load_demo_series()
    has_real = False

    REPO_FILES = {
        "IRF-M":     "IRFM",
        "IMA":       "IMA",
        "IHFA":      "IHFA",
        "IDA-DI":    "IDADI",
        "Ibovespa":  "Ibovespa",
    }

    for cfg in ASSET_CFG:
        if cfg["key"] == "INTL":
            continue
        nome = cfg["name"]

        # 1. Upload manual (prioridade máxima)
        f = uploads.get(nome)
        if f:
            raw = read_uploaded(f)
            if raw is not None:
                series[nome] = to_monthly(raw)
                has_real = True
                continue

        # 2. yfinance automático (só para Ibovespa)
        if nome == "Ibovespa":
            ibov_yf = fetch_yfinance("^BVSP", start="2005-01-01")
            if ibov_yf is not None and len(ibov_yf) > 12:
                series[nome] = to_monthly(ibov_yf)
                has_real = True
                continue

        # 3. Arquivo na pasta dados/ do repositório
        repo_df, repo_path = load_from_repo(REPO_FILES.get(nome, nome))
        if repo_df is not None:
            raw = read_from_df(repo_df, repo_path)
            if raw is not None:
                series[nome] = to_monthly(raw)
                has_real = True
                continue

        # 4. Fallback: dados simulados
        series[nome] = demo[nome]

    # PTAX — buscar automaticamente ou via upload
    ptax_data = None
    if use_auto_ptax:
        with st.spinner("Buscando PTAX..."):
            ptax_data = fetch_ptax()
        if ptax_data is not None:
            ptax_src = "📡 PTAX via BCB"
        else:
            ptax_src = "⚠️ PTAX indisponível — usando USD puro"
    elif ptax_file:
        ptax_raw = read_uploaded(ptax_file)
        if ptax_raw is not None:
            ptax_raw.index = ptax_raw.index + pd.offsets.MonthEnd(0)
            ptax_raw["retorno"] = ptax_raw["valor"].pct_change()
            ptax_data = ptax_raw
            ptax_src = "📂 PTAX local"
        else:
            ptax_src = "⚠️ PTAX não carregada — usando USD puro"
    else:
        ptax_src = "⚠️ PTAX desabilitada — usando USD puro"

    # Internacional — prioridade: upload > yfinance automático > repo > demo
    spy_df, tlt_df = None, None

    if spy_file:
        spy_df = read_uploaded(spy_file)

    if spy_df is None:
        # Tentar yfinance automático
        spy_yf = fetch_yfinance("SPY")
        if spy_yf is not None and len(spy_yf) > 12:
            spy_df = spy_yf

    if spy_df is None:
        repo_spy, rp = load_from_repo("SPY")
        if repo_spy is not None:
            spy_df = read_from_df(repo_spy, rp)

    if tlt_file:
        tlt_df = read_uploaded(tlt_file)

    if tlt_df is None:
        tlt_yf = fetch_yfinance("TLT")
        if tlt_yf is not None and len(tlt_yf) > 12:
            tlt_df = tlt_yf

    if tlt_df is None:
        repo_tlt, rp = load_from_repo("TLT")
        if repo_tlt is not None:
            tlt_df = read_from_df(repo_tlt, rp)

    if spy_df is not None and tlt_df is not None:
        spy_m = to_monthly(spy_df)
        tlt_m = to_monthly(tlt_df)
        series["Internac."] = calc_intl(spy_m, tlt_m, spy_w, tlt_w, ptax_data)
        has_real = True
    else:
        series["Internac."] = demo["Internac."]

    # ── Séries diárias para monitoramento ────────────────────────────────────
    # Carrega as séries no formato diário (sem agregar para mensal)
    daily_series = {}
    DAILY_FILES = {"IRF-M":"IRFM","IMA":"IMA","IHFA":"IHFA","IDA-DI":"IDADI"}
    for cfg in ASSET_CFG:
        if cfg["key"] in ["IBOV","INTL"]:
            continue
        nome = cfg["name"]
        f_up = uploads.get(nome)
        if f_up:
            raw = read_uploaded(f_up)
            if raw is not None:
                daily_series[nome] = raw
                continue
        repo_df, rp = load_from_repo(DAILY_FILES.get(nome, nome))
        if repo_df is not None:
            raw = read_from_df(repo_df, rp)
            if raw is not None:
                daily_series[nome] = raw

    # Ibovespa diário via yfinance
    # Ibovespa diário — prioridade: yfinance > arquivo repositório
    ibov_daily_yf = fetch_yfinance("^BVSP", start="2005-01-01")
    if ibov_daily_yf is not None and len(ibov_daily_yf) > 30:
        daily_series["Ibovespa"] = ibov_daily_yf
    else:
        # Fallback: usar arquivo Ibovespa.csv do repositório (dados diários Investing.com)
        repo_ibov_d, _ = load_from_repo("Ibovespa")
        if repo_ibov_d is not None:
            ibov_raw_d = read_from_df(repo_ibov_d, "Ibovespa.csv")
            if ibov_raw_d is not None and len(ibov_raw_d) > 30:
                daily_series["Ibovespa"] = ibov_raw_d

    # Internacional diário via yfinance
    spy_d = fetch_yfinance("SPY")
    tlt_d = fetch_yfinance("TLT")
    if spy_d is not None and tlt_d is not None:
        ret_spy_d = spy_d["valor"].pct_change()
        ret_tlt_d = tlt_d["valor"].pct_change()
        # Alinhar
        idx_d = ret_spy_d.index.intersection(ret_tlt_d.index)
        ret_intl_d = 0.40*ret_spy_d.reindex(idx_d) + 0.60*ret_tlt_d.reindex(idx_d)
        if ptax_data is not None:
            ptax_d = ptax_data["retorno"].reindex(idx_d, method="ffill").fillna(0)
            ret_intl_d = (1+ret_intl_d)*(1+ptax_d)-1
        intl_daily = (1+ret_intl_d).cumprod()*100
        daily_series["Internac."] = pd.DataFrame({"valor": intl_daily.values},
                                                   index=intl_daily.index)

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

    # ── IPCA puro ─────────────────────────────────────────────────────────────
    # IPCA mensal via API BCB (série 433) ou fallback com média histórica
    ipca_raw = fetch_ipca()
    if ipca_raw is not None:
        ipca_ret = ipca_raw["valor"].reindex(common_idx).ffill().fillna(0.004)
        ipca_src = "BCB"
    else:
        idx_yr = common_idx.year
        ipca_vals = np.where(idx_yr < 2016, 0.006,
                    np.where(idx_yr < 2019, 0.003,
                    np.where(idx_yr < 2022, 0.004, 0.005)))
        ipca_ret = pd.Series(ipca_vals, index=common_idx)
        ipca_src = "estimado"

    ipca6_ret = ipca_ret   # usando IPCA puro (variável mantida para compatibilidade)
    ipca6_cum = (1 + ipca6_ret).cumprod() * 100
    acum_ipca6 = (ipca6_cum.iloc[-1] / 100 - 1) * 100
    ann_ret_ipca6 = (1 + ipca6_ret.mean()) ** 12 - 1

# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    _cdi_ok = fetch_cdi() is not None
    _spy_ok = fetch_yfinance("SPY") is not None
    data_src = "📡 CDI via BCB" if (use_auto_cdi and _cdi_ok) else "📂 CDI local"
    intl_src = "📡 SPY+TLT via Yahoo" if _spy_ok else "📂 SPY+TLT local"
    real_tag = "📂 dados reais" if has_real else "🔬 dados simulados"
    # Verificar se veio do repo
    import os
    repo_ok = os.path.exists("dados") or any(
        os.path.exists(f) for f in ["IHFA.xls","IHFA.xlsx","IHFA.csv","dados/IHFA.xls"]
    )
    if has_real and repo_ok and not any([
        uploads.get("IRF-M"), uploads.get("IMA"), uploads.get("IHFA"),
        uploads.get("IDA-DI"), uploads.get("Ibovespa")
    ]):
        real_tag = "📁 dados do repositório"
    st.markdown(f"""
    <h2 style='margin:0;font-size:24px;font-weight:500;color:#1a1a18'>
        HRP + Black-Litterman
    </h2>
    <p style='margin:4px 0 0;font-size:13px;color:#888780'>
        {common_idx[0].strftime('%b/%Y')} → {common_idx[-1].strftime('%b/%Y')} &nbsp;·&nbsp;
        {len(common_idx)} meses &nbsp;·&nbsp; {data_src} &nbsp;·&nbsp;
        {ptax_src} &nbsp;·&nbsp; {real_tag}
    </p>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style='text-align:right;padding-top:4px'>
        <span class='badge badge-blue'>rf = CDI médio {rf_ann*100:.2f}% a.a.</span>
    </div>
    """, unsafe_allow_html=True)

# ── Alertas de rebalanceamento ────────────────────────────────────────────────
# Calcula drift atual dos pesos e exibe alerta se algum ativo ultrapassar a banda
_banda_alerta = st.session_state.get("banda", 3.0)
# Ler pesos atuais — verificar todas as chaves possíveis do session_state
_pesos_atuais = {}
for cfg in ASSET_CFG:
    chave = f"rebal_{cfg['name']}"
    # Tentar ler do session_state, fallback para o peso target
    val = st.session_state.get(chave, None)
    if val is None:
        val = cfg["w"] * 100
    _pesos_atuais[cfg["name"]] = float(val)

_total_atual = sum(_pesos_atuais.values())

# Calcular drift apenas se os pesos foram customizados (total != 100 ou algum diferente)
_pesos_sao_default = all(
    abs(_pesos_atuais[cfg["name"]] - cfg["w"]*100) < 0.05
    for cfg in ASSET_CFG
)

_ativos_fora = []
_ativos_atencao = []

if not _pesos_sao_default:
    for cfg in ASSET_CFG:
        _w_atual  = _pesos_atuais[cfg["name"]]
        _w_target = cfg["w"] * 100
        _drift    = _w_atual - _w_target
        if abs(_drift) > _banda_alerta:
            _ativos_fora.append((cfg["name"], _drift, cfg["color"]))
        elif abs(_drift) > _banda_alerta * 0.7:
            _ativos_atencao.append((cfg["name"], _drift, cfg["color"]))

if _ativos_fora:
    _msgs = " · ".join([f"**{n}** {d:+.1f}%" for n, d, _ in _ativos_fora])
    st.warning(
        f"⚠️ **Rebalanceamento necessário** — {len(_ativos_fora)} ativo(s) "
        f"fora da banda ±{_banda_alerta:.0f}%: {_msgs} · "
        f"Acesse a aba **🔄 Rebalanceamento** para ver os trades sugeridos."
    )
elif _ativos_atencao:
    _msgs = " · ".join([f"**{n}** {d:+.1f}%" for n, d, _ in _ativos_atencao])
    st.info(
        f"ℹ️ **Atenção** — {len(_ativos_atencao)} ativo(s) "
        f"se aproximando da banda ±{_banda_alerta:.0f}%: {_msgs}"
    )

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

# ── Linha 1: HRP+BL original ──
st.markdown("<div style='font-size:11px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;"
            "color:#378ADD;margin-bottom:6px'>HRP + Black-Litterman</div>", unsafe_allow_html=True)
k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
k1.markdown(kpi("Acumulado", f"+{acum_port:.1f}%", f"CDI +{acum_cdi:.1f}%", "pos"), unsafe_allow_html=True)
k2.markdown(kpi("Retorno a.a.", f"{m_port['ann_ret']*100:.2f}%", f"IBOV {m_port['ann_ret']*100 - m_ibov['ann_ret']*100:+.1f}%"), unsafe_allow_html=True)
k3.markdown(kpi("Volatilidade a.a.", f"{m_port['ann_vol']*100:.2f}%", f"IBOV {m_ibov['ann_vol']*100:.1f}%", "good"), unsafe_allow_html=True)
cls_sharpe = "pos" if m_port["sharpe"] > 0.2 else "warn" if m_port["sharpe"] > 0 else "neg"
k4.markdown(kpi("Sharpe (rf=CDI)", f"{m_port['sharpe']:.3f}", f"IBOV {m_ibov['sharpe']:.3f}", cls_sharpe), unsafe_allow_html=True)
cls_sort = "pos" if m_port["sortino"] and m_port["sortino"] > 0.3 else "warn"
k5.markdown(kpi("Sortino", f"{m_port['sortino']:.3f}" if m_port["sortino"] else "—", "penaliza só quedas", cls_sort), unsafe_allow_html=True)
k6.markdown(kpi("Max Drawdown", f"{m_port['max_dd']*100:.2f}%", f"IBOV {m_ibov['max_dd']*100:.1f}%", "warn"), unsafe_allow_html=True)
k7.markdown(kpi("Calmar ratio", f"{m_port['calmar']:.3f}", f"IBOV {m_ibov['calmar']:.3f}", "pos"), unsafe_allow_html=True)

# ── Linha 2: Portfólio customizado ──
custom_w_hdr = {}
for cfg in ASSET_CFG:
    key = f"rebal_{cfg['name']}"
    custom_w_hdr[cfg["name"]] = st.session_state.get(key, cfg["w"] * 100) / 100
total_cw_hdr = sum(custom_w_hdr.values())
custom_valid_hdr = abs(total_cw_hdr - 1.0) < 0.02

if custom_valid_hdr:
    c_ret = sum(
        custom_w_hdr[a["name"]] * series[a["name"]]["valor"].pct_change().dropna().reindex(common_idx).ffill()
        for a in ASSET_CFG
    )
    c_cum = (1 + c_ret).cumprod() * 100
    m_cust = metrics(c_ret, cdi_aligned)
    acum_cust = (c_cum.iloc[-1] / 100 - 1) * 100

    st.markdown("<div style='font-size:11px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;"
                "color:#E24B4A;margin:8px 0 6px'>Portfólio customizado</div>", unsafe_allow_html=True)
    ck1,ck2,ck3,ck4,ck5,ck6,ck7 = st.columns(7)
    diff_acum = acum_cust - acum_port
    ck1.markdown(kpi("Acumulado", f"+{acum_cust:.1f}%",
        f"{'↑' if diff_acum>=0 else '↓'} {diff_acum:+.1f}% vs HRP+BL",
        "pos" if acum_cust >= acum_port else "warn"), unsafe_allow_html=True)
    diff_ret = (m_cust["ann_ret"] - m_port["ann_ret"])*100
    ck2.markdown(kpi("Retorno a.a.", f"{m_cust['ann_ret']*100:.2f}%",
        f"{diff_ret:+.1f}% vs HRP+BL",
        "pos" if diff_ret >= 0 else "warn"), unsafe_allow_html=True)
    diff_vol = (m_cust["ann_vol"] - m_port["ann_vol"])*100
    cls_vol = "pos" if diff_vol <= 0 else "warn"
    ck3.markdown(kpi("Volatilidade a.a.", f"{m_cust['ann_vol']*100:.2f}%",
        f"{diff_vol:+.2f}% vs HRP+BL", cls_vol), unsafe_allow_html=True)
    cls_cs = "pos" if m_cust["sharpe"] > 0.2 else "warn" if m_cust["sharpe"] > 0 else "neg"
    diff_sh = m_cust["sharpe"] - m_port["sharpe"]
    ck4.markdown(kpi("Sharpe (rf=CDI)", f"{m_cust['sharpe']:.3f}",
        f"{diff_sh:+.3f} vs HRP+BL", cls_cs), unsafe_allow_html=True)
    cls_cso = "pos" if m_cust["sortino"] and m_cust["sortino"] > 0.3 else "warn"
    ck5.markdown(kpi("Sortino", f"{m_cust['sortino']:.3f}" if m_cust["sortino"] else "—",
        "penaliza só quedas", cls_cso), unsafe_allow_html=True)
    diff_dd = m_cust["max_dd"]*100 - m_port["max_dd"]*100
    ck6.markdown(kpi("Max Drawdown", f"{m_cust['max_dd']*100:.2f}%",
        f"{diff_dd:+.1f}% vs HRP+BL",
        "pos" if diff_dd >= 0 else "warn"), unsafe_allow_html=True)
    diff_cal = m_cust["calmar"] - m_port["calmar"]
    ck7.markdown(kpi("Calmar ratio", f"{m_cust['calmar']:.3f}",
        f"{diff_cal:+.3f} vs HRP+BL",
        "pos" if diff_cal >= 0 else "warn"), unsafe_allow_html=True)
else:
    st.markdown("<div style='font-size:11px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;"
                "color:#888780;margin:8px 0 6px'>Portfólio customizado — ajuste os pesos na aba Rebalanceamento</div>",
                unsafe_allow_html=True)

st.divider()


# ── Eventos de cauda ─────────────────────────────────────────────────────────
TAIL_EVENTS = [
    {"name": "Manifestações de Junho",  "start": "2013-05-31", "end": "2013-07-31", "color": "#E24B4A",
     "desc": "Protestos generalizados, queda de confiança, pressão sobre ativos de risco.", "tipo": "Doméstico"},
    {"name": "Greve dos Caminhoneiros", "start": "2018-05-31", "end": "2018-06-30", "color": "#BA7517",
     "desc": "Paralisação nacional, pressão inflacionária e incerteza política pré-eleitoral.", "tipo": "Doméstico"},
    {"name": "Eleições 2018",           "start": "2018-09-30", "end": "2018-10-31", "color": "#378ADD",
     "desc": "Alta volatilidade eleitoral, rali de ativos de risco após resultado.", "tipo": "Doméstico"},
    {"name": "COVID-19",                "start": "2020-01-31", "end": "2020-03-31", "color": "#7F77DD",
     "desc": "Crash global sincronizado, fuga para liquidez, colapso de ativos de risco. Ibov −29.9% em mar/20.", "tipo": "Global"},
    {"name": "Invasão da Ucrânia",      "start": "2022-01-31", "end": "2022-03-31", "color": "#888780",
     "desc": "Choque de commodities, inflação global, ciclo de alta de juros nos EUA.", "tipo": "Global"},
    {"name": "Crise SVB",               "start": "2023-02-28", "end": "2023-03-31", "color": "#1D9E75",
     "desc": "Colapso de bancos regionais americanos, fuga para treasuries.", "tipo": "Global"},
    {"name": "Americanas + 8/Jan",      "start": "2022-12-31", "end": "2023-01-31", "color": "#E24B4A",
     "desc": "Fraude contábil de R$20bi + crise política. Choque de crédito local.", "tipo": "Doméstico"},
    {"name": "Ataque Irã → Israel",     "start": "2024-03-31", "end": "2024-04-30", "color": "#854F0B",
     "desc": "Escalada geopolítica, pressão sobre petróleo e ativos de risco globais.", "tipo": "Global"},
    {"name": "Crise fiscal BR",         "start": "2024-10-31", "end": "2024-12-31", "color": "#E24B4A",
     "desc": "Pacote de gastos sem cobertura, colapso do real, abertura brutal da curva.", "tipo": "Doméstico"},
    {"name": "Tarifas Trump",           "start": "2025-03-31", "end": "2025-04-30", "color": "#7F77DD",
     "desc": "Guerra comercial global, crash do SPY, fuga de risco generalizada.", "tipo": "Global"},
    {"name": "Joesley Day",             "start": "2017-04-30", "end": "2017-05-31", "color": "#C4770A",
     "desc": "Delação JBS/Joesley Batista — maior queda diária do Ibovespa em anos (−8%). Crise política aguda.", "tipo": "Doméstico"},
    {"name": "Invasão Irã-EUA",         "start": "2026-02-28", "end": "2026-03-31", "color": "#854F0B",
     "desc": "Escalada militar entre Irã e EUA, pressão sobre commodities e ativos de risco globais.", "tipo": "Global"},
    {"name": "Bloqueio Ormuz",          "start": "2026-04-13", "end": "2026-04-30", "color": "#E24B4A",
     "desc": "Bloqueio do Estreito de Ormuz, choque de petróleo e fuga de risco global.", "tipo": "Global"},
]


def hex_to_rgba(hex_color, alpha=0.5):
    """Converte cor hex para rgba com transparência."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color

# ── Tabs principais ─────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "📖 Guia",
    "📈 Retorno acumulado",
    "📉 Drawdown",
    "📊 Métricas",
    "🔄 Rebalanceamento",
    "🌐 Cenários",
    "⚡ Eventos de cauda",
    "📦 Ativos individuais",
    "🔬 Análise comparativa",
    "📡 Monitoramento diário",
    "📐 Janelas móveis",
    "🎲 Monte Carlo",
    "📊 Atribuição de retorno",
])

# ── Tab 0: Guia ──────────────────────────────────────────────────────────────
with tab0:
    st.markdown("""
    <div style='max-width:900px'>
    <h2 style='font-size:22px;font-weight:500;color:#1a1a18;margin-bottom:4px'>
        Guia do Dashboard HRP + Black-Litterman
    </h2>
    <p style='color:#888780;font-size:13px;margin-bottom:2rem'>
        Referência rápida sobre o modelo, os índices, as métricas e como usar cada aba.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Seção 1: O Modelo ─────────────────────────────────────────────────────
    with st.expander("🧠 O modelo — HRP + Black-Litterman", expanded=True):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
**Hierarchical Risk Parity (HRP)**
_López de Prado, 2016_

Distribui o risco do portfólio de forma hierárquica — primeiro entre grupos de ativos
similares (clusters), depois dentro de cada grupo. Diferente do Risk Parity clássico,
o HRP nunca inverte a matriz de covariância, tornando os pesos mais estáveis e robustos.

**Como funciona na prática:**
- Calcula as correlações entre todos os ativos
- Agrupa os similares (ex: IRF-M e IMA ficam juntos por serem renda fixa)
- Distribui o risco proporcionalmente ao inverso da variância de cada grupo
- Resultado: portfólio diversificado sem concentrações indesejadas
            """)
        with col_m2:
            st.markdown("""
**Black-Litterman (BL)**
_Black & Litterman, 1992_

Combina o equilíbrio histórico do mercado com visões táticas do investidor usando
estatística Bayesiana. O resultado é um vetor de retornos esperados que incorpora
tanto o que os dados históricos dizem quanto o que você acredita que vai acontecer.

**Como funciona na prática:**
- Parte dos retornos de equilíbrio (pesos HRP × covariância)
- Incorpora visões táticas (cenários macro que você define)
- Combina os dois com peso proporcional à sua confiança
- Resultado: retornos esperados ajustados que alimentam o HRP
            """)
        st.info(
            "💡 **Juntos:** o HRP define a estrutura de risco e o BL define as expectativas "
            "de retorno. Os pesos finais refletem tanto a diversificação robusta do HRP "
            "quanto as visões táticas do investidor via BL."
        )

    # ── Seção 2: Os Ativos ────────────────────────────────────────────────────
    with st.expander("📦 Os ativos da carteira", expanded=False):
        ativos_info = [
            ("IRF-M", "Renda fixa", "#E24B4A", "26.8%", "ANBIMA",
             "Índice de títulos públicos **prefixados** (LTN e NTN-F). "
             "Representa o mercado de renda fixa pré com duration variável (~2.5 anos). "
             "Sobe quando os juros caem e cai quando os juros sobem (efeito MTM)."),
            ("IMA-Geral", "Renda fixa", "#1D9E75", "18.8%", "ANBIMA",
             "Índice de títulos públicos atrelados ao **IPCA** (NTN-B). "
             "Protege contra a inflação e tem duration longa (~6 anos). "
             "Muito sensível a movimentos na curva de juros reais."),
            ("IHFA", "Âncora", "#378ADD", "17.2%", "ANBIMA",
             "Índice de **fundos multimercado** brasileiros. "
             "Representa a capacidade dos gestores de gerar alpha independente "
             "do ciclo de mercado. Tem correlação baixa com os demais ativos."),
            ("IDA-DI", "Âncora", "#888780", "14.4%", "ANBIMA",
             "Índice de **debêntures atreladas ao CDI** (crédito privado pós-fixado). "
             "Entrega CDI + spread de crédito. O spread abre em momentos de stress "
             "(ex: caso Americanas) — capturando o risco de crédito do mercado corporativo."),
            ("Ibovespa", "Equity", "#BA7517", "14.5%", "Yahoo Finance / Investing.com",
             "Índice das **ações** mais negociadas na B3. "
             "Representa o mercado acionário brasileiro. Alta volatilidade mas "
             "potencial de retorno superior no longo prazo. Descorrelacionado da renda fixa."),
            ("Internacional", "Equity", "#7F77DD", "8.3%", "Yahoo Finance (SPY + TLT)",
             "**40% SPY** (S&P 500) + **60% TLT** (Treasuries 20 anos), convertidos para BRL via PTAX. "
             "Diversificação geográfica e proteção cambial — quando o Brasil vai mal, "
             "o dólar sobe e amplifica o retorno desta parcela em reais."),
        ]
        for nome, cluster, cor, peso, fonte, desc in ativos_info:
            st.markdown(
                f"<div style='border-left:3px solid {cor};padding:10px 16px;"
                f"margin-bottom:10px;background:#f8f7f4;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<strong style='font-size:14px;color:#1a1a18'>{nome}</strong>"
                f"<span style='font-size:11px;color:#888780'>{cluster} · Peso: {peso} · Fonte: {fonte}</span>"
                f"</div>"
                f"<p style='font-size:13px;color:#444441;margin:6px 0 0'>{desc}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

    # ── Seção 3: As Métricas ──────────────────────────────────────────────────
    with st.expander("📊 As métricas — o que cada uma mede", expanded=False):
        metricas = [
            ("Retorno acumulado", "pos",
             "Total ganho no período, em percentual. Se o portfólio foi de R$100 para R$615, "
             "o retorno acumulado é +515%."),
            ("Retorno a.a.", "good",
             "Taxa de retorno anualizada — equivalente a quanto o portfólio rendeu "
             "por ano em média, considerando juros compostos."),
            ("Volatilidade a.a.", "warn",
             "Desvio padrão dos retornos mensais, anualizado. Mede o quanto o portfólio "
             "oscila. Menor volatilidade = trajetória mais suave."),
            ("Sharpe (rf=CDI)", "pos",
             "Retorno excedente ao CDI dividido pela volatilidade. "
             "Responde: 'quanto de retorno extra você recebeu por unidade de risco?' "
             "Acima de 0.5 é excelente para um portfólio conservador."),
            ("Sortino", "pos",
             "Similar ao Sharpe mas considera apenas a volatilidade das quedas "
             "(não das altas). Penaliza só o risco ruim — mais justo para portfólios "
             "assimétricos que têm mais altas que quedas."),
            ("Max Drawdown", "warn",
             "Maior queda do portfólio a partir de um pico. Se o portfólio foi a 100, "
             "caiu para 93 e depois voltou, o drawdown foi -7%. "
             "Mede o pior cenário que o investidor teria vivido."),
            ("Calmar Ratio", "pos",
             "Retorno anualizado dividido pelo Max Drawdown. "
             "Quanto maior, melhor o portfólio compensa o risco de queda. "
             "Acima de 1.0 é considerado bom."),
            ("VaR 95% mensal", "warn",
             "Value at Risk — em 95% dos meses o portfólio não perde mais do que esse valor. "
             "Em 5% dos meses (os piores), a perda pode ser maior."),
        ]
        cols_met = st.columns(2)
        for i, (nome, cls, desc) in enumerate(metricas):
            cor_m = {"pos":"#0F6E56","warn":"#854F0B","good":"#185FA5"}.get(cls,"#1a1a18")
            with cols_met[i % 2]:
                st.markdown(
                    f"<div style='margin-bottom:12px;padding:10px 14px;"
                    f"background:#f8f7f4;border-radius:8px;"
                    f"border-top:2px solid {cor_m}'>"
                    f"<strong style='font-size:13px;color:{cor_m}'>{nome}</strong>"
                    f"<p style='font-size:12px;color:#444441;margin:4px 0 0'>{desc}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # ── Seção 4: As Abas ─────────────────────────────────────────────────────
    with st.expander("🗂️ O que cada aba mostra e quando usar", expanded=False):
        abas_info = [
            ("📈 Retorno acumulado", "Visão principal do portfólio. Compare HRP+BL com CDI, Ibovespa, IPCA e o portfólio customizado. Use o filtro de período para ver janelas específicas e ative os eventos de cauda para contextualizar os movimentos."),
            ("📉 Drawdown", "Mostra as quedas do portfólio a partir dos picos. Use para entender o risco real vivido pelo investidor e comparar com o Ibovespa. Quanto mais rasa a linha, mais protegido o portfólio."),
            ("📊 Métricas", "Tabela comparativa de todas as métricas entre HRP+BL, CDI, Ibovespa e IPCA. Use para uma visão consolidada da qualidade do portfólio."),
            ("🔄 Rebalanceamento", "Insira os pesos atuais da sua carteira e veja o drift em relação ao target HRP+BL. Use semestralmente para checar se o portfólio precisa ser rebalanceado. O botão de recálculo HRP sugere novos pesos com dados recentes."),
            ("🌐 Cenários", "Simule o impacto de cenários macro (Selic, IPCA, PIB, câmbio) nos retornos esperados. Use o BL Dinâmico para obter sugestão de novos pesos baseada no cenário. Os resultados alimentam automaticamente o Monte Carlo."),
            ("⚡ Eventos de cauda", "Analise como o portfólio se comportou em cada crise histórica. Use para demonstrar a resiliência do HRP+BL em momentos de stress e comparar com o Ibovespa."),
            ("📦 Ativos individuais", "Retorno de cada ativo separadamente com filtros de período e marcação de eventos. Use para entender a contribuição individual e identificar qual ativo protegeu ou ganhou em cada crise."),
            ("🔬 Análise comparativa", "Distribuição mensal dos retornos, consistência vs CDI e estatísticas descritivas. Use para entender o padrão de comportamento do portfólio além do retorno médio."),
            ("📡 Monitoramento diário", "Retorno do dia/semana/mês/ano em frequência diária. Use mensalmente após atualizar as bases para verificar a performance recente e o drift dos pesos."),
            ("📐 Janelas móveis", "Analisa quantas janelas de 36 ou 48 meses bateram o CDI ou o IPCA. É a 'prova real' da consistência do modelo — mostra que o resultado não depende do momento de entrada."),
            ("🎲 Monte Carlo", "Projeta milhares de trajetórias possíveis para os próximos 5 anos com regimes de Markov. Use em conjunto com a aba Cenários para projeções prospectivas conectadas ao cenário macro."),
        ]
        for nome_aba, desc_aba in abas_info:
            st.markdown(
                f"<div style='margin-bottom:8px;padding:8px 14px;"
                f"background:#f8f7f4;border-radius:6px'>"
                f"<strong style='font-size:13px;color:#1a1a18'>{nome_aba}</strong>"
                f"<p style='font-size:12px;color:#444441;margin:4px 0 0'>{desc_aba}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

    # ── Seção 5: Rotina de atualização ────────────────────────────────────────
    with st.expander("🔄 Rotina de atualização mensal", expanded=False):
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("""
**✅ Automático — não precisa fazer nada**
- CDI mensal — API Banco Central (SGS 4391)
- PTAX (câmbio) — API Banco Central (SGS 3698)
- IPCA — API Banco Central (SGS 433)
- Ibovespa — Yahoo Finance (^BVSP)
- SPY — Yahoo Finance
- TLT — Yahoo Finance
            """)
        with col_r2:
            st.markdown("""
**📂 Manual — uma vez por mês**
1. Baixar os 4 índices da ANBIMA:
   [anbima.com.br/indices](https://www.anbima.com.br/pt_br/informar/indices.htm)
2. Renomear: `IRFM.xls`, `IMA.xls`, `IHFA.xls`, `IDADI.xls`
3. Subir no GitHub:
   [upload/main](https://github.com/santosluisfelipe5502-spec/portfolio-hrp-bl/upload/main)
4. Streamlit atualiza automaticamente em ~30 segundos
            """)
        st.markdown("""
**📅 Semestral — manutenção do modelo**
- Recalcular pesos HRP: aba **Rebalanceamento** → botão "Recalcular pesos HRP com dados recentes"
- Revisar visões BL: aba **Cenários** → ajustar premissas macro e rodar BL Dinâmico
- Atualizar câmbio base no código se o dólar mudar significativamente
        """)

    # ── Rodapé do guia ────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='text-align:center;font-size:12px;color:#888780;padding:8px'>"
        "Dashboard HRP + Black-Litterman · Desenvolvido com Streamlit · "
        "Metodologia: López de Prado (2016) e Black & Litterman (1992) · "
        "Dados: ANBIMA, BCB, Yahoo Finance, Investing.com"
        "</div>",
        unsafe_allow_html=True
    )


# ── Tab 1: Retorno acumulado ──────────────────────────────────────────────────
with tab1:
    # ── Pesos customizados vindos da aba Rebalanceamento (session_state) ──
    def get_custom_weights():
        w = {}
        for cfg in ASSET_CFG:
            key = f"rebal_{cfg['name']}"
            w[cfg["name"]] = st.session_state.get(key, cfg["w"] * 100) / 100
        return w

    custom_w = get_custom_weights()
    total_custom = sum(custom_w.values())
    custom_valid = abs(total_custom - 1.0) < 0.02

    # Calcular portfólio customizado
    custom_ret = sum(custom_w[a["name"]] * series[a["name"]]["valor"].pct_change().dropna().reindex(common_idx).ffill()
                     for a in ASSET_CFG)
    custom_cum = (1 + custom_ret).cumprod() * 100

    # Calcular 1/N
    eq_ret = sum((1/6) * series[a["name"]]["valor"].pct_change().dropna().reindex(common_idx).ffill()
                 for a in ASSET_CFG)
    eq_cum = (1 + eq_ret).cumprod() * 100

    # ── Filtro de período ──
    periodos = {
        "Mês passado": 1,
        "3 meses":     3,
        "12 meses":    12,
        "24 meses":    24,
        "36 meses":    36,
        "5 anos":      60,
        "10 anos":     120,
        "Histórico completo": None,
    }
    pf_col, cb_col1, cb_col2, cb_col3, cb_col4 = st.columns([2,1,1,1,1])
    periodo_sel = pf_col.selectbox("Período", list(periodos.keys()),
                                    index=len(periodos)-1, key="periodo_acum",
                                    label_visibility="collapsed")
    show_cdi    = cb_col1.checkbox("CDI",                  value=True)
    show_ibov   = cb_col2.checkbox("Ibovespa",             value=True)
    show_igual  = cb_col3.checkbox("1/N igual",            value=False)
    show_custom = cb_col4.checkbox("Portfólio customizado",value=True)
    show_ipca6  = st.checkbox("IPCA",          value=True,
                               help=f"Construção: IPCA mensal (BCB) + 6% a.a. | Fonte IPCA: {ipca_src}")

    if show_custom and not custom_valid:
        st.warning("⚠️ Os pesos na aba Rebalanceamento não somam 100%. Ajuste antes de comparar.")

    # ── Aplicar filtro de período ──
    n_meses = periodos[periodo_sel]
    if n_meses is not None:
        corte = common_idx[-1] - pd.DateOffset(months=n_meses)
        idx_filtrado = common_idx[common_idx >= corte]
    else:
        idx_filtrado = common_idx

    def rebase(series_cum, idx):
        """Rebase para % de retorno acumulado (0% = início do período)."""
        s = series_cum[series_cum.index.isin(idx)]
        if len(s) == 0: return s
        return ((s / s.iloc[0]) - 1) * 100   # 0% no início, % de retorno

    port_f   = rebase(port_cum,   idx_filtrado)
    cdi_f    = rebase(cdi_cum,    idx_filtrado)
    ibov_f   = rebase(ibov_cum,   idx_filtrado)
    eq_f     = rebase(eq_cum,     idx_filtrado)
    custom_f = rebase(custom_cum, idx_filtrado) if custom_valid else None

    # ── Gráfico retorno acumulado ──
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_f.index, y=port_f.values.round(2),
        name="HRP+BL original", line=dict(color="#378ADD", width=2.5)))
    if show_cdi:
        fig.add_trace(go.Scatter(x=cdi_f.index, y=cdi_f.values.round(2),
            name="CDI", line=dict(color="#1D9E75", width=1.5, dash="dot")))
    if show_ibov:
        fig.add_trace(go.Scatter(x=ibov_f.index, y=ibov_f.values.round(2),
            name="Ibovespa", line=dict(color="#BA7517", width=1.5, dash="dash")))
    if show_igual:
        fig.add_trace(go.Scatter(x=eq_f.index, y=eq_f.values.round(2),
            name="1/N igual", line=dict(color="#7F77DD", width=1.5, dash="longdash")))
    if show_custom and custom_valid and custom_f is not None:
        fig.add_trace(go.Scatter(x=custom_f.index, y=custom_f.values.round(2),
            name="Portfólio customizado", line=dict(color="#E24B4A", width=2, dash="dashdot")))
    if show_ipca6:
        ipca6_f = rebase(ipca6_cum, idx_filtrado)
        fig.add_trace(go.Scatter(x=ipca6_f.index, y=ipca6_f.values.round(2),
            name="IPCA", line=dict(color="#C4770A", width=1.5, dash="longdashdot")))

    # ── Marcações de eventos de cauda ──
    # Filtrar apenas eventos dentro do período selecionado
    data_ini = idx_filtrado[0]  if len(idx_filtrado) > 0 else port_cum.index[0]
    data_fim = idx_filtrado[-1] if len(idx_filtrado) > 0 else port_cum.index[-1]

    show_events = st.checkbox("Marcar eventos de cauda no gráfico", value=True, key="ev_acum")
    if show_events:
        for ev in TAIL_EVENTS:
            try:
                ev_start = pd.Timestamp(ev["start"])
                ev_end   = pd.Timestamp(ev["end"])
                # Só mostrar se o evento está dentro do período selecionado
                if ev_start >= data_ini and ev_start <= data_fim:
                    fig.add_vrect(
                        x0=ev_start, x1=min(ev_end, data_fim),
                        fillcolor=ev["color"], opacity=0.12,
                        layer="below", line_width=0,
                    )
                    fig.add_annotation(
                        x=ev_start, y=1.0, yref="paper",
                        text=ev["name"], textangle=-90,
                        showarrow=False,
                        font=dict(size=9, color=ev["color"]),
                        xanchor="right", yanchor="top",
                        bgcolor="rgba(248,247,244,0.7)",
                    )
            except Exception:
                pass

    fig.update_layout(
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        height=420, hovermode="x unified",
        font=dict(color="#1a1a18"),
        margin=dict(l=0,r=40,t=8,b=0),
        # ── Limitar eixo X ao período selecionado ──
        xaxis_range=[data_ini, data_fim],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#1a1a18")),
        yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0",
                   tickfont=dict(color="#444441", size=11), color="#1a1a18",
                   hoverformat=".1f"),
        xaxis=dict(gridcolor="#e8e6e0", tickfont=dict(color="#444441", size=11), color="#1a1a18"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Nota sobre portfólio customizado ──
    if show_custom and custom_valid:
        # Acumulado no período selecionado
        port_f_vals   = port_f if len(port_f) > 0 else port_cum
        custom_f_vals = custom_f if custom_f is not None and len(custom_f) > 0 else custom_cum
        acum_custom = custom_f_vals.iloc[-1] if len(custom_f_vals) > 0 else 0
        acum_hrpbl  = port_f_vals.iloc[-1]   if len(port_f_vals) > 0 else 0
        diff = acum_custom - acum_hrpbl
        if diff >= 0:
            st.success(f"📊 Com os pesos customizados, o retorno acumulado seria **+{acum_custom:.1f}%** — {diff:+.1f}% vs HRP+BL original ({acum_hrpbl:.1f}%)")
        else:
            st.info(f"📊 Com os pesos customizados, o retorno acumulado seria **+{acum_custom:.1f}%** — {diff:+.1f}% vs HRP+BL original ({acum_hrpbl:.1f}%)")

    # ── Gráfico retorno anual — filtrado pelo período selecionado ──
    st.markdown("<div class='section-title'>Retorno anual</div>", unsafe_allow_html=True)
    # Usar série filtrada para retorno anual
    port_cum_f  = port_cum[port_cum.index >= data_ini]
    cdi_cum_f   = cdi_cum[cdi_cum.index   >= data_ini]
    ibov_cum_f  = ibov_cum[ibov_cum.index >= data_ini]
    eq_cum_f    = eq_cum[eq_cum.index     >= data_ini]
    ipca6_cum_f = ipca6_cum[ipca6_cum.index >= data_ini]

    ann_port   = port_cum_f.resample("YE").last().pct_change().dropna() * 100
    ann_cdi    = cdi_cum_f.resample("YE").last().pct_change().dropna() * 100
    ann_ibov   = ibov_cum_f.resample("YE").last().pct_change().dropna() * 100
    ann_igual  = eq_cum_f.resample("YE").last().pct_change().dropna() * 100
    ann_custom = custom_cum.resample("YE").last().pct_change().dropna() * 100
    ann_ipca6  = ipca6_cum_f.resample("YE").last().pct_change().dropna() * 100
    years = sorted(set(ann_port.index.year) & set(ann_cdi.index.year) & set(ann_ibov.index.year))

    def get_ann(series_ann, yr):
        vals = series_ann[series_ann.index.year == yr]
        return round(vals.values[0], 1) if len(vals) else None

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=[str(y) for y in years],
        y=[get_ann(ann_port, y) for y in years],
        name="HRP+BL original", marker_color="#378ADD"))
    if show_cdi:
        fig2.add_trace(go.Bar(x=[str(y) for y in years],
            y=[get_ann(ann_cdi, y) for y in years],
            name="CDI", marker_color="rgba(29,158,117,0.5)"))
    if show_ibov:
        fig2.add_trace(go.Bar(x=[str(y) for y in years],
            y=[get_ann(ann_ibov, y) for y in years],
            name="Ibovespa", marker_color="rgba(186,117,23,0.4)"))
    if show_igual:
        fig2.add_trace(go.Bar(x=[str(y) for y in years],
            y=[get_ann(ann_igual, y) for y in years],
            name="1/N igual", marker_color="rgba(127,119,221,0.5)"))
    if show_custom and custom_valid:
        fig2.add_trace(go.Bar(x=[str(y) for y in years],
            y=[get_ann(ann_custom, y) for y in years],
            name="Portfólio customizado", marker_color="rgba(226,75,74,0.7)"))
    if show_ipca6:
        fig2.add_trace(go.Bar(x=[str(y) for y in years],
            y=[get_ann(ann_ipca6, y) for y in years],
            name="IPCA", marker_color="rgba(196,119,10,0.6)"))

    fig2.update_layout(
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        font=dict(color="#1a1a18"),
        margin=dict(l=0,r=0,t=8,b=0),
        height=300, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#1a1a18")),
        xaxis=dict(gridcolor="#e8e6e0", tickfont=dict(color="#444441", size=11), color="#1a1a18"),
        yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0", tickfont=dict(color="#444441", size=11), color="#1a1a18"),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2: Drawdown ───────────────────────────────────────────────────────────
with tab2:
    # ── Filtro de período ──
    periodos_dd = {
        "12 meses": 12, "24 meses": 24, "36 meses": 36,
        "5 anos": 60, "Histórico completo": None,
    }
    col_dd_per, col_dd_chk = st.columns([2, 3])
    periodo_dd = col_dd_per.selectbox("Período", list(periodos_dd.keys()),
                                       index=4, key="periodo_dd")
    show_cust_dd = col_dd_chk.checkbox("Mostrar portfólio customizado",
                                        value=True, key="show_cust_dd")

    # Aplicar filtro
    n_dd_m = periodos_dd[periodo_dd]
    if n_dd_m:
        corte_dd = common_idx[-1] - pd.DateOffset(months=n_dd_m)
        idx_dd   = common_idx[common_idx >= corte_dd]
    else:
        idx_dd = common_idx

    # Recalcular drawdown no período filtrado
    def calc_dd_filtrado(ret_series, idx):
        r = ret_series.reindex(idx).ffill().dropna()
        cum = (1 + r).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax() * 100
        return dd

    dd_port_f = calc_dd_filtrado(port_ret, idx_dd)
    dd_ibov_f = calc_dd_filtrado(ibov_ret, idx_dd)

    # Drawdown do customizado
    custom_w_dd = {cfg["name"]: st.session_state.get(f"rebal_{cfg['name']}",
                   cfg["w"]*100)/100 for cfg in ASSET_CFG}
    total_dd = sum(custom_w_dd.values())
    custom_valid_dd = abs(total_dd - 1.0) < 0.02
    if custom_valid_dd:
        c_ret_dd = sum(
            custom_w_dd[a["name"]] * series[a["name"]]["valor"]
            .pct_change().dropna().reindex(common_idx).ffill()
            for a in ASSET_CFG
        )
        dd_cust_f = calc_dd_filtrado(c_ret_dd, idx_dd)

    # ── Gráfico ──
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_port_f.index, y=dd_port_f.values.round(2),
        name="HRP+BL", fill="tozeroy",
        line=dict(color="#378ADD", width=1.5),
        fillcolor="rgba(55,138,221,0.15)",
        hovertemplate="%{x|%b/%Y}<br>DD: %{y:.2f}%<extra>HRP+BL</extra>"))
    fig_dd.add_trace(go.Scatter(
        x=dd_ibov_f.index, y=dd_ibov_f.values.round(2),
        name="Ibovespa", fill="tozeroy",
        line=dict(color="#E24B4A", width=1.2, dash="dot"),
        fillcolor="rgba(226,75,74,0.08)",
        hovertemplate="%{x|%b/%Y}<br>DD: %{y:.2f}%<extra>Ibovespa</extra>"))
    if show_cust_dd and custom_valid_dd:
        fig_dd.add_trace(go.Scatter(
            x=dd_cust_f.index, y=dd_cust_f.values.round(2),
            name="Customizado", fill="tozeroy",
            line=dict(color="#E24B4A", width=1.5, dash="dashdot"),
            fillcolor="rgba(226,75,74,0.05)",
            hovertemplate="%{x|%b/%Y}<br>DD: %{y:.2f}%<extra>Customizado</extra>"))
    elif show_cust_dd and not custom_valid_dd:
        st.caption("⚠️ Configure os pesos na aba Rebalanceamento para ver o drawdown customizado.")

    fig_dd.update_layout(
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        font=dict(color="#1a1a18"),
        margin=dict(l=0,r=0,t=8,b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(color="#1a1a18")),
        xaxis=dict(gridcolor="#e8e6e0", tickfont=dict(color="#444441", size=11),
                   color="#1a1a18", range=[idx_dd[0], idx_dd[-1]]),
        yaxis=dict(gridcolor="#e8e6e0", tickfont=dict(color="#444441", size=11),
                   color="#1a1a18", ticksuffix="%"),
        height=320,
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Métricas comparativas ──
    dd_max_port = dd_port_f.min()
    dd_max_ibov = dd_ibov_f.min()
    n_cols = 4 if (show_cust_dd and custom_valid_dd) else 3
    cols_dd = st.columns(n_cols)
    cols_dd[0].metric("Max DD — HRP+BL",   f"{dd_max_port:.2f}%")
    cols_dd[1].metric("Max DD — Ibovespa", f"{dd_max_ibov:.2f}%",
                       delta=f"{dd_max_port-dd_max_ibov:.1f}%")
    cols_dd[2].metric("Proteção HRP+BL",   f"{dd_max_port-dd_max_ibov:.1f}%",
                       delta="vs Ibovespa")
    if show_cust_dd and custom_valid_dd:
        dd_max_cust = dd_cust_f.min()
        diff_cust = dd_max_cust - dd_max_port
        cols_dd[3].metric("Max DD — Customizado", f"{dd_max_cust:.2f}%",
                           delta=f"{diff_cust:+.1f}% vs HRP+BL")

# ── Tab 3: Métricas ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-title'>Comparativo de métricas</div>", unsafe_allow_html=True)

    rows = {
        "Retorno a.a.": (
            f"{m_port['ann_ret']*100:.2f}%", f"{rf_ann*100:.2f}%",
            f"{m_ibov['ann_ret']*100:.2f}%", f"{ann_ret_ipca6*100:.2f}%"),
        "Volatilidade a.a.": (
            f"{m_port['ann_vol']*100:.2f}%", "~0%",
            f"{m_ibov['ann_vol']*100:.2f}%", "baixa"),
        "Sharpe (rf=CDI)": (
            f"{m_port['sharpe']:.3f}", "—",
            f"{m_ibov['sharpe']:.3f}", "—"),
        "Sortino": (
            f"{m_port['sortino']:.3f}" if m_port["sortino"] else "—", "—",
            f"{m_ibov['sortino']:.3f}" if m_ibov["sortino"] else "—", "—"),
        "Calmar ratio": (
            f"{m_port['calmar']:.3f}", "—",
            f"{m_ibov['calmar']:.3f}", "—"),
        "Max drawdown": (
            f"{m_port['max_dd']*100:.2f}%", "0%",
            f"{m_ibov['max_dd']*100:.2f}%", "~0%"),
        "VaR 95% mensal": (
            f"{m_port['var95']*100:.2f}%", "positivo",
            f"{m_ibov['var95']*100:.2f}%", "positivo"),
        "Acumulado": (
            f"+{acum_port:.1f}%", f"+{acum_cdi:.1f}%",
            f"+{acum_ibov:.1f}%", f"+{acum_ipca6:.1f}%"),
    }
    df_metrics = pd.DataFrame(rows, index=["HRP+BL","CDI","Ibovespa","IPCA"]).T
    df_metrics.index.name = "Métrica"
    st.dataframe(df_metrics, use_container_width=True)

    st.caption(
        f"IPCA puro: IPCA mensal via BCB (fonte: {ipca_src}) + 6% a.a. | "
        "Representa o retorno de uma NTN-B com prêmio real de 6% sobre a inflação."
    )

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
    st.markdown("Insira os pesos atuais da carteira e veja o drift em relação ao target HRP+BL.")

    # ── Seção de recálculo semestral ──────────────────────────────────────────
    with st.expander("🔄 Recalcular pesos HRP com dados recentes", expanded=False):
        st.markdown(
            "Roda o algoritmo HRP (López de Prado, 2016) com os últimos **N meses** "
            "de retornos reais dos ativos e sugere novos pesos. "
            "Use semestralmente para manter o modelo atualizado."
        )
        col_hrp1, col_hrp2 = st.columns(2)
        janela_hrp = col_hrp1.selectbox(
            "Janela histórica", [24, 36, 48, 60],
            index=1,
            format_func=lambda x: f"{x} meses ({x//12} anos)",
            key="janela_hrp"
        )
        rodar_hrp = col_hrp2.button("▶ Calcular novos pesos HRP", key="btn_hrp", type="primary")

        if rodar_hrp:
            with st.spinner(f"Calculando HRP com os últimos {janela_hrp} meses..."):
                try:
                    # Montar DataFrame de retornos dos últimos N meses
                    idx_hrp = common_idx[-janela_hrp:]
                    ret_dict = {}
                    for cfg in ASSET_CFG:
                        r = series[cfg["name"]]["valor"].pct_change().dropna()
                        r = r.reindex(idx_hrp).ffill().dropna()
                        if len(r) >= janela_hrp * 0.7:
                            ret_dict[cfg["name"]] = r
                    ret_df = pd.DataFrame(ret_dict).dropna()

                    if len(ret_df.columns) < 3:
                        st.warning("⚠️ Dados insuficientes para recalcular. Verifique as séries carregadas.")
                    else:
                        novos_pesos = compute_hrp(ret_df)

                        # Comparativo visual
                        st.markdown("#### Novos pesos sugeridos vs pesos atuais")
                        rows_hrp = []
                        for cfg in ASSET_CFG:
                            nome    = cfg["name"]
                            w_atual = cfg["w"] * 100
                            w_novo  = novos_pesos.get(nome, 0) * 100
                            delta   = w_novo - w_atual
                            rows_hrp.append({
                                "Ativo":        nome,
                                "Cluster":      cfg["cluster"],
                                "Peso atual":   f"{w_atual:.1f}%",
                                "Novo peso HRP":f"{w_novo:.1f}%",
                                "Δ":            f"{delta:+.1f}%",
                                "Ação":         "⬆️ Aumentar" if delta > 1 else
                                                "⬇️ Reduzir"  if delta < -1 else
                                                "✅ Manter",
                            })
                        df_hrp = pd.DataFrame(rows_hrp).set_index("Ativo")
                        st.dataframe(df_hrp, use_container_width=True)

                        # Gráfico comparativo
                        fig_hrp = go.Figure()
                        nomes_hrp = [cfg["name"] for cfg in ASSET_CFG]
                        fig_hrp.add_trace(go.Bar(
                            x=nomes_hrp,
                            y=[cfg["w"]*100 for cfg in ASSET_CFG],
                            name="Pesos atuais",
                            marker_color="rgba(55,138,221,0.7)",
                        ))
                        fig_hrp.add_trace(go.Bar(
                            x=nomes_hrp,
                            y=[novos_pesos.get(n,0)*100 for n in nomes_hrp],
                            name=f"Novos pesos HRP ({janela_hrp}m)",
                            marker_color="rgba(29,158,117,0.7)",
                        ))
                        fig_hrp.update_layout(
                            plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
                            height=280, barmode="group",
                            font=dict(color="#1a1a18"),
                            margin=dict(l=0,r=0,t=8,b=0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                        xanchor="left", x=0,
                                        font=dict(color="#1a1a18")),
                            xaxis=dict(gridcolor="#e8e6e0",
                                       tickfont=dict(color="#444441",size=11),
                                       color="#1a1a18"),
                            yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0",
                                       tickfont=dict(color="#444441",size=11),
                                       color="#1a1a18"),
                        )
                        st.plotly_chart(fig_hrp, use_container_width=True)

                        st.info(
                            f"💡 Para aplicar os novos pesos: ajuste os campos abaixo "
                            f"com os valores da coluna 'Novo peso HRP' e atualize o código "
                            f"com os novos pesos em ASSET_CFG semestralmente."
                        )

                        # Mostrar pesos prontos para copiar
                        pesos_str = " | ".join([
                            f"{cfg['name']} {novos_pesos.get(cfg['name'],0)*100:.1f}%"
                            for cfg in ASSET_CFG
                        ])
                        st.code(pesos_str, language=None)

                except Exception as e:
                    st.error(f"Erro no cálculo HRP: {e}")

    st.divider()

    st.markdown(
        "<div style='font-size:12px;color:#888780;margin-bottom:.75rem'>"
        "💡 Digite o percentual de cada ativo (ex: 26.8). Os pesos devem somar 100%."
        "</div>", unsafe_allow_html=True
    )

    atual = {}
    col_s1, col_s2 = st.columns(2)
    for i, cfg in enumerate(ASSET_CFG):
        col = col_s1 if i < 3 else col_s2
        with col:
            atual[cfg["name"]] = col.number_input(
                f"{cfg['name']} (%)",
                min_value=0.0, max_value=100.0,
                value=round(cfg["w"]*100, 1), step=0.1,
                format="%.1f",
                key=f"rebal_{cfg['name']}",
                help=f"Target HRP+BL: {cfg['w']*100:.1f}%"
            )

    total_atual = sum(atual.values())
    # Barra visual de progresso do total
    prog_color = "#1D9E75" if abs(total_atual-100) < 0.5 else "#E24B4A"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;margin:8px 0'>"
        f"<span style='font-size:13px;color:#888780'>Total alocado:</span>"
        f"<strong style='font-size:18px;color:{prog_color}'>{total_atual:.1f}%</strong>"
        f"<span style='font-size:12px;color:{prog_color}'>"
        f"{'✅ ok' if abs(total_atual-100)<0.5 else f'⚠️ {total_atual-100:+.1f}% vs 100%'}"
        f"</span></div>", unsafe_allow_html=True
    )
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
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        font=dict(color="#1a1a18"),
        margin=dict(l=0,r=0,t=8,b=0),
        legend=dict(font=dict(color="#1a1a18"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#e8e6e0", tickfont=dict(color="#444441", size=11), color="#1a1a18"),
        yaxis=dict(gridcolor="#e8e6e0", tickfont=dict(color="#444441", size=11), color="#1a1a18"), 
        height=260,
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

    # ── Período da amostra ──
    periodo_str = f"{common_idx[0].strftime('%d/%m/%Y')} → {common_idx[-1].strftime('%d/%m/%Y')} · {len(common_idx)} meses"
    st.markdown(
        f"<div style='display:inline-flex;align-items:center;gap:6px;font-size:12px;"
        f"background:#E6F1FB;color:#185FA5;padding:3px 12px;border-radius:12px;margin-bottom:1rem'>"
        f"📅 Amostra histórica: <strong>{periodo_str}</strong></div>",
        unsafe_allow_html=True
    )

    presets = {
        "Base (Selic 14.5%)": dict(selic=14.50, ipca=5.5, pib=2.0, fx=4.99),
        "Juro terminal 2026": dict(selic=13.75, ipca=4.5, pib=2.5, fx=5.5),
        "Juro alto":          dict(selic=16.5,  ipca=6.5, pib=0.5, fx=6.2),
        "Recessão":           dict(selic=10.0,  ipca=3.0, pib=-2.0,fx=6.5),
        "Rali de risco":      dict(selic=11.0,  ipca=4.0, pib=3.5, fx=4.8),
        "Customizado":        dict(selic=14.50, ipca=5.5, pib=2.0, fx=4.99),
    }
    preset = st.selectbox("Cenário pré-definido", list(presets.keys()))
    p = presets[preset]

    # ── Inputs numéricos ──
    c1, c2, c3, c4 = st.columns(4)
    selic = c1.number_input("Selic (% a.a.)", min_value=2.0,  max_value=30.0,
                             value=float(p["selic"]), step=0.25, format="%.2f")
    ipca  = c2.number_input("IPCA (% a.a.)",  min_value=-2.0, max_value=30.0,
                             value=float(p["ipca"]),  step=0.25, format="%.2f")
    pib   = c3.number_input("PIB (% a.a.)",   min_value=-10.0,max_value=15.0,
                             value=float(p["pib"]),   step=0.25, format="%.2f")
    fx    = c4.number_input("USD/BRL",         min_value=2.0,  max_value=15.0,
                             value=float(p["fx"]),    step=0.10, format="%.2f")

    # ── Correlações cruzadas implícitas ──────────────────────────────────────
    # As variáveis macro não são independentes. Aplicamos ajustes automáticos
    # baseados nas relações históricas observadas na economia brasileira.
    use_correlacoes = st.checkbox(
        "Aplicar correlações cruzadas automáticas (câmbio ↔ inflação ↔ juros ↔ PIB)",
        value=True,
        help="Quando ativo, ajusta automaticamente as variáveis inter-relacionadas."
    )

    selic_adj, ipca_adj, pib_adj, fx_adj = selic, ipca, pib, fx

    if use_correlacoes:
        ajustes = []

        # NOTA: A Selic inserida pelo usuário é sempre mantida como âncora.
        # As correlações ajustam apenas IPCA, câmbio e PIB — variáveis consequentes.
        # Isso evita que o modelo sobrescreva o input do usuário com estimativas do Copom.

        # 1. Recessão puxa câmbio para cima (Selic NÃO é ajustada — usuário é a âncora)
        if pib < 0:
            delta_fx_recessao = abs(pib) * 0.30
            fx_adj = min(12.0, fx + delta_fx_recessao)
            ajustes.append(
                f"📉 Recessão (PIB {pib:.1f}%): "
                f"câmbio ajustado +{delta_fx_recessao:.2f} → {fx_adj:.2f}"
            )

        # 2. Câmbio alto puxa inflação (pass-through histórico BR ≈ 0.10-0.15)
        delta_fx_base = fx_adj - 4.99  # base = PTAX mai/2026
        if abs(delta_fx_base) > 0.3:
            passthrough = 0.12
            ipca_adj = min(20.0, ipca + delta_fx_base * passthrough * 10)
            ajustes.append(
                f"💱 Pass-through cambial ({fx_adj:.2f} vs base 4.99): "
                f"IPCA ajustado +{delta_fx_base*passthrough*10:.2f}% → {ipca_adj:.2f}%"
            )

        # 3. Juro alto (inserido pelo usuário) deprime PIB
        # Selic é âncora — usamos o valor original do usuário aqui
        if selic > 13.0:
            depressao_pib = (selic - 13.0) * 0.12
            pib_adj = max(-6.0, pib_adj - depressao_pib)
            ajustes.append(
                f"📊 Juro alto ({selic:.1f}%): PIB ajustado "
                f"-{depressao_pib:.2f}% → {pib_adj:.2f}%"
            )

        # 4. Rali de risco — câmbio cai quando PIB alto e juro baixo
        if pib > 3.0 and selic < 11.0:
            fx_adj = max(4.0, fx_adj - (pib - 3.0) * 0.15)
            ajustes.append(
                f"🚀 Rali de risco (PIB {pib:.1f}%, Selic {selic:.1f}%): "
                f"câmbio ajustado → {fx_adj:.2f}"
            )

        if ajustes:
            with st.expander("Ver ajustes de correlação aplicados", expanded=False):
                for aj in ajustes:
                    st.markdown(f"<div style='font-size:12px;color:#444441;padding:4px 0'>{aj}</div>",
                                unsafe_allow_html=True)
            # Mostrar valores ajustados
            ca1,ca2,ca3,ca4 = st.columns(4)
            def delta_badge(orig, adj, unit=""):
                d = adj - orig
                if abs(d) < 0.01: return ""
                col = "#0F6E56" if d < 0 else "#854F0B"
                return (f"<span style='font-size:10px;color:{col};margin-left:4px'>"
                        f"ajustado: {adj:.2f}{unit}</span>")
            ca1.markdown(f"Selic: <strong>{selic:.2f}%</strong>"
                         f"<span style='font-size:10px;color:#888780;margin-left:4px'>"
                         f"âncora do usuário</span>", unsafe_allow_html=True)
            ca2.markdown(f"IPCA efetivo: <strong>{ipca_adj:.2f}%</strong>"
                         f"{delta_badge(ipca, ipca_adj, '%')}", unsafe_allow_html=True)
            ca3.markdown(f"PIB efetivo: <strong>{pib_adj:.2f}%</strong>"
                         f"{delta_badge(pib, pib_adj, '%')}", unsafe_allow_html=True)
            ca4.markdown(f"USD/BRL efetivo: <strong>{fx_adj:.2f}</strong>"
                         f"{delta_badge(fx, fx_adj)}", unsafe_allow_html=True)

    # Usar variáveis ajustadas nas fórmulas
    # Selic mantida como inserida pelo usuário (âncora)
    # Apenas IPCA, PIB e câmbio recebem ajustes das correlações
    selic, ipca, pib, fx = selic, ipca_adj, pib_adj, fx_adj

    # ── Retornos esperados por ativo ──────────────────────────────────────────
    # Todas as fórmulas são estimativas baseadas em relações históricas.
    # Servem para comparar cenários entre si, não como previsões absolutas.

    # CDI implícito no cenário (Selic menos haircut de liquidez diária ~0.1%)
    cdi_sc = selic - 0.1

    # ── IRF-M (títulos públicos prefixados — LTN e NTN-F) ──
    # Duration variável de mercado ≈ 2-3 anos (menor que IDKA Pré 5A).
    # Mais representativo do mercado real de renda fixa prefixada.
    # Em alta de juros, preço cai pelo MTM, mas duration menor = menos sensível.
    duration_irfm = 2.5
    delta_selic   = selic - 14.50   # variação vs cenário base (Selic atual 14.5%)
    mtm_irfm      = -duration_irfm * delta_selic * 0.6
    ret_pre       = max(-12, min(22, selic * 0.84 + mtm_irfm))

    # ── IMA-Geral (inflação — NTN-B) ──
    # NTN-B paga IPCA + taxa real. Com Selic alta, a taxa real sobe e o preço cai (duration).
    # Duration média IMA ≈ 6-7 anos → mais sensível que o pré.
    duration_ima = 6.5
    taxa_real    = selic - ipca              # juro real implícito
    mtm_ima      = -duration_ima * delta_selic * 0.5
    ret_ima      = max(-20, min(30, ipca + max(3.5, taxa_real * 0.4) + mtm_ima))

    # ── IDA-DI (debêntures atreladas ao CDI — crédito privado pós-fixado puro) ──
    # Mais puro que o IDA-DI: só debêntures CDI+spread, sem IPCA+ ou prefixados.
    # Spread de crédito varia com ciclo econômico mas duration muito baixa.
    # Praticamente sem risco de MTM — a sensibilidade é quase toda no spread.
    spread_di_base  = 1.5   # spread histórico médio das debs CDI+ investment grade
    spread_di_ciclo = max(-0.8, min(2.0,
        (-pib * 0.20)                       # recessão abre spread
        + (max(0, selic - 14) * 0.10)       # juro alto = leve stress de crédito
        + (max(0, fx - 6.0) * 0.20)         # câmbio alto = fuga de crédito
    ))
    ret_ida_geral = max(-5, min(20, cdi_sc + spread_di_base + spread_di_ciclo))

    # ── IHFA (multimercados) ──
    # Multimercados são estratégias ativas — têm beta ao CDI + alpha direcional.
    # A dispersão entre gestores é enorme: P10 ≈ CDI-1%, P90 ≈ CDI+6%.
    # Modelamos como distribuição com média, mínimo e máximo esperados.
    alpha_central = 2.0    # alpha médio histórico (mediana dos fundos)
    alpha_p10     = -1.0   # 10% piores gestores ficam abaixo do CDI
    alpha_p90     = 5.5    # 10% melhores gestores entregam CDI+5.5%
    beta_pib      = max(-1.5, min(1.5, pib * 0.2))
    # Em recessão, alpha médio cai (correlações inesperadas)
    penalidade_recessao = min(0, pib * 0.3) if pib < 0 else 0
    # Em juro muito alto, gestores direcionais ganham mais
    bonus_juro_alto = max(0, (selic - 13) * 0.2) if selic > 13 else 0

    ret_ihfa      = max(-5, min(30,
        cdi_sc * 0.65 + alpha_central + beta_pib + penalidade_recessao + bonus_juro_alto))
    ret_ihfa_min  = max(-10, min(25,
        cdi_sc * 0.60 + alpha_p10 + beta_pib + penalidade_recessao * 1.5))
    ret_ihfa_max  = max(0, min(40,
        cdi_sc * 0.70 + alpha_p90 + beta_pib + bonus_juro_alto * 1.5))

    # ── Ibovespa (ações brasileiras) ──
    # Equities são sensíveis ao crescimento (earnings), juro (desconto) e câmbio (exportadoras).
    # Prêmio de risco histórico do Ibov ≈ 5-6% sobre o CDI em ciclos normais.
    premio_risco  = 5.5
    sens_pib      = pib * 2.8                          # elasticidade earnings-PIB
    custo_capital = max(0, (selic - 10) * 1.2)        # juro alto comprime múltiplos
    efeit_cambio  = (fx - 4.99) * 1.5                 # câmbio vs PTAX atual 4.99
    ret_ibov      = max(-40, min(50,
        cdi_sc + premio_risco + sens_pib - custo_capital + efeit_cambio
    ))

    # ── Internacional (SPY + TLT em BRL) ──
    # SPY: correlação com crescimento global e earnings americanos
    # TLT: correlação inversa com juros EUA (quando Fed sobe, TLT cai)
    # Câmbio: USD/BRL amplifica (depreciation do real = ganho em BRL)
    ret_spy_usd  = max(-25, min(35, 9.0 + pib * 1.2 - max(0, (selic-12)*0.3)))
    ret_tlt_usd  = max(-20, min(20, 4.5 - max(0, (selic-10)*0.8)))
    ret_intl_usd = 0.40 * ret_spy_usd + 0.60 * ret_tlt_usd
    # Efeito cambial em BRL
    delta_fx     = (fx - 4.99) / 4.99 * 100  # variação % do câmbio vs PTAX atual 4.99
    ret_intl_brl = (1 + ret_intl_usd/100) * (1 + delta_fx/100) - 1
    ret_intl      = max(-30, min(50, ret_intl_brl * 100))

    asset_rets = {
        "IRF-M":     round(ret_pre,       2),
        "IMA":       round(ret_ima,       2),
        "IHFA":      round(ret_ihfa,      2),
        "IDA-DI":    round(ret_ida_geral, 2),
        "Ibovespa":  round(ret_ibov,      2),
        "Internac.": round(ret_intl,      2),
    }

    # ── Salvar retornos no session_state para uso no Monte Carlo ──────────────
    st.session_state["mc_asset_rets"]   = asset_rets
    st.session_state["mc_cenario_nome"] = preset
    st.session_state["mc_selic"]        = selic

    # ── Pesos customizados ──
    custom_w_sc = {}
    for cfg in ASSET_CFG:
        key = f"rebal_{cfg['name']}"
        custom_w_sc[cfg["name"]] = st.session_state.get(key, cfg["w"] * 100) / 100
    total_cw_sc = sum(custom_w_sc.values())
    custom_valid_sc = abs(total_cw_sc - 1.0) < 0.02

    # ── Retornos totais ponderados ──
    port_ret_sc    = sum(WEIGHTS[k] * v for k, v in asset_rets.items())
    custom_ret_sc  = sum(custom_w_sc[k] * v for k, v in asset_rets.items()) if custom_valid_sc else None
    equal_ret_sc   = sum((1/6) * v for v in asset_rets.values())
    premium_sc     = port_ret_sc - selic
    sharpe_sc      = premium_sc / (m_port["ann_vol"] * 100)

    # ── KPIs comparativos ──
    st.divider()
    st.markdown("<div class='section-title'>retorno total ponderado — comparativo de carteiras</div>", unsafe_allow_html=True)
    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1.metric("HRP+BL",     f"{port_ret_sc:.2f}%",   delta=f"{port_ret_sc-selic:+.2f}% vs CDI")
    kc2.metric("CDI (Selic)", f"{selic:.2f}%",          delta="benchmark")
    kc3.metric("Ibovespa",   f"{asset_rets['Ibovespa']:.2f}%", delta=f"{asset_rets['Ibovespa']-selic:+.2f}% vs CDI")
    kc4.metric("1/N igual",  f"{equal_ret_sc:.2f}%",   delta=f"{equal_ret_sc-selic:+.2f}% vs CDI")
    if custom_valid_sc:
        kc5.metric("Customizado", f"{custom_ret_sc:.2f}%", delta=f"{custom_ret_sc-selic:+.2f}% vs CDI")
    else:
        kc5.metric("Customizado", "—", delta="ajuste os pesos")

    st.markdown("<div class='section-title'>sharpe e vol estimados</div>", unsafe_allow_html=True)
    ks1, ks2, ks3, ks4 = st.columns(4)
    ks1.metric("Sharpe HRP+BL",    f"{sharpe_sc:.3f}")
    ks2.metric("Vol. HRP+BL",      f"{m_port['ann_vol']*100:.1f}%")
    if custom_valid_sc:
        sharpe_cust = (custom_ret_sc - selic) / (m_port["ann_vol"] * 100)
        ks3.metric("Sharpe Customizado", f"{sharpe_cust:.3f}")
    ks4.metric("Vol. Ibovespa",    f"{m_ibov['ann_vol']*100:.1f}%")

    # ── Gráfico duplo: HRP+BL vs Customizado por ativo ──
    st.divider()
    st.markdown("<div class='section-title'>retorno esperado por ativo — HRP+BL vs customizado</div>", unsafe_allow_html=True)

    ativos    = list(asset_rets.keys())
    rets_vals = [round(v, 2) for v in asset_rets.values()]
    colors    = [COLORS[k] for k in ativos]

    # Range de confiança do IHFA
    ihfa_idx  = ativos.index("IHFA")

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Bar(
        x=ativos, y=rets_vals,
        name="Retorno esperado (central)",
        marker_color=colors,
        showlegend=True,
    ))
    # Faixa de confiança P10-P90 para o IHFA
    fig_sc.add_trace(go.Scatter(
        x=["IHFA","IHFA"],
        y=[round(ret_ihfa_min,2), round(ret_ihfa_max,2)],
        mode="lines+markers",
        name="IHFA — range P10/P90",
        line=dict(color="#378ADD", width=3),
        marker=dict(symbol="line-ew", size=14, color="#378ADD",
                    line=dict(width=3, color="#378ADD")),
        showlegend=True,
    ))
    fig_sc.add_annotation(
        x="IHFA", y=ret_ihfa_max + 0.5,
        text=f"P90: {ret_ihfa_max:.1f}%",
        showarrow=False, font=dict(size=10, color="#378ADD"),
        yanchor="bottom",
    )
    fig_sc.add_annotation(
        x="IHFA", y=ret_ihfa_min - 0.5,
        text=f"P10: {ret_ihfa_min:.1f}%",
        showarrow=False, font=dict(size=10, color="#E24B4A"),
        yanchor="top",
    )

    # Linha de peso HRP+BL
    fig_sc.add_trace(go.Scatter(
        x=ativos,
        y=[round(WEIGHTS[k]*100, 1) for k in ativos],
        name="Peso HRP+BL (%)",
        mode="lines+markers",
        line=dict(color="#378ADD", width=2, dash="dot"),
        marker=dict(size=8),
        yaxis="y2",
    ))

    if custom_valid_sc:
        fig_sc.add_trace(go.Scatter(
            x=ativos,
            y=[round(custom_w_sc[k]*100, 1) for k in ativos],
            name="Peso Customizado (%)",
            mode="lines+markers",
            line=dict(color="#E24B4A", width=2, dash="dashdot"),
            marker=dict(size=8),
            yaxis="y2",
        ))

    fig_sc.add_hline(y=selic, line_dash="dot", line_color="#1D9E75",
                     annotation_text=f"Selic {selic:.2f}%",
                     annotation_font_color="#1D9E75",
                     annotation_position="right")

    fig_sc.update_layout(
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        font=dict(color="#1a1a18"),
        margin=dict(l=0, r=60, t=8, b=0),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#1a1a18")),
        xaxis=dict(gridcolor="#e8e6e0", tickfont=dict(color="#444441", size=11), color="#1a1a18"),
        yaxis=dict(title="Retorno esperado (%)", ticksuffix="%",
                   gridcolor="#e8e6e0", tickfont=dict(color="#444441", size=11), color="#1a1a18"),
        yaxis2=dict(title="Peso na carteira (%)", ticksuffix="%",
                    overlaying="y", side="right",
                    tickfont=dict(color="#444441", size=11), color="#1a1a18",
                    showgrid=False, range=[0, 50]),
        barmode="group",
    )
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption("Barras = retorno esperado de cada ativo no cenário. Linhas = peso de cada carteira (eixo direito).")

    # ── Alerta final ──
    st.divider()
    if premium_sc > 0:
        st.success(f"✅ HRP+BL supera o CDI em **{premium_sc:.2f}%** a.a. neste cenário.")
    else:
        st.warning(f"⚠️ CDI supera o HRP+BL em **{abs(premium_sc):.2f}%** a.a. Considere revisar as visões no Black-Litterman.")
    if custom_valid_sc and custom_ret_sc is not None:
        diff_cust = custom_ret_sc - port_ret_sc
        if diff_cust >= 0:
            st.info(f"📊 Portfólio customizado supera o HRP+BL em **{diff_cust:+.2f}%** a.a. neste cenário.")
        else:
            st.info(f"📊 Portfólio customizado fica **{diff_cust:.2f}%** abaixo do HRP+BL neste cenário.")

    # ── Tabela comparativa detalhada HRP+BL vs Customizado no cenário ──────────
    if custom_valid_sc and custom_ret_sc is not None:
        st.divider()
        st.markdown("<div class='section-title'>comparativo detalhado por ativo — HRP+BL vs customizado</div>",
                    unsafe_allow_html=True)
        st.caption("Para cada ativo: retorno esperado no cenário × peso em cada carteira × contribuição ao retorno total.")

        comp_rows = []
        for a in ASSET_CFG:
            nome = a["name"]
            ret_a = asset_rets.get(nome, 0)
            w_hrp = WEIGHTS.get(nome, 0)
            w_cust = custom_w_sc.get(nome, 0)
            contrib_hrp  = w_hrp  * ret_a
            contrib_cust = w_cust * ret_a
            comp_rows.append({
                "Ativo":             nome,
                "Cluster":           a["cluster"],
                "Retorno esperado":  f"{ret_a:.2f}%",
                "Peso HRP+BL":       f"{w_hrp*100:.1f}%",
                "Contrib. HRP+BL":   f"{contrib_hrp:.2f}%",
                "Peso Customizado":  f"{w_cust*100:.1f}%",
                "Contrib. Custom.":  f"{contrib_cust:.2f}%",
                "Δ Contribuição":    f"{contrib_cust-contrib_hrp:+.2f}%",
            })
        df_comp_sc = pd.DataFrame(comp_rows)

        # Linha de total
        total_row = {
            "Ativo": "TOTAL",
            "Cluster": "—",
            "Retorno esperado": "—",
            "Peso HRP+BL":     "100.0%",
            "Contrib. HRP+BL": f"{port_ret_sc:.2f}%",
            "Peso Customizado":"100.0%",
            "Contrib. Custom.":f"{custom_ret_sc:.2f}%",
            "Δ Contribuição":  f"{custom_ret_sc-port_ret_sc:+.2f}%",
        }
        df_comp_sc = pd.concat([df_comp_sc,
                                 pd.DataFrame([total_row])],
                                ignore_index=True)
        st.dataframe(df_comp_sc, use_container_width=True, hide_index=True)

        # Gráfico de contribuição comparativa
        fig_contrib = go.Figure()
        nomes_at = [r["Ativo"] for r in comp_rows]
        contrib_hrp_vals  = [WEIGHTS.get(a["name"],0)*asset_rets.get(a["name"],0)
                              for a in ASSET_CFG]
        contrib_cust_vals = [custom_w_sc.get(a["name"],0)*asset_rets.get(a["name"],0)
                              for a in ASSET_CFG]
        fig_contrib.add_trace(go.Bar(
            x=nomes_at, y=[round(v,3) for v in contrib_hrp_vals],
            name="Contribuição HRP+BL", marker_color="rgba(55,138,221,0.8)"))
        fig_contrib.add_trace(go.Bar(
            x=nomes_at, y=[round(v,3) for v in contrib_cust_vals],
            name="Contribuição Customizado", marker_color="rgba(226,75,74,0.7)"))
        fig_contrib.update_layout(
            plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
            height=260, barmode="group",
            font=dict(color="#1a1a18"),
            margin=dict(l=0,r=0,t=8,b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(color="#1a1a18")),
            xaxis=dict(gridcolor="#e8e6e0",
                       tickfont=dict(color="#444441", size=11), color="#1a1a18"),
            yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0",
                       tickfont=dict(color="#444441", size=11), color="#1a1a18",
                       title="Contribuição ao retorno (%)"),
        )
        st.plotly_chart(fig_contrib, use_container_width=True)
        st.caption("Contribuição = peso × retorno esperado do ativo no cenário. "
                   "Diferenças mostram onde os pesos customizados apostam diferente do HRP+BL.")

    # ── Forma 2: BL Dinâmico — sugestão de pesos ─────────────────────────────
    st.divider()
    st.markdown("### 🎯 Black-Litterman Dinâmico — Sugestão de pesos para este cenário")
    st.markdown(
        "O modelo usa os retornos esperados deste cenário como **visões táticas** do Black-Litterman, "
        "combina com o equilíbrio histórico HRP e sugere uma nova alocação ótima. "
        "Quanto maior a confiança, mais os pesos se afastam do HRP original em direção ao cenário."
    )

    col_bl1, col_bl2 = st.columns(2)
    tau_bl = col_bl1.slider(
        "Nível de confiança nas visões do cenário",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0 = ignora o cenário (mantém HRP puro) | 1 = segue totalmente o cenário"
    )
    rodar_bl = col_bl2.button("▶ Calcular pesos BL para este cenário", key="btn_bl_dyn", type="primary")

    if rodar_bl:
        with st.spinner("Calculando Black-Litterman dinâmico..."):
            try:
                # ── Montar retornos históricos ──
                ret_dict_bl = {}
                for cfg in ASSET_CFG:
                    r = series[cfg["name"]]["valor"].pct_change().dropna()
                    r = r.reindex(common_idx).ffill().dropna()
                    ret_dict_bl[cfg["name"]] = r
                ret_df_bl = pd.DataFrame(ret_dict_bl).dropna()

                # ── Covariância histórica ──
                cov_bl = ret_df_bl.cov() * 12  # anualizar

                # ── Retornos de equilíbrio (Pi = delta * Sigma * w) ──
                delta_bl = 2.5  # aversão ao risco de mercado
                w_hrp_bl = np.array([WEIGHTS.get(a["name"], 0) for a in ASSET_CFG])
                ativos_bl = [a["name"] for a in ASSET_CFG]
                cov_np    = cov_bl.loc[ativos_bl, ativos_bl].values
                pi_bl     = delta_bl * cov_np @ w_hrp_bl  # retornos de equilíbrio

                # ── Visões do cenário (Q = retornos esperados por ativo) ──
                q_bl = np.array([asset_rets.get(a, 0) / 100 for a in ativos_bl])

                # ── Matriz de visões P (identidade — visão sobre cada ativo) ──
                P_bl = np.eye(len(ativos_bl))

                # ── Incerteza das visões (Omega proporcional à variância) ──
                tau_sigma = (1 - tau_bl + 0.01) * cov_np  # tau controla confiança
                omega_bl  = np.diag(np.diag(P_bl @ tau_sigma @ P_bl.T))

                # ── Fórmula BL ──
                # E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
                tau_sigma_inv = np.linalg.inv(tau_sigma + np.eye(len(ativos_bl))*1e-8)
                omega_inv     = np.linalg.inv(omega_bl + np.eye(len(ativos_bl))*1e-8)

                M1 = tau_sigma_inv + P_bl.T @ omega_inv @ P_bl
                M2 = tau_sigma_inv @ pi_bl + P_bl.T @ omega_inv @ q_bl

                er_bl = np.linalg.solve(M1, M2)  # retornos ajustados BL

                # ── Rodar HRP com retornos BL ajustados ──
                # Ajustar matriz de covariância pelos novos retornos esperados
                er_series = pd.Series(er_bl, index=ativos_bl)

                # Calcular pesos pelo inverso da variância ajustada pelo retorno BL
                var_ajustada = pd.Series({
                    a: max(cov_np[i,i], 1e-6) / max(er_bl[i]**2, 1e-6)
                    for i, a in enumerate(ativos_bl)
                })
                # Normalizar
                w_bl_raw = 1 / var_ajustada
                w_bl_raw = w_bl_raw / w_bl_raw.sum()

                # Combinar com HRP original pela confiança
                w_hrp_s  = pd.Series(w_hrp_bl, index=ativos_bl)
                w_bl_fin = (tau_bl * w_bl_raw + (1-tau_bl) * w_hrp_s)
                w_bl_fin = w_bl_fin / w_bl_fin.sum()

                # ── Retorno esperado com novos pesos ──
                ret_bl_port = sum(w_bl_fin[a] * asset_rets.get(a, 0)
                                  for a in ativos_bl)
                ret_hrp_port = sum(WEIGHTS.get(a, 0) * asset_rets.get(a, 0)
                                   for a in ativos_bl)

                # ── Exibir resultado ──
                st.markdown("#### Pesos sugeridos pelo BL Dinâmico")

                rows_bl = []
                for cfg in ASSET_CFG:
                    nome   = cfg["name"]
                    w_orig = cfg["w"] * 100
                    w_novo = w_bl_fin.get(nome, 0) * 100
                    r_esp  = asset_rets.get(nome, 0)
                    delta  = w_novo - w_orig
                    rows_bl.append({
                        "Ativo":          nome,
                        "Cluster":        cfg["cluster"],
                        "Ret. esperado":  f"{r_esp:.2f}%",
                        "Peso HRP atual": f"{w_orig:.1f}%",
                        "Peso BL novo":   f"{w_novo:.1f}%",
                        "Δ Peso":         f"{delta:+.1f}%",
                        "Ação":           "⬆️ Aumentar" if delta > 1.0 else
                                          "⬇️ Reduzir"  if delta < -1.0 else
                                          "✅ Manter",
                    })
                df_bl = pd.DataFrame(rows_bl).set_index("Ativo")
                st.dataframe(df_bl, use_container_width=True)

                # KPIs comparativos
                k1, k2, k3 = st.columns(3)
                k1.metric("Retorno HRP+BL original", f"{ret_hrp_port:.2f}%",
                           delta=f"{ret_hrp_port-selic:+.2f}% vs CDI")
                k2.metric("Retorno BL Dinâmico",     f"{ret_bl_port:.2f}%",
                           delta=f"{ret_bl_port-selic:+.2f}% vs CDI")
                k3.metric("Ganho do rebalanceamento", f"{ret_bl_port-ret_hrp_port:+.2f}%",
                           delta="vs HRP original")

                # Gráfico comparativo de pesos
                fig_bl = go.Figure()
                fig_bl.add_trace(go.Bar(
                    x=ativos_bl,
                    y=[cfg["w"]*100 for cfg in ASSET_CFG],
                    name="HRP+BL original",
                    marker_color="rgba(55,138,221,0.7)",
                ))
                fig_bl.add_trace(go.Bar(
                    x=ativos_bl,
                    y=[w_bl_fin.get(a,0)*100 for a in ativos_bl],
                    name=f"BL Dinâmico (confiança {tau_bl:.0%})",
                    marker_color="rgba(29,158,117,0.7)",
                ))
                fig_bl.update_layout(
                    plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
                    height=280, barmode="group",
                    font=dict(color="#1a1a18"),
                    margin=dict(l=0,r=0,t=8,b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="left", x=0, font=dict(color="#1a1a18")),
                    xaxis=dict(gridcolor="#e8e6e0",
                               tickfont=dict(color="#444441",size=11), color="#1a1a18"),
                    yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0",
                               tickfont=dict(color="#444441",size=11), color="#1a1a18"),
                )
                st.plotly_chart(fig_bl, use_container_width=True)

                # Botão para aplicar como customizado
                pesos_str = " | ".join([
                    f"{a}: {w_bl_fin.get(a,0)*100:.1f}%"
                    for a in ativos_bl
                ])
                st.code(pesos_str, language=None)
                st.caption(
                    "💡 Para aplicar: copie os valores acima e insira nos campos "
                    "da aba Rebalanceamento. O portfólio customizado será recalculado "
                    "automaticamente em todas as abas."
                )

                # Salvar pesos BL no session_state para uso no Monte Carlo
                for a in ativos_bl:
                    st.session_state[f"bl_dyn_{a}"] = round(w_bl_fin.get(a,0)*100, 1)
                st.session_state["bl_dyn_scenario"] = preset
                st.success(
                    f"✅ Pesos BL calculados para o cenário **{preset}** "
                    f"com confiança de **{tau_bl:.0%}**. "
                    f"Disponíveis para uso no Monte Carlo."
                )

            except Exception as e:
                st.error(f"Erro no cálculo BL Dinâmico: {e}")


# ── Tab 6: Eventos de cauda ───────────────────────────────────────────────────
with tab6:
    st.markdown("Análise de performance do portfólio nos principais eventos de estresse históricos. "
                "Os retornos são calculados com os dados reais carregados.")

    # Filtros
    col_f1, col_f2 = st.columns(2)
    tipo_filtro = col_f1.multiselect("Tipo de evento", ["Doméstico", "Global"],
                                      default=["Doméstico", "Global"])
    show_custom_ev = col_f2.checkbox("Incluir portfólio customizado", value=True)

    eventos_filtrados = [ev for ev in TAIL_EVENTS if ev["tipo"] in tipo_filtro]

    # ── Tabela de performance por evento ──
    st.markdown("<div class='section-title'>performance em cada evento</div>", unsafe_allow_html=True)

    custom_w_ev = {}
    for cfg in ASSET_CFG:
        key = f"rebal_{cfg['name']}"
        custom_w_ev[cfg["name"]] = st.session_state.get(key, cfg["w"] * 100) / 100

    custom_ret_ev = sum(
        custom_w_ev[a["name"]] * series[a["name"]]["valor"].pct_change().dropna().reindex(common_idx).ffill()
        for a in ASSET_CFG
    )
    custom_cum_ev = (1 + custom_ret_ev).cumprod() * 100

    def period_ret(cum_series, start, end):
        try:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            v0 = cum_series[cum_series.index <= s]
            v1 = cum_series[cum_series.index <= e]
            if len(v0) == 0 or len(v1) == 0:
                return None
            return round((v1.iloc[-1] / v0.iloc[-1] - 1) * 100, 2)
        except Exception:
            return None

    rows_ev = []
    for ev in eventos_filtrados:
        r_hrp  = period_ret(port_cum,    ev["start"], ev["end"])
        r_cdi  = period_ret(cdi_cum,     ev["start"], ev["end"])
        r_ibov = period_ret(ibov_cum,    ev["start"], ev["end"])
        r_cust = period_ret(custom_cum_ev, ev["start"], ev["end"])
        eq_cum_ev = (1 + sum((1/6)*series[a["name"]]["valor"].pct_change().dropna().reindex(common_idx).ffill()
                              for a in ASSET_CFG)).cumprod() * 100
        r_igual = period_ret(eq_cum_ev, ev["start"], ev["end"])

        row = {
            "Evento":   ev["name"],
            "Tipo":     ev["tipo"],
            "Período":  f"{pd.Timestamp(ev['start']).strftime('%d/%m/%Y')} → {pd.Timestamp(ev['end']).strftime('%d/%m/%Y')}",
            "HRP+BL":   f"{r_hrp:+.1f}%" if r_hrp is not None else "—",
            "CDI":      f"{r_cdi:+.1f}%"  if r_cdi  is not None else "—",
            "Ibovespa": f"{r_ibov:+.1f}%" if r_ibov is not None else "—",
            "1/N":      f"{r_igual:+.1f}%" if r_igual is not None else "—",
        }
        if show_custom_ev:
            row["Customizado"] = f"{r_cust:+.1f}%" if r_cust is not None else "—"
        rows_ev.append(row)

    df_ev = pd.DataFrame(rows_ev)
    st.dataframe(df_ev, use_container_width=True, hide_index=True)

    st.divider()

    # ── Gráfico de barras comparativo ──
    st.markdown("<div class='section-title'>comparativo visual por evento</div>", unsafe_allow_html=True)

    ev_names  = [r["Evento"]   for r in rows_ev]
    ev_hrp    = [float(r["HRP+BL"].replace("%","").replace("+","")) if r["HRP+BL"] != "—" else 0 for r in rows_ev]
    ev_ibov   = [float(r["Ibovespa"].replace("%","").replace("+","")) if r["Ibovespa"] != "—" else 0 for r in rows_ev]
    ev_cdi    = [float(r["CDI"].replace("%","").replace("+","")) if r["CDI"] != "—" else 0 for r in rows_ev]
    ev_igual  = [float(r["1/N"].replace("%","").replace("+","")) if r["1/N"] != "—" else 0 for r in rows_ev]

    fig_ev = go.Figure()
    fig_ev.add_trace(go.Bar(x=ev_names, y=ev_hrp,  name="HRP+BL",   marker_color="#378ADD"))
    fig_ev.add_trace(go.Bar(x=ev_names, y=ev_cdi,  name="CDI",      marker_color="rgba(29,158,117,0.6)"))
    fig_ev.add_trace(go.Bar(x=ev_names, y=ev_ibov, name="Ibovespa", marker_color="rgba(186,117,23,0.5)"))
    fig_ev.add_trace(go.Bar(x=ev_names, y=ev_igual,name="1/N igual", marker_color="rgba(127,119,221,0.6)"))
    ev_ipca6 = [float(r.get("IPCA","0").replace("%","").replace("+","")) if r.get("IPCA","—") != "—" else 0 for r in rows_ev]
    fig_ev.add_trace(go.Bar(x=ev_names, y=ev_ipca6, name="IPCA", marker_color="rgba(196,119,10,0.6)"))
    if show_custom_ev:
        ev_cust = [float(r["Customizado"].replace("%","").replace("+",""))
                   if r.get("Customizado","—") != "—" else 0 for r in rows_ev]
        fig_ev.add_trace(go.Bar(x=ev_names, y=ev_cust, name="Customizado", marker_color="rgba(226,75,74,0.7)"))

    fig_ev.add_hline(y=0, line_color="#888780", line_width=1)
    fig_ev.update_layout(
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        height=360, barmode="group",
        font=dict(color="#1a1a18"),
        margin=dict(l=0,r=0,t=8,b=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#1a1a18")),
        xaxis=dict(gridcolor="#e8e6e0", tickfont=dict(color="#444441", size=10),
                   color="#1a1a18", tickangle=-35),
        yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0",
                   tickfont=dict(color="#444441", size=11), color="#1a1a18",
                   zeroline=True, zerolinecolor="#888780"),
    )
    st.plotly_chart(fig_ev, use_container_width=True)

    st.divider()

    # ── Gráfico acumulado com eventos marcados ──
    st.markdown("<div class='section-title'>retorno acumulado com eventos destacados</div>", unsafe_allow_html=True)

    fig_ev2 = go.Figure()
    fig_ev2.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values.round(2),
        name="HRP+BL", line=dict(color="#378ADD", width=2.5)))
    fig_ev2.add_trace(go.Scatter(x=cdi_cum.index, y=cdi_cum.values.round(2),
        name="CDI", line=dict(color="#1D9E75", width=1.5, dash="dot")))
    fig_ev2.add_trace(go.Scatter(x=ibov_cum.index, y=ibov_cum.values.round(2),
        name="Ibovespa", line=dict(color="#BA7517", width=1.5, dash="dash")))

    for ev in eventos_filtrados:
        try:
            ev_start = pd.Timestamp(ev["start"])
            ev_end   = pd.Timestamp(ev["end"])
            if ev_start >= port_cum.index[0] and ev_start <= port_cum.index[-1]:
                fig_ev2.add_vrect(
                    x0=ev_start, x1=ev_end,
                    fillcolor=ev["color"], opacity=0.15,
                    layer="below", line_width=0,
                )
                fig_ev2.add_annotation(
                    x=ev_start, y=1.0, yref="paper",
                    text=ev["name"], textangle=-90,
                    showarrow=False,
                    font=dict(size=9, color=ev["color"]),
                    xanchor="right", yanchor="top",
                    bgcolor="rgba(248,247,244,0.8)",
                )
        except Exception:
            pass

    fig_ev2.update_layout(
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        height=420, hovermode="x unified",
        font=dict(color="#1a1a18"),
        margin=dict(l=0,r=40,t=8,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#1a1a18")),
        yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0", tickfont=dict(color="#444441", size=11), color="#1a1a18", hoverformat=".1f"),
        xaxis=dict(gridcolor="#e8e6e0", tickfont=dict(color="#444441", size=11), color="#1a1a18"),
    )
    st.plotly_chart(fig_ev2, use_container_width=True)

    # ── Legenda dos eventos ──
    st.markdown("<div class='section-title'>legenda dos eventos</div>", unsafe_allow_html=True)
    cols_leg = st.columns(2)
    for i, ev in enumerate(eventos_filtrados):
        col = cols_leg[i % 2]
        tipo_badge = "🇧🇷" if ev["tipo"] == "Doméstico" else "🌍"
        col.markdown(
            f"<div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:8px'>"
            f"<span style='width:10px;height:10px;border-radius:50%;background:{ev['color']};flex-shrink:0;margin-top:4px'></span>"
            f"<div><strong style='font-size:13px'>{tipo_badge} {ev['name']}</strong>"
            f"<br><span style='font-size:12px;color:#888780'>{ev['desc']}</span></div></div>",
            unsafe_allow_html=True
        )



# ── Tab 7: Ativos individuais ─────────────────────────────────────────────────
with tab7:
    st.markdown("Retorno acumulado de cada ativo da carteira de forma individual, "
                "com marcações dos eventos de cauda e filtros de período.")

    # ── Controles ──
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])

    periodos_at = {
        "Mês passado": 1, "3 meses": 3, "12 meses": 12,
        "24 meses": 24, "36 meses": 36, "5 anos": 60,
        "10 anos": 120, "Histórico completo": None,
    }
    periodo_at = ctrl1.selectbox("Período", list(periodos_at.keys()),
                                  index=len(periodos_at)-1, key="periodo_at")

    ativos_disponiveis = [a["name"] for a in ASSET_CFG]
    ativos_sel = ctrl2.multiselect("Ativos", ativos_disponiveis,
                                    default=ativos_disponiveis)

    tipo_ev_at = ctrl3.multiselect("Tipo de evento", ["Doméstico", "Global"],
                                    default=["Doméstico", "Global"], key="tipo_ev_at")

    show_ev_at = st.checkbox("Marcar eventos de cauda no gráfico", value=True, key="ev_at")

    if not ativos_sel:
        st.warning("Selecione pelo menos um ativo.")
        st.stop()

    # ── Filtro de período ──
    n_at = periodos_at[periodo_at]
    if n_at is not None:
        corte_at = common_idx[-1] - pd.DateOffset(months=n_at)
        idx_at = common_idx[common_idx >= corte_at]
    else:
        idx_at = common_idx

    # ── Construir séries rebased ──
    fig_at = go.Figure()
    for cfg in ASSET_CFG:
        if cfg["name"] not in ativos_sel:
            continue
        s = series[cfg["name"]]["valor"].reindex(idx_at).ffill().dropna()
        if len(s) == 0:
            continue
        s_rb = (s / s.iloc[0]) * 100
        fig_at.add_trace(go.Scatter(
            x=s_rb.index, y=s_rb.values.round(2),
            name=cfg["name"],
            line=dict(color=cfg["color"], width=2),
            hovertemplate=f"<b>{cfg['name']}</b><br>%{{x|%b/%Y}}<br>%{{y:.1f}}<extra></extra>",
        ))

    # ── Marcações de eventos ──
    if show_ev_at:
        evs_at = [ev for ev in TAIL_EVENTS if ev["tipo"] in tipo_ev_at]
        for ev in evs_at:
            try:
                ev_s = pd.Timestamp(ev["start"])
                ev_e = pd.Timestamp(ev["end"])
                if ev_s >= idx_at[0] and ev_s <= idx_at[-1]:
                    fig_at.add_vrect(
                        x0=ev_s, x1=ev_e,
                        fillcolor=ev["color"], opacity=0.12,
                        layer="below", line_width=0,
                    )
                    fig_at.add_annotation(
                        x=ev_s, y=1.0, yref="paper",
                        text=ev["name"], textangle=-90,
                        showarrow=False,
                        font=dict(size=9, color=ev["color"]),
                        xanchor="right", yanchor="top",
                        bgcolor="rgba(248,247,244,0.8)",
                    )
            except Exception:
                pass

    fig_at.update_layout(
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        height=460, hovermode="x unified",
        font=dict(color="#1a1a18"),
        margin=dict(l=0, r=40, t=8, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(color="#1a1a18")),
        yaxis=dict(title="Base 100", gridcolor="#e8e6e0",
                   tickfont=dict(color="#444441", size=11), color="#1a1a18"),
        xaxis=dict(gridcolor="#e8e6e0",
                   tickfont=dict(color="#444441", size=11), color="#1a1a18"),
    )
    st.plotly_chart(fig_at, use_container_width=True)
    st.caption(f"Base 100 = primeiro dia do período selecionado ({idx_at[0].strftime('%d/%m/%Y')}). "
               f"Retornos em frequência mensal.")

    st.divider()

    # ── Tabela de retornos por ativo no período ──
    st.markdown("<div class='section-title'>retorno acumulado no período selecionado</div>",
                unsafe_allow_html=True)

    rows_at = []
    for cfg in ASSET_CFG:
        if cfg["name"] not in ativos_sel:
            continue
        s = series[cfg["name"]]["valor"].reindex(idx_at).ffill().dropna()
        if len(s) < 2:
            continue
        ret_ac  = round((s.iloc[-1] / s.iloc[0] - 1) * 100, 2)
        ret_mo  = s.pct_change().dropna()
        vol_an  = round(ret_mo.std() * (12**0.5) * 100, 2)
        rf_at   = cdi_aligned.reindex(idx_at).ffill()
        sh = round((ret_mo.mean() - rf_at.mean()) / ret_mo.std() * (12**0.5), 3) if ret_mo.std() > 0 else None
        cum_s   = (1 + ret_mo).cumprod()
        dd_s    = (cum_s - cum_s.cummax()) / cum_s.cummax()
        max_dd  = round(dd_s.min() * 100, 2)
        rows_at.append({
            "Ativo":            cfg["name"],
            "Cluster":          cfg["cluster"],
            "Retorno acum.":    f"{ret_ac:+.1f}%",
            "Vol. a.a.":        f"{vol_an:.1f}%",
            "Sharpe":           f"{sh:.3f}" if sh else "—",
            "Max Drawdown":     f"{max_dd:.1f}%",
            "Peso HRP+BL":      f"{cfg['w']*100:.1f}%",
        })

    if rows_at:
        df_at = pd.DataFrame(rows_at)
        st.dataframe(df_at, use_container_width=True, hide_index=True)

    st.divider()

    # ── Performance de cada ativo nos eventos de cauda ──
    st.markdown("<div class='section-title'>impacto dos eventos de cauda por ativo</div>",
                unsafe_allow_html=True)

    evs_tbl = [ev for ev in TAIL_EVENTS if ev["tipo"] in tipo_ev_at]
    if evs_tbl and ativos_sel:
        rows_imp = []
        for ev in evs_tbl:
            row_imp = {
                "Evento":  ev["name"],
                "Tipo":    ev["tipo"],
                "Período": f"{pd.Timestamp(ev['start']).strftime('%d/%m/%Y')} → {pd.Timestamp(ev['end']).strftime('%d/%m/%Y')}",
            }
            for cfg in ASSET_CFG:
                if cfg["name"] not in ativos_sel:
                    continue
                s_full = series[cfg["name"]]["valor"].reindex(common_idx).ffill()
                s_cum_full = (s_full / s_full.iloc[0]) * 100
                try:
                    v0 = s_cum_full[s_cum_full.index <= pd.Timestamp(ev["start"])]
                    v1 = s_cum_full[s_cum_full.index <= pd.Timestamp(ev["end"])]
                    if len(v0) and len(v1):
                        r = round((v1.iloc[-1] / v0.iloc[-1] - 1) * 100, 2)
                        row_imp[cfg["name"]] = f"{r:+.1f}%"
                    else:
                        row_imp[cfg["name"]] = "—"
                except Exception:
                    row_imp[cfg["name"]] = "—"
            rows_imp.append(row_imp)

        df_imp = pd.DataFrame(rows_imp)
        st.dataframe(df_imp, use_container_width=True, hide_index=True)
    else:
        st.info("Selecione ativos e tipos de eventos para ver o impacto.")



# ── Tab 8: Análise comparativa ────────────────────────────────────────────────
with tab8:
    st.markdown("Análise comparativa detalhada entre os portfólios — distribuição mensal "
                "de retornos, consistência vs CDI e estatísticas descritivas.")

    # ── Recalcular portfólio customizado ──────────────────────────────────────
    custom_w_t8 = {cfg["name"]: st.session_state.get(f"rebal_{cfg['name']}", cfg["w"]*100)/100
                   for cfg in ASSET_CFG}
    total_t8 = sum(custom_w_t8.values())
    custom_valid_t8 = abs(total_t8 - 1.0) < 0.02

    c_ret_t8 = sum(
        custom_w_t8[a["name"]] * series[a["name"]]["valor"].pct_change().dropna().reindex(common_idx).ffill()
        for a in ASSET_CFG
    ) if custom_valid_t8 else port_ret

    # Séries de retorno mensais de cada portfólio
    portfolios = {
        "HRP+BL":    port_ret,
        "CDI":       cdi_aligned,
        "Ibovespa":  ibov_ret,
        "IPCA":      ipca6_ret,
    }
    if custom_valid_t8:
        portfolios["Customizado"] = c_ret_t8

    cores = {
        "HRP+BL":    "#378ADD",
        "CDI":       "#1D9E75",
        "Ibovespa":  "#BA7517",
        "IPCA":      "#C4770A",
        "Customizado":"#E24B4A",
    }

    # ── Bloco 1: Consistência vs CDI ─────────────────────────────────────────
    st.markdown("<div class='section-title'>consistência mensal vs CDI</div>",
                unsafe_allow_html=True)

    cols_cons = st.columns(len(portfolios))
    for i, (nome, ret) in enumerate(portfolios.items()):
        if nome == "CDI":
            continue
        acima = (ret > cdi_aligned).sum()
        abaixo = (ret <= cdi_aligned).sum()
        total_m = acima + abaixo
        pct_acima = acima / total_m * 100
        with cols_cons[i]:
            st.markdown(
                f"<div class='metric-card' style='border-top:3px solid {cores[nome]}'>"
                f"<div class='metric-label'>{nome}</div>"
                f"<div class='metric-value pos' style='font-size:26px'>{acima}</div>"
                f"<div class='metric-sub'>meses acima do CDI ({pct_acima:.0f}%)</div>"
                f"<div style='margin-top:8px;font-size:13px;color:#A32D2D'>"
                f"<strong>{abaixo}</strong> meses abaixo ({100-pct_acima:.0f}%)</div>"
                f"</div>", unsafe_allow_html=True
            )

    st.divider()

    # ── Bloco 2: Retornos positivos vs negativos ──────────────────────────────
    st.markdown("<div class='section-title'>meses positivos vs negativos</div>",
                unsafe_allow_html=True)

    cols_pn = st.columns(len(portfolios))
    for i, (nome, ret) in enumerate(portfolios.items()):
        positivos = (ret > 0).sum()
        negativos = (ret <= 0).sum()
        total_m = positivos + negativos
        pct_pos = positivos / total_m * 100
        with cols_pn[i]:
            st.markdown(
                f"<div class='metric-card' style='border-top:3px solid {cores[nome]}'>"
                f"<div class='metric-label'>{nome}</div>"
                f"<div class='metric-value pos' style='font-size:26px'>{positivos}</div>"
                f"<div class='metric-sub'>meses positivos ({pct_pos:.0f}%)</div>"
                f"<div style='margin-top:8px;font-size:13px;color:#A32D2D'>"
                f"<strong>{negativos}</strong> meses negativos ({100-pct_pos:.0f}%)</div>"
                f"</div>", unsafe_allow_html=True
            )

    st.divider()

    # ── Bloco 3: Gráfico de barras — meses acima/abaixo CDI ──────────────────
    st.markdown("<div class='section-title'>comparativo visual — consistência vs CDI</div>",
                unsafe_allow_html=True)

    nomes_graf = [n for n in portfolios if n != "CDI"]
    acima_vals = [(portfolios[n] > cdi_aligned).sum() for n in nomes_graf]
    abaixo_vals = [(portfolios[n] <= cdi_aligned).sum() for n in nomes_graf]

    fig_cons = go.Figure()
    fig_cons.add_trace(go.Bar(
        x=nomes_graf, y=acima_vals,
        name="Meses acima do CDI",
        marker_color=[cores[n] for n in nomes_graf],
    ))
    fig_cons.add_trace(go.Bar(
        x=nomes_graf, y=[-v for v in abaixo_vals],
        name="Meses abaixo do CDI",
        marker_color=[hex_to_rgba(cores[n], 0.45) for n in nomes_graf],
    ))
    # Anotações manuais acima/abaixo das barras
    for i, n in enumerate(nomes_graf):
        pct_a = acima_vals[i] / (acima_vals[i] + abaixo_vals[i]) * 100
        fig_cons.add_annotation(
            x=n, y=acima_vals[i] + 3,
            text=f"{acima_vals[i]} ({pct_a:.0f}%)",
            showarrow=False, font=dict(size=11, color=cores[n]), yanchor="bottom"
        )
        fig_cons.add_annotation(
            x=n, y=-abaixo_vals[i] - 3,
            text=f"{abaixo_vals[i]} ({100-pct_a:.0f}%)",
            showarrow=False, font=dict(size=11, color="#A32D2D"), yanchor="top"
        )
    fig_cons.update_layout(
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        height=320, barmode="overlay",
        font=dict(color="#1a1a18"),
        margin=dict(l=0,r=0,t=30,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(color="#1a1a18")),
        xaxis=dict(gridcolor="#e8e6e0", tickfont=dict(color="#444441",size=12),
                   color="#1a1a18"),
        yaxis=dict(gridcolor="#e8e6e0", tickfont=dict(color="#444441",size=11),
                   color="#1a1a18", zeroline=True, zerolinecolor="#888780",
                   title="Número de meses"),
    )
    st.plotly_chart(fig_cons, use_container_width=True)

    st.divider()

    # ── Bloco 4: Estatísticas descritivas ─────────────────────────────────────
    st.markdown("<div class='section-title'>estatísticas descritivas dos retornos mensais</div>",
                unsafe_allow_html=True)

    stats_rows = []
    for nome, ret in portfolios.items():
        r = ret.dropna()
        stats_rows.append({
            "Portfólio":       nome,
            "Média mensal":    f"{r.mean()*100:.3f}%",
            "Mediana mensal":  f"{r.median()*100:.3f}%",
            "Desvio padrão":   f"{r.std()*100:.3f}%",
            "Melhor mês":      f"+{r.max()*100:.2f}%",
            "Pior mês":        f"{r.min()*100:.2f}%",
            "Assimetria":      f"{float(r.skew()):.3f}",
            "Curtose":         f"{float(r.kurt()):.3f}",
            "Meses totais":    str(len(r)),
        })
    df_stats = pd.DataFrame(stats_rows).set_index("Portfólio")
    st.dataframe(df_stats, use_container_width=True)
    st.caption(
        "Assimetria positiva = distribuição com mais retornos extremos positivos (bom). "
        "Curtose alta = caudas pesadas (mais eventos extremos que uma distribuição normal)."
    )

    st.divider()

    # ── Bloco 5: Distribuição dos retornos mensais (histograma) ───────────────
    st.markdown("<div class='section-title'>distribuição dos retornos mensais</div>",
                unsafe_allow_html=True)

    port_sel = st.multiselect(
        "Selecionar portfólios", list(portfolios.keys()),
        default=["HRP+BL","CDI","Ibovespa"],
        key="hist_sel"
    )
    fig_hist = go.Figure()
    for nome in port_sel:
        ret = portfolios[nome].dropna() * 100
        fig_hist.add_trace(go.Histogram(
            x=ret.values, name=nome,
            nbinsx=40, opacity=0.65,
            marker_color=cores[nome],
            hovertemplate=f"<b>{nome}</b><br>Retorno: %{{x:.2f}}%<br>Freq: %{{y}}<extra></extra>",
        ))
    fig_hist.update_layout(
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        height=300, barmode="overlay",
        font=dict(color="#1a1a18"),
        margin=dict(l=0,r=0,t=8,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(color="#1a1a18")),
        xaxis=dict(gridcolor="#e8e6e0", ticksuffix="%",
                   tickfont=dict(color="#444441",size=11), color="#1a1a18",
                   title="Retorno mensal (%)"),
        yaxis=dict(gridcolor="#e8e6e0",
                   tickfont=dict(color="#444441",size=11), color="#1a1a18",
                   title="Frequência"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("Distribuição dos retornos mensais. Ideal: assimétrica à direita (cauda positiva mais longa).")

    st.divider()

    # ── Bloco 6: Tabela comparativa HRP+BL vs Customizado ────────────────────
    if custom_valid_t8:
        st.markdown("<div class='section-title'>HRP+BL vs portfólio customizado — detalhado</div>",
                    unsafe_allow_html=True)

        m_c = metrics(c_ret_t8, cdi_aligned)
        acum_c = ((1+c_ret_t8).cumprod().iloc[-1] - 1) * 100

        comp_rows = {
            "Retorno acumulado":   (f"+{acum_port:.1f}%",             f"+{acum_c:.1f}%"),
            "Retorno a.a.":        (f"{m_port['ann_ret']*100:.2f}%",   f"{m_c['ann_ret']*100:.2f}%"),
            "Volatilidade a.a.":   (f"{m_port['ann_vol']*100:.2f}%",   f"{m_c['ann_vol']*100:.2f}%"),
            "Sharpe (rf=CDI)":     (f"{m_port['sharpe']:.3f}",         f"{m_c['sharpe']:.3f}"),
            "Sortino":             (f"{m_port['sortino']:.3f}" if m_port["sortino"] else "—",
                                    f"{m_c['sortino']:.3f}" if m_c["sortino"] else "—"),
            "Max drawdown":        (f"{m_port['max_dd']*100:.2f}%",    f"{m_c['max_dd']*100:.2f}%"),
            "Calmar ratio":        (f"{m_port['calmar']:.3f}",         f"{m_c['calmar']:.3f}"),
            "VaR 95% mensal":      (f"{m_port['var95']*100:.2f}%",     f"{m_c['var95']*100:.2f}%"),
            "Meses acima CDI":     (f"{(port_ret > cdi_aligned).sum()} ({(port_ret > cdi_aligned).mean()*100:.0f}%)",
                                    f"{(c_ret_t8 > cdi_aligned).sum()} ({(c_ret_t8 > cdi_aligned).mean()*100:.0f}%)"),
            "Meses positivos":     (f"{(port_ret > 0).sum()} ({(port_ret > 0).mean()*100:.0f}%)",
                                    f"{(c_ret_t8 > 0).sum()} ({(c_ret_t8 > 0).mean()*100:.0f}%)"),
            "Melhor mês":          (f"+{port_ret.max()*100:.2f}%",     f"+{c_ret_t8.max()*100:.2f}%"),
            "Pior mês":            (f"{port_ret.min()*100:.2f}%",      f"{c_ret_t8.min()*100:.2f}%"),
            "Assimetria":          (f"{float(port_ret.skew()):.3f}",   f"{float(c_ret_t8.skew()):.3f}"),
        }
        df_comp = pd.DataFrame(comp_rows, index=["HRP+BL","Customizado"]).T
        df_comp.index.name = "Métrica"
        st.dataframe(df_comp, use_container_width=True)
    else:
        st.info("Configure os pesos na aba Rebalanceamento para ver a comparação com o portfólio customizado.")



# ── Tab 9: Monitoramento diário ───────────────────────────────────────────────
with tab9:
    st.markdown("Monitoramento em frequência diária — drawdown real, retorno do período "
                "corrente e drift dos pesos em relação ao target HRP+BL.")

    if not daily_series:
        st.warning("⚠️ Nenhuma série diária disponível. "
                   "Faça upload dos arquivos ANBIMA ou aguarde o carregamento via Yahoo Finance.")
        st.stop()

    # ── Período disponível ────────────────────────────────────────────────────
    all_idx = None
    for s in daily_series.values():
        all_idx = s.index if all_idx is None else all_idx.intersection(s.index)

    if all_idx is None or len(all_idx) == 0:
        st.warning("⚠️ Não foi possível alinhar as séries diárias.")
        st.stop()

    hoje     = all_idx[-1]
    ini_mes  = hoje.replace(day=1)
    ini_sem  = hoje - pd.Timedelta(days=hoje.weekday())
    ini_ano  = hoje.replace(month=1, day=1)

    st.markdown(
        f"<div style='font-size:13px;color:#888780;margin-bottom:1rem'>"
        f"Último dado disponível: <strong>{hoje.strftime('%d/%m/%Y')}</strong> · "
        f"{len(all_idx)} dias úteis no histórico</div>",
        unsafe_allow_html=True
    )

    # ── Bloco 1: Retorno do período corrente ──────────────────────────────────
    st.markdown("<div class='section-title'>retorno do período corrente</div>",
                unsafe_allow_html=True)

    def ret_periodo(nome, ini):
        s = daily_series.get(nome)
        if s is None: return None
        s = s["valor"].dropna()
        sub = s[s.index >= ini]
        if len(sub) < 2: return None
        return round((sub.iloc[-1]/sub.iloc[0]-1)*100, 2)

    def port_ret_periodo(ini):
        rets = []
        for cfg in ASSET_CFG:
            r = ret_periodo(cfg["name"], ini)
            if r is not None:
                rets.append(cfg["w"] * r/100)
        if not rets: return None
        total = sum(WEIGHTS[cfg["name"]] for cfg in ASSET_CFG
                    if ret_periodo(cfg["name"], ini) is not None)
        return round(sum(rets)/total*100, 2) if total > 0 else None

    periodos_mon = {
        "Dia":  hoje - pd.Timedelta(days=1),
        "Semana": ini_sem,
        "Mês":  ini_mes,
        "Ano":  ini_ano,
    }

    cols_mon = st.columns(len(periodos_mon))
    for i, (label, ini) in enumerate(periodos_mon.items()):
        r = port_ret_periodo(ini)
        with cols_mon[i]:
            cls = "pos" if r and r >= 0 else "neg"
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Portfólio — {label}</div>"
                f"<div class='metric-value {cls}'>"
                f"{f'+{r:.2f}%' if r and r>=0 else f'{r:.2f}%' if r else '—'}</div>"
                f"<div class='metric-sub'>HRP+BL ponderado</div>"
                f"</div>", unsafe_allow_html=True
            )

    st.divider()

    # ── Bloco 2: Retorno por ativo no período ─────────────────────────────────
    st.markdown("<div class='section-title'>retorno por ativo no período</div>",
                unsafe_allow_html=True)

    periodo_sel_d = st.selectbox("Período", list(periodos_mon.keys()),
                                  index=2, key="mon_periodo")
    ini_sel = periodos_mon[periodo_sel_d]

    rows_d = []
    for cfg in ASSET_CFG:
        r = ret_periodo(cfg["name"], ini_sel)
        rows_d.append({
            "Ativo":   cfg["name"],
            "Cluster": cfg["cluster"],
            "Retorno": f"{f'+{r:.2f}%' if r and r>=0 else f'{r:.2f}%' if r else '—'}",
            "Peso HRP+BL": f"{cfg['w']*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows_d), use_container_width=True, hide_index=True)

    st.divider()

    # ── Bloco 3: Drawdown diário real ─────────────────────────────────────────
    st.markdown("<div class='section-title'>drawdown diário real</div>",
                unsafe_allow_html=True)

    # Calcular portfólio diário
    port_d_idx = all_idx
    port_d_ret = pd.Series(0.0, index=port_d_idx)
    for cfg in ASSET_CFG:
        s = daily_series.get(cfg["name"])
        if s is None: continue
        r = s["valor"].pct_change().reindex(port_d_idx).ffill().fillna(0)
        port_d_ret += cfg["w"] * r

    port_d_cum = (1 + port_d_ret).cumprod()
    port_d_dd  = (port_d_cum - port_d_cum.cummax()) / port_d_cum.cummax() * 100

    # Ibovespa diário drawdown
    ibov_d = daily_series.get("Ibovespa")
    ibov_d_dd = None
    if ibov_d is not None:
        ibov_d_r   = ibov_d["valor"].pct_change().reindex(port_d_idx).ffill().fillna(0)
        ibov_d_cum = (1 + ibov_d_r).cumprod()
        ibov_d_dd  = (ibov_d_cum - ibov_d_cum.cummax()) / ibov_d_cum.cummax() * 100

    # Filtro de período
    janelas_dd = {"3 meses":90,"6 meses":180,"12 meses":252,"3 anos":756,"Histórico":None}
    jan_dd = st.selectbox("Janela", list(janelas_dd.keys()), index=2, key="jan_dd")
    n_dd   = janelas_dd[jan_dd]
    dd_plot = port_d_dd.iloc[-n_dd:] if n_dd else port_d_dd
    labels_dd = [d.strftime("%d/%m/%Y") for d in dd_plot.index]

    fig_dd_d = go.Figure()
    fig_dd_d.add_trace(go.Scatter(
        x=dd_plot.index, y=dd_plot.values.round(2),
        name="HRP+BL (diário)", fill="tozeroy",
        line=dict(color="#378ADD", width=1.5),
        fillcolor="rgba(55,138,221,0.15)",
        hovertemplate="%{x|%d/%m/%Y}<br>DD: %{y:.2f}%<extra>HRP+BL</extra>"
    ))
    if ibov_d_dd is not None:
        ibov_dd_plot = ibov_d_dd.iloc[-n_dd:] if n_dd else ibov_d_dd
        fig_dd_d.add_trace(go.Scatter(
            x=ibov_dd_plot.index, y=ibov_dd_plot.values.round(2),
            name="Ibovespa (diário)", fill="tozeroy",
            line=dict(color="#E24B4A", width=1, dash="dot"),
            fillcolor="rgba(226,75,74,0.08)",
            hovertemplate="%{x|%d/%m/%Y}<br>DD: %{y:.2f}%<extra>Ibovespa</extra>"
        ))

    fig_dd_d.update_layout(
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        height=300, hovermode="x unified",
        font=dict(color="#1a1a18"),
        margin=dict(l=0,r=0,t=8,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(color="#1a1a18")),
        xaxis=dict(gridcolor="#e8e6e0",
                   tickfont=dict(color="#444441", size=10), color="#1a1a18"),
        yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0",
                   tickfont=dict(color="#444441", size=11), color="#1a1a18",
                   zeroline=True, zerolinecolor="#888780"),
    )
    st.plotly_chart(fig_dd_d, use_container_width=True)

    max_dd_d = port_d_dd.min()
    max_dd_d_data = port_d_dd.idxmin()
    st.caption(f"Max drawdown diário HRP+BL: **{max_dd_d:.2f}%** em "
               f"{max_dd_d_data.strftime('%d/%m/%Y')} · "
               f"vs mensal: {m_port['max_dd']*100:.2f}%")

    st.divider()

    # ── Bloco 4: Drift diário dos pesos ──────────────────────────────────────
    st.markdown("<div class='section-title'>drift dos pesos vs target HRP+BL</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#888780;margin-bottom:.75rem'>"
        "Estimativa do drift atual baseado na performance relativa de cada ativo "
        "desde o último rebalanceamento.</div>", unsafe_allow_html=True
    )

    # Calcular drift nos últimos 21 dias úteis (≈ 1 mês)
    n_drift = min(21, len(port_d_idx)-1)
    drift_rows = []
    total_drift_port = 0
    for cfg in ASSET_CFG:
        s = daily_series.get(cfg["name"])
        if s is None:
            drift_rows.append({
                "Ativo": cfg["name"], "Target": f"{cfg['w']*100:.1f}%",
                "Estimado": "—", "Drift": "—", "Status": "—"
            })
            continue
        retorno_mes = s["valor"].pct_change().reindex(port_d_idx).ffill().fillna(0)
        ret_ativo   = (1 + retorno_mes.iloc[-n_drift:]).prod() - 1
        ret_port    = (1 + port_d_ret.iloc[-n_drift:]).prod() - 1
        # Peso estimado após performance relativa
        w_novo = cfg["w"] * (1 + ret_ativo) / (1 + ret_port) if ret_port != -1 else cfg["w"]
        drift  = (w_novo - cfg["w"]) * 100
        banda  = 3.0  # banda padrão
        status = "✅ OK" if abs(drift) <= banda else ("⬆️ Acima" if drift > 0 else "⬇️ Abaixo")
        drift_rows.append({
            "Ativo":    cfg["name"],
            "Cluster":  cfg["cluster"],
            "Target":   f"{cfg['w']*100:.1f}%",
            "Estimado": f"{w_novo*100:.1f}%",
            "Drift":    f"{drift:+.2f}%",
            "Status":   status,
        })
    st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)
    st.caption("Drift estimado com base nos últimos 21 dias úteis. "
               "Para rebalanceamento preciso, use a aba Rebalanceamento com os pesos atuais reais.")



# ── Tab 10: Janelas móveis ────────────────────────────────────────────────────
with tab10:
    st.markdown(
        "Análise de consistência do portfólio em janelas móveis de 36 e 48 meses. "
        "Responde a pergunta: **em qualquer momento que eu tivesse entrado, "
        "qual a chance de ter batido o CDI e o IPCA?**"
    )

    # ── Controles ──────────────────────────────────────────────────────────────
    col_jm1, col_jm2, col_jm3 = st.columns(3)
    janela_meses = col_jm1.selectbox(
        "Janela de análise", [24, 36, 48, 60],
        index=1, format_func=lambda x: f"{x} meses ({x//12} anos)",
        key="janela_movel"
    )
    show_cust_jm = col_jm2.checkbox("Incluir portfólio customizado", value=True, key="jm_cust")
    benchmark_jm = col_jm3.selectbox("Benchmark principal", ["CDI", "IPCA"], key="jm_bench")

    # ── Calcular retornos em janelas móveis ────────────────────────────────────
    # Portfólio customizado
    custom_w_jm = {cfg["name"]: st.session_state.get(f"rebal_{cfg['name']}",
                   cfg["w"]*100)/100 for cfg in ASSET_CFG}
    total_jm = sum(custom_w_jm.values())
    custom_valid_jm = abs(total_jm - 1.0) < 0.02
    if custom_valid_jm:
        c_ret_jm = sum(
            custom_w_jm[a["name"]] * series[a["name"]]["valor"]
            .pct_change().dropna().reindex(common_idx).ffill()
            for a in ASSET_CFG
        )

    # 1/N igual
    eq_ret_jm = sum((1/6) * series[a["name"]]["valor"]
                    .pct_change().dropna().reindex(common_idx).ffill()
                    for a in ASSET_CFG)

    def rolling_acum(ret_series, n):
        """Retorno acumulado de cada janela de n meses."""
        cum = (1 + ret_series).cumprod()
        result = {}
        for i in range(n, len(cum)):
            end_date   = cum.index[i]
            start_date = cum.index[i - n]
            acum = (cum.iloc[i] / cum.iloc[i - n] - 1) * 100
            result[end_date] = round(acum, 2)
        return pd.Series(result)

    roll_port  = rolling_acum(port_ret,    janela_meses)
    roll_cdi   = rolling_acum(cdi_aligned, janela_meses)
    roll_ibov  = rolling_acum(ibov_ret,    janela_meses)
    roll_ipca  = rolling_acum(ipca6_ret,   janela_meses)
    roll_igual = rolling_acum(eq_ret_jm,   janela_meses)
    roll_cust  = rolling_acum(c_ret_jm, janela_meses) if custom_valid_jm else None

    bench_series = roll_cdi if benchmark_jm == "CDI" else roll_ipca
    bench_label  = "CDI" if benchmark_jm == "CDI" else "IPCA"

    # ── Bloco 1: KPIs de consistência ─────────────────────────────────────────
    st.markdown("<div class='section-title'>consistência vs benchmark</div>",
                unsafe_allow_html=True)

    def kpi_consistencia(roll, bench, nome, cor):
        idx_comum = roll.index.intersection(bench.index)
        r = roll.reindex(idx_comum)
        b = bench.reindex(idx_comum)
        n_total   = len(r)
        n_acima   = (r > b).sum()
        pct_acima = n_acima / n_total * 100 if n_total > 0 else 0
        melhor    = r.max()
        pior      = r.min()
        mediana   = r.median()
        return {
            "nome": nome, "cor": cor,
            "n_total": n_total, "n_acima": n_acima,
            "pct_acima": pct_acima,
            "melhor": melhor, "pior": pior, "mediana": mediana,
        }

    portfolios_jm = [
        (roll_port,  port_ret,  "HRP+BL",   "#378ADD"),
        (roll_igual, eq_ret_jm, "1/N igual", "#7F77DD"),
        (roll_ibov,  ibov_ret,  "Ibovespa",  "#BA7517"),
    ]
    if custom_valid_jm and roll_cust is not None and show_cust_jm:
        portfolios_jm.insert(1, (roll_cust, c_ret_jm, "Customizado", "#E24B4A"))

    stats_list = []
    for roll, _, nome, cor in portfolios_jm:
        stats_list.append(kpi_consistencia(roll, bench_series, nome, cor))

    cols_jm = st.columns(len(stats_list))
    for i, s in enumerate(stats_list):
        cls = "pos" if s["pct_acima"] >= 70 else "warn" if s["pct_acima"] >= 50 else "neg"
        with cols_jm[i]:
            st.markdown(
                f"<div class='metric-card' style='border-top:3px solid {s["cor"]}'>"
                f"<div class='metric-label'>{s["nome"]} vs {bench_label}</div>"
                f"<div class='metric-value {cls}'>{s["pct_acima"]:.0f}%</div>"
                f"<div class='metric-sub'>das janelas acima do {bench_label} "
                f"({s["n_acima"]}/{s["n_total"]})</div>"
                f"<div style='margin-top:8px;font-size:12px;color:#888780'>"
                f"Mediana: {s['mediana']:+.1f}% | "
                f"Melhor: {s['melhor']:+.1f}% | "
                f"Pior: {s['pior']:+.1f}%</div>"
                f"</div>", unsafe_allow_html=True
            )

    st.divider()

    # ── Bloco 2: Gráfico de retorno por janela ────────────────────────────────
    st.markdown("<div class='section-title'>retorno acumulado por janela móvel</div>",
                unsafe_allow_html=True)
    st.caption(f"Cada ponto = retorno acumulado dos {janela_meses} meses anteriores. "
               f"Acima da linha = bateu o {bench_label} naquela janela.")

    fig_jm = go.Figure()

    # Área do benchmark
    fig_jm.add_trace(go.Scatter(
        x=bench_series.index, y=bench_series.values,
        name=bench_label, fill="tozeroy",
        line=dict(color="#1D9E75" if benchmark_jm=="CDI" else "#C4770A",
                  width=1.5, dash="dot"),
        fillcolor="rgba(29,158,117,0.08)" if benchmark_jm=="CDI"
                  else "rgba(196,119,10,0.08)",
    ))

    cores_jm = {"HRP+BL":"#378ADD","Customizado":"#E24B4A",
                "1/N igual":"#7F77DD","Ibovespa":"#BA7517"}
    for roll, _, nome, cor in portfolios_jm:
        idx_c = roll.index.intersection(bench_series.index)
        fig_jm.add_trace(go.Scatter(
            x=roll.reindex(idx_c).index,
            y=roll.reindex(idx_c).values,
            name=nome, line=dict(color=cor, width=2),
            hovertemplate=f"<b>{nome}</b><br>%{{x|%b/%Y}}<br>"
                          f"Acum {janela_meses}m: %{{y:.1f}}%<extra></extra>",
        ))

    fig_jm.update_layout(
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        height=380, hovermode="x unified",
        font=dict(color="#1a1a18"),
        margin=dict(l=0,r=0,t=8,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(color="#1a1a18")),
        xaxis=dict(gridcolor="#e8e6e0",
                   tickfont=dict(color="#444441",size=11), color="#1a1a18"),
        yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0",
                   tickfont=dict(color="#444441",size=11), color="#1a1a18",
                   zeroline=True, zerolinecolor="#888780",
                   title=f"Retorno acumulado {janela_meses} meses (%)"),
    )
    st.plotly_chart(fig_jm, use_container_width=True)

    st.divider()

    # ── Bloco 3: Tabela estatística completa ──────────────────────────────────
    st.markdown("<div class='section-title'>estatísticas por portfólio</div>",
                unsafe_allow_html=True)

    tbl_rows = []
    for roll, ret_s, nome, cor in portfolios_jm:
        idx_c = roll.index.intersection(bench_series.index)
        r = roll.reindex(idx_c)
        b = bench_series.reindex(idx_c)
        n = len(r)
        tbl_rows.append({
            "Portfólio":           nome,
            f"% janelas > {bench_label}": f"{(r>b).mean()*100:.0f}%",
            "Mediana acum.":       f"{r.median():+.1f}%",
            "Melhor janela":       f"{r.max():+.1f}%",
            "Pior janela":         f"{r.min():+.1f}%",
            "Janelas positivas":   f"{(r>0).mean()*100:.0f}%",
            f"Excesso vs {bench_label}": f"{(r-b).median():+.1f}%",
            "Total janelas":       str(n),
        })

    df_jm = pd.DataFrame(tbl_rows).set_index("Portfólio")
    st.dataframe(df_jm, use_container_width=True)

    st.divider()

    # ── Bloco 4: Histograma do excesso de retorno vs benchmark ───────────────
    st.markdown("<div class='section-title'>distribuição do excesso de retorno vs benchmark</div>",
                unsafe_allow_html=True)
    st.caption(f"Distribuição do retorno do HRP+BL menos o {bench_label} "
               f"em cada janela de {janela_meses} meses. "
               f"Barras à direita do zero = janelas em que bateu o benchmark.")

    fig_hist_jm = go.Figure()
    for roll, _, nome, cor in portfolios_jm:
        idx_c = roll.index.intersection(bench_series.index)
        excesso = (roll.reindex(idx_c) - bench_series.reindex(idx_c)).dropna()
        fig_hist_jm.add_trace(go.Histogram(
            x=excesso.values, name=nome,
            nbinsx=30, opacity=0.65,
            marker_color=cor,
            hovertemplate=f"<b>{nome}</b><br>Excesso: %{{x:.1f}}%<br>"
                          f"Freq: %{{y}}<extra></extra>",
        ))
    fig_hist_jm.add_vline(x=0, line_color="#888780", line_width=2,
                           annotation_text="Zero", annotation_font_color="#888780")
    fig_hist_jm.update_layout(
        plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
        height=280, barmode="overlay",
        font=dict(color="#1a1a18"),
        margin=dict(l=0,r=0,t=8,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(color="#1a1a18")),
        xaxis=dict(gridcolor="#e8e6e0", ticksuffix="%",
                   tickfont=dict(color="#444441",size=11), color="#1a1a18",
                   title=f"Excesso de retorno vs {bench_label} (%)"),
        yaxis=dict(gridcolor="#e8e6e0",
                   tickfont=dict(color="#444441",size=11), color="#1a1a18",
                   title="Frequência"),
    )
    st.plotly_chart(fig_hist_jm, use_container_width=True)

    # Insight automático
    idx_c = roll_port.index.intersection(bench_series.index)
    excesso_port = (roll_port.reindex(idx_c) - bench_series.reindex(idx_c)).dropna()
    pct_acima_port = (excesso_port > 0).mean() * 100
    mediana_excesso = excesso_port.median()

    if pct_acima_port >= 70:
        st.success(
            f"✅ Em **{pct_acima_port:.0f}%** das janelas de {janela_meses} meses "
            f"o HRP+BL bateu o {bench_label}, com excesso mediano de "
            f"**{mediana_excesso:+.1f}%** no período."
        )
    elif pct_acima_port >= 50:
        st.info(
            f"📊 Em **{pct_acima_port:.0f}%** das janelas de {janela_meses} meses "
            f"o HRP+BL bateu o {bench_label}, com excesso mediano de "
            f"**{mediana_excesso:+.1f}%** no período."
        )
    else:
        st.warning(
            f"⚠️ Em apenas **{pct_acima_port:.0f}%** das janelas de {janela_meses} meses "
            f"o HRP+BL bateu o {bench_label}. "
            f"Considere revisar os pesos ou as visões do modelo."
        )



# ── Tab 11: Monte Carlo ───────────────────────────────────────────────────────
with tab11:
    st.markdown(
        "Simulação de Monte Carlo com **regimes de Markov** calibrados no histórico brasileiro. "
        "Projeta milhares de trajetórias possíveis para o portfólio nos próximos 5 anos, "
        "considerando transições entre regimes de juro alto, juro baixo e crise."
    )

    # ── Controles ─────────────────────────────────────────────────────────────
    col_mc1, col_mc2, col_mc3 = st.columns(3)
    n_sim    = col_mc1.selectbox("Número de simulações",
                                  [1000, 5000, 10000], index=1,
                                  format_func=lambda x: f"{x:,}")
    horizonte= col_mc2.selectbox("Horizonte (anos)",
                                  [1, 2, 3, 5], index=3)
    semente  = col_mc3.number_input("Semente aleatória", value=42,
                                     min_value=1, max_value=9999,
                                     help="Fixar para reproduzir resultados")

    # ── Conexão com aba Cenários ──────────────────────────────────────────────
    st.markdown("<div class='section-title'>conexão com cenários macro</div>",
                unsafe_allow_html=True)

    col_cn1, col_cn2 = st.columns(2)

    # Verificar se há retornos esperados salvos da aba Cenários
    usar_cenario_mc = col_cn1.checkbox(
        "Usar retornos esperados da aba Cenários como mu",
        value=True,
        help="Quando ativo, o Monte Carlo usa os retornos esperados do cenário macro "
             "definido na aba Cenários como média das simulações — tornando a projeção "
             "prospectiva em vez de puramente histórica."
    )

    # Opção de usar pesos BL Dinâmico
    usar_bl_mc = False
    if any(f"bl_dyn_{a['name']}" in st.session_state for a in ASSET_CFG):
        cenario_bl = st.session_state.get("bl_dyn_scenario", "—")
        usar_bl_mc = col_cn2.checkbox(
            f"Usar pesos BL Dinâmico (cenário: {cenario_bl})",
            value=True,
            help="Pesos calculados na aba Cenários → BL Dinâmico"
        )

    # Mostrar status da conexão
    if usar_cenario_mc:
        # Verificar se há retornos do cenário disponíveis no session_state
        cenario_nome = st.session_state.get("mc_cenario_nome", None)
        cenario_rets = st.session_state.get("mc_asset_rets", None)
        if cenario_rets:
            st.success(
                f"✅ Conectado ao cenário **{cenario_nome}** — "
                f"retornos esperados carregados para {len(cenario_rets)} ativos."
            )
        else:
            st.warning(
                "⚠️ Nenhum cenário calculado ainda. Vá à aba **Cenários**, "
                "defina os parâmetros macro e role até o fim da página — "
                "os retornos serão salvos automaticamente para o Monte Carlo."
            )
    else:
        st.info("📊 Usando retornos históricos (média 2009-2026) como mu das simulações.")

    st.markdown("""
    <style>
    div[data-testid="stButton"] button {
        background:#378ADD !important;
        color:#ffffff !important;
        border:none !important;
        font-weight:500 !important;
    }
    div[data-testid="stButton"] button:hover {
        background:#185FA5 !important;
        color:#ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)
    rodar_mc = st.button("▶ Rodar simulação Monte Carlo", key="btn_mc")

    if rodar_mc:
        with st.spinner(f"Rodando {n_sim:,} simulações de {horizonte} anos..."):
            try:
                np.random.seed(int(semente))
                n_meses = horizonte * 12

                # ── Retornos históricos mensais ───────────────────────────────
                ret_dict_mc = {}
                for cfg in ASSET_CFG:
                    r = series[cfg["name"]]["valor"].pct_change().dropna()
                    r = r.reindex(common_idx).ffill().dropna()
                    ret_dict_mc[cfg["name"]] = r
                ret_df_mc = pd.DataFrame(ret_dict_mc).dropna()

                # ── Pesos a simular ───────────────────────────────────────────
                w_hrp_mc = np.array([WEIGHTS.get(a["name"],0) for a in ASSET_CFG])
                ativos_mc = [a["name"] for a in ASSET_CFG]

                if usar_bl_mc:
                    w_bl_mc = np.array([
                        st.session_state.get(f"bl_dyn_{a}", WEIGHTS.get(a,0)*100)/100
                        for a in ativos_mc
                    ])
                    w_bl_mc = w_bl_mc / w_bl_mc.sum()
                else:
                    w_bl_mc = None

                # ── Regimes de Markov calibrados no histórico BR ──────────────
                # Regime 0: Juro alto (Selic > 12%) — maior vol, menor retorno ações
                # Regime 1: Juro baixo/neutro (Selic 6-12%) — retornos equilibrados
                # Regime 2: Crise (drawdown > 5% em 1 mês) — correlações aumentam
                regimes = {
                    0: {"nome": "Juro alto",   "prob_stay": 0.75, "peso_vol": 1.0,  "peso_ret": 0.90},
                    1: {"nome": "Juro neutro", "prob_stay": 0.70, "peso_vol": 0.85, "peso_ret": 1.05},
                    2: {"nome": "Crise",       "prob_stay": 0.40, "peso_vol": 1.80, "peso_ret": 0.60},
                }
                # Matriz de transição calibrada no histórico BR 2009-2026
                trans = np.array([
                    [0.75, 0.20, 0.05],  # De juro alto → permanece, vai neutro, vai crise
                    [0.15, 0.70, 0.15],  # De juro neutro
                    [0.10, 0.50, 0.40],  # De crise → sai rapidamente
                ])
                regime_atual = 0  # Brasil hoje está em juro alto

                # ── Parâmetros históricos por ativo ──────────────────────────
                cov_hist = ret_df_mc.cov().values         # covariância mensal

                # ── Mu: histórico ou cenário prospectivo ──────────────────────
                cenario_rets_mc = st.session_state.get("mc_asset_rets", None)

                if usar_cenario_mc and cenario_rets_mc:
                    # Converter retornos anuais do cenário para mensais
                    mu_cenario = np.array([
                        (1 + cenario_rets_mc.get(a, ret_df_mc[a].mean()*12)/100)**(1/12) - 1
                        for a in ativos_mc
                    ])
                    # Blend: 60% cenário + 40% histórico (evita extremos)
                    mu_hist_arr = ret_df_mc.mean().values
                    mu_hist = 0.60 * mu_cenario + 0.40 * mu_hist_arr
                    cenario_fonte = f"cenário '{st.session_state.get('mc_cenario_nome','—')}' (60%) + histórico (40%)"
                else:
                    mu_hist = ret_df_mc.mean().values
                    cenario_fonte = "histórico puro 2009-2026"

                # ── Simulação ────────────────────────────────────────────────
                def simular_portfolio(pesos, n_sim, n_meses):
                    resultados = np.ones((n_sim, n_meses + 1))
                    regimes_traj = []

                    for sim in range(n_sim):
                        regime = regime_atual
                        traj_regime = []
                        for t in range(n_meses):
                            # Transição de regime
                            regime = np.random.choice(3, p=trans[regime])
                            traj_regime.append(regime)

                            # Parâmetros do regime
                            r_cfg = regimes[regime]
                            mu_r  = mu_hist * r_cfg["peso_ret"]
                            cov_r = cov_hist * r_cfg["peso_vol"]**2

                            # Garantir covariância positiva definida
                            cov_r = cov_r + np.eye(len(ativos_mc)) * 1e-7

                            # Gerar retorno
                            ret_sim = np.random.multivariate_normal(mu_r, cov_r)
                            ret_port = float(pesos @ ret_sim)
                            resultados[sim, t+1] = resultados[sim, t] * (1 + ret_port)

                        regimes_traj.append(traj_regime)

                    return resultados, regimes_traj

                # Simular HRP original
                sim_hrp, _ = simular_portfolio(w_hrp_mc, n_sim, n_meses)

                # Simular BL Dinâmico se disponível
                sim_bl = None
                if w_bl_mc is not None:
                    sim_bl, _ = simular_portfolio(w_bl_mc, n_sim, n_meses)

                # Simular CDI (determinístico + leve variação)
                cdi_mensal = cdi_aligned.mean()
                sim_cdi = np.ones((n_sim, n_meses + 1))
                for t in range(n_meses):
                    variacao = np.random.normal(0, 0.001, n_sim)
                    sim_cdi[:, t+1] = sim_cdi[:, t] * (1 + cdi_mensal + variacao)

                # ── Calcular percentis ────────────────────────────────────────
                meses_eixo = list(range(n_meses + 1))

                def percentis(sim):
                    return {
                        "p10": np.percentile(sim, 10, axis=0),
                        "p25": np.percentile(sim, 25, axis=0),
                        "p50": np.percentile(sim, 50, axis=0),
                        "p75": np.percentile(sim, 75, axis=0),
                        "p90": np.percentile(sim, 90, axis=0),
                    }

                p_hrp = percentis(sim_hrp)
                p_cdi = percentis(sim_cdi)
                p_bl  = percentis(sim_bl) if sim_bl is not None else None

                # Converter para % de retorno acumulado
                def to_pct(p):
                    return {k: (v - 1) * 100 for k, v in p.items()}

                pp_hrp = to_pct(p_hrp)
                pp_cdi = to_pct(p_cdi)
                pp_bl  = to_pct(p_bl) if p_bl else None

                # ── Gráfico de leque ──────────────────────────────────────────
                st.markdown("<div class='section-title'>leque de trajetórias — retorno acumulado</div>",
                            unsafe_allow_html=True)

                fig_mc = go.Figure()

                # Faixa HRP P10-P90
                fig_mc.add_trace(go.Scatter(
                    x=meses_eixo + meses_eixo[::-1],
                    y=list(pp_hrp["p90"]) + list(pp_hrp["p10"])[::-1],
                    fill="toself", fillcolor="rgba(55,138,221,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="HRP+BL P10-P90", showlegend=True,
                ))
                # Mediana HRP
                fig_mc.add_trace(go.Scatter(
                    x=meses_eixo, y=pp_hrp["p50"],
                    name="HRP+BL (mediana)",
                    line=dict(color="#378ADD", width=2.5),
                ))

                # BL Dinâmico se disponível
                if pp_bl:
                    fig_mc.add_trace(go.Scatter(
                        x=meses_eixo + meses_eixo[::-1],
                        y=list(pp_bl["p90"]) + list(pp_bl["p10"])[::-1],
                        fill="toself", fillcolor="rgba(29,158,117,0.10)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="BL Dinâmico P10-P90", showlegend=True,
                    ))
                    fig_mc.add_trace(go.Scatter(
                        x=meses_eixo, y=pp_bl["p50"],
                        name="BL Dinâmico (mediana)",
                        line=dict(color="#1D9E75", width=2.5, dash="dash"),
                    ))

                # CDI projetado
                fig_mc.add_trace(go.Scatter(
                    x=meses_eixo, y=pp_cdi["p50"],
                    name="CDI projetado",
                    line=dict(color="#888780", width=1.5, dash="dot"),
                ))

                # Linha de referência zero
                fig_mc.add_hline(y=0, line_color="#888780", line_width=1)

                # Labels eixo X em anos
                tick_vals = [i*12 for i in range(horizonte+1)]
                tick_text = [f"Ano {i}" for i in range(horizonte+1)]

                fig_mc.update_layout(
                    plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
                    height=420, hovermode="x unified",
                    font=dict(color="#1a1a18"),
                    margin=dict(l=0,r=0,t=8,b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="left", x=0, font=dict(color="#1a1a18")),
                    xaxis=dict(gridcolor="#e8e6e0", tickvals=tick_vals,
                               ticktext=tick_text,
                               tickfont=dict(color="#444441",size=11), color="#1a1a18"),
                    yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0",
                               tickfont=dict(color="#444441",size=11), color="#1a1a18",
                               title="Retorno acumulado (%)"),
                )
                st.plotly_chart(fig_mc, use_container_width=True)

                # ── Tabela de probabilidades ──────────────────────────────────
                st.divider()
                st.markdown("<div class='section-title'>probabilidades por horizonte</div>",
                            unsafe_allow_html=True)

                horizontes_tbl = [h for h in [12,24,36,60] if h <= n_meses]
                prob_rows = []
                for h in horizontes_tbl:
                    ret_hrp_h  = (sim_hrp[:, h] - 1) * 100
                    ret_cdi_h  = (sim_cdi[:, h] - 1) * 100
                    ret_bl_h   = (sim_bl[:, h] - 1) * 100 if sim_bl is not None else None

                    row = {
                        "Horizonte": f"{h} meses ({h//12} ano{'s' if h//12>1 else ''})",
                        "HRP+BL mediana":    f"{np.median(ret_hrp_h):+.1f}%",
                        "HRP+BL P10/P90":   f"{np.percentile(ret_hrp_h,10):+.1f}% / {np.percentile(ret_hrp_h,90):+.1f}%",
                        "P(bater CDI)":     f"{(ret_hrp_h > ret_cdi_h).mean()*100:.0f}%",
                        "P(bater IPCA)":    f"{(ret_hrp_h > ipca*h/12).mean()*100:.0f}%",
                        "P(DD > 5%)":       f"{(ret_hrp_h < -5).mean()*100:.0f}%",
                        "P(DD > 10%)":      f"{(ret_hrp_h < -10).mean()*100:.0f}%",
                    }
                    if ret_bl_h is not None:
                        row["BL Dinâmico mediana"] = f"{np.median(ret_bl_h):+.1f}%"
                        row["P(BL > HRP)"]         = f"{(ret_bl_h > ret_hrp_h).mean()*100:.0f}%"
                    prob_rows.append(row)

                df_mc_prob = pd.DataFrame(prob_rows).set_index("Horizonte")
                st.dataframe(df_mc_prob, use_container_width=True)

                # ── Histograma retorno final ──────────────────────────────────
                st.divider()
                st.markdown(f"<div class='section-title'>distribuição do retorno acumulado em {horizonte} anos</div>",
                            unsafe_allow_html=True)

                ret_final_hrp = (sim_hrp[:, -1] - 1) * 100
                ret_final_cdi = (sim_cdi[:, -1] - 1) * 100

                fig_hist_mc = go.Figure()
                fig_hist_mc.add_trace(go.Histogram(
                    x=ret_final_hrp, name="HRP+BL",
                    nbinsx=50, opacity=0.7,
                    marker_color="rgba(55,138,221,0.7)",
                ))
                if sim_bl is not None:
                    ret_final_bl = (sim_bl[:, -1] - 1) * 100
                    fig_hist_mc.add_trace(go.Histogram(
                        x=ret_final_bl, name="BL Dinâmico",
                        nbinsx=50, opacity=0.6,
                        marker_color="rgba(29,158,117,0.7)",
                    ))
                fig_hist_mc.add_vline(
                    x=float(np.median(ret_final_cdi)),
                    line_color="#888780", line_width=2, line_dash="dot",
                    annotation_text=f"CDI mediano: {np.median(ret_final_cdi):.1f}%",
                    annotation_font_color="#888780",
                )
                fig_hist_mc.update_layout(
                    plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
                    height=300, barmode="overlay",
                    font=dict(color="#1a1a18"),
                    margin=dict(l=0,r=0,t=8,b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="left", x=0, font=dict(color="#1a1a18")),
                    xaxis=dict(gridcolor="#e8e6e0", ticksuffix="%",
                               tickfont=dict(color="#444441",size=11), color="#1a1a18",
                               title=f"Retorno acumulado em {horizonte} anos (%)"),
                    yaxis=dict(gridcolor="#e8e6e0",
                               tickfont=dict(color="#444441",size=11), color="#1a1a18",
                               title="Frequência"),
                )
                st.plotly_chart(fig_hist_mc, use_container_width=True)

                # ── Insight automático ────────────────────────────────────────
                p_bate_cdi = (ret_final_hrp > ret_final_cdi).mean() * 100
                mediana_exc = float(np.median(ret_final_hrp - ret_final_cdi))
                st.caption(
                    f"Baseado em {n_sim:,} simulações com regimes de Markov calibrados "
                    f"no histórico BR 2009-2026. "
                    f"Regime de partida: juro alto (Selic atual 14.5%). "
                    f"Retornos esperados (mu): {cenario_fonte}."
                )
                if p_bate_cdi >= 60:
                    st.success(
                        f"✅ Em **{p_bate_cdi:.0f}%** das simulações o HRP+BL supera o CDI "
                        f"em {horizonte} anos, com excesso mediano de **{mediana_exc:+.1f}%**."
                    )
                else:
                    st.info(
                        f"📊 Em **{p_bate_cdi:.0f}%** das simulações o HRP+BL supera o CDI "
                        f"em {horizonte} anos, com excesso mediano de **{mediana_exc:+.1f}%**. "
                        f"Ambiente de juro alto desafia portfólios diversificados vs CDI."
                    )

            except Exception as e:
                st.error(f"Erro na simulação: {e}")
                import traceback
                st.code(traceback.format_exc())

    else:
        st.info("Configure os parâmetros acima e clique em **▶ Rodar simulação** para ver os resultados.")
        st.markdown(
            "<div style='font-size:13px;color:#888780;margin-top:1rem'>"
            "💡 Dica: rode primeiro o BL Dinâmico na aba Cenários para comparar "
            "HRP+BL original vs pesos otimizados pelo cenário macro."
            "</div>", unsafe_allow_html=True
        )



# ── Tab 12: Atribuição de retorno ─────────────────────────────────────────────
with tab12:
    st.markdown(
        "Decomposição do retorno mensal do portfólio HRP+BL por ativo — "
        "mostra quanto cada ativo contribuiu para o resultado em cada período."
    )

    # ── Controles ──────────────────────────────────────────────────────────────
    col_at1, col_at2 = st.columns(2)
    periodos_at2 = {
        "12 meses": 12, "24 meses": 24, "36 meses": 36,
        "5 anos": 60, "Histórico completo": None,
    }
    periodo_at2 = col_at1.selectbox("Período", list(periodos_at2.keys()),
                                     index=4, key="periodo_atrib")
    tipo_viz = col_at2.selectbox("Visualização",
                                  ["Contribuição mensal empilhada",
                                   "Contribuição acumulada",
                                   "Heatmap mensal"],
                                  key="tipo_atrib")

    # Filtrar período
    n_at2 = periodos_at2[periodo_at2]
    if n_at2:
        idx_at2 = common_idx[-n_at2:]
    else:
        idx_at2 = common_idx

    # ── Calcular contribuição de cada ativo ───────────────────────────────────
    contrib_dict = {}
    for cfg in ASSET_CFG:
        ret_a = (series[cfg["name"]]["valor"]
                 .pct_change().dropna()
                 .reindex(idx_at2).ffill().fillna(0))
        contrib_dict[cfg["name"]] = ret_a * cfg["w"] * 100  # contribuição em %

    df_contrib = pd.DataFrame(contrib_dict)
    df_contrib.index = pd.to_datetime(df_contrib.index)
    df_contrib["Total"] = df_contrib.sum(axis=1)

    # ── Gráfico principal ─────────────────────────────────────────────────────
    cores_ativos = {cfg["name"]: cfg["color"] for cfg in ASSET_CFG}

    if tipo_viz == "Contribuição mensal empilhada":
        st.markdown("<div class='section-title'>contribuição mensal por ativo (%)</div>",
                    unsafe_allow_html=True)

        fig_atrib = go.Figure()
        for cfg in ASSET_CFG:
            fig_atrib.add_trace(go.Bar(
                x=df_contrib.index,
                y=df_contrib[cfg["name"]].round(3),
                name=cfg["name"],
                marker_color=cfg["color"],
                hovertemplate=f"<b>{cfg['name']}</b><br>%{{x|%b/%Y}}<br>"
                              f"Contribuição: %{{y:.3f}}%<extra></extra>",
            ))
        # Linha do total
        fig_atrib.add_trace(go.Scatter(
            x=df_contrib.index,
            y=df_contrib["Total"].round(3),
            name="Total portfólio",
            line=dict(color="#1a1a18", width=2),
            hovertemplate="<b>Total</b><br>%{x|%b/%Y}<br>%{y:.3f}%<extra></extra>",
        ))
        fig_atrib.add_hline(y=0, line_color="#888780", line_width=1)
        fig_atrib.update_layout(
            plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
            height=400, barmode="relative", hovermode="x unified",
            font=dict(color="#1a1a18"),
            margin=dict(l=0,r=0,t=8,b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(color="#1a1a18")),
            xaxis=dict(gridcolor="#e8e6e0",
                       tickfont=dict(color="#444441", size=10), color="#1a1a18"),
            yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0",
                       tickfont=dict(color="#444441", size=11), color="#1a1a18",
                       zeroline=True, zerolinecolor="#888780",
                       title="Contribuição mensal (%)"),
        )
        st.plotly_chart(fig_atrib, use_container_width=True)

    elif tipo_viz == "Contribuição acumulada":
        st.markdown("<div class='section-title'>contribuição acumulada por ativo (%)</div>",
                    unsafe_allow_html=True)

        df_acum = df_contrib[list(cores_ativos.keys())].cumsum()

        fig_acum = go.Figure()
        for cfg in ASSET_CFG:
            # Converter hex para rgba para compatibilidade com Plotly
            h = cfg["color"].lstrip("#")
            r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            fill_rgba = f"rgba({r},{g},{b},0.35)"
            fig_acum.add_trace(go.Scatter(
                x=df_acum.index,
                y=df_acum[cfg["name"]].round(2),
                name=cfg["name"],
                fill="tonexty" if cfg != ASSET_CFG[0] else "tozeroy",
                line=dict(color=cfg["color"], width=1),
                fillcolor=fill_rgba,
                hovertemplate=f"<b>{cfg['name']}</b><br>%{{x|%b/%Y}}<br>"
                              f"Contrib. acum.: %{{y:.2f}}%<extra></extra>",
            ))
        fig_acum.update_layout(
            plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
            height=400, hovermode="x unified",
            font=dict(color="#1a1a18"),
            margin=dict(l=0,r=0,t=8,b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(color="#1a1a18")),
            xaxis=dict(gridcolor="#e8e6e0",
                       tickfont=dict(color="#444441", size=10), color="#1a1a18"),
            yaxis=dict(ticksuffix="%", gridcolor="#e8e6e0",
                       tickfont=dict(color="#444441", size=11), color="#1a1a18",
                       title="Contribuição acumulada (%)"),
        )
        st.plotly_chart(fig_acum, use_container_width=True)

    else:  # Heatmap mensal
        st.markdown("<div class='section-title'>heatmap de contribuição mensal por ativo</div>",
                    unsafe_allow_html=True)

        # Preparar dados para heatmap
        df_heat = df_contrib[list(cores_ativos.keys())].copy()
        df_heat.index = df_heat.index.strftime("%b/%Y")

        # Limitar a 36 meses para legibilidade
        df_heat = df_heat.tail(min(36, len(df_heat)))

        fig_heat = go.Figure(go.Heatmap(
            z=df_heat.values.T,
            x=df_heat.index.tolist(),
            y=list(cores_ativos.keys()),
            colorscale=[
                [0.0, "#A32D2D"],
                [0.4, "#FCEBEB"],
                [0.5, "#f8f7f4"],
                [0.6, "#EAF3DE"],
                [1.0, "#0F6E56"],
            ],
            zmid=0,
            hovertemplate="<b>%{y}</b><br>%{x}<br>Contribuição: %{z:.3f}%<extra></extra>",
            colorbar=dict(
                title="Contrib. (%)",
                tickfont=dict(color="#444441"),
            ),
        ))
        fig_heat.update_layout(
            plot_bgcolor="#f8f7f4", paper_bgcolor="#f8f7f4",
            height=320,
            font=dict(color="#1a1a18"),
            margin=dict(l=0,r=0,t=8,b=0),
            xaxis=dict(tickfont=dict(color="#444441", size=9), color="#1a1a18",
                       tickangle=-45),
            yaxis=dict(tickfont=dict(color="#444441", size=11), color="#1a1a18"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Verde = contribuição positiva | Vermelho = contribuição negativa | "
                   "Limitado aos últimos 36 meses para legibilidade.")

    st.divider()

    # ── Tabela de contribuição média por ativo ────────────────────────────────
    st.markdown("<div class='section-title'>contribuição média e total por ativo no período</div>",
                unsafe_allow_html=True)

    rows_contrib = []
    for cfg in ASSET_CFG:
        c = df_contrib[cfg["name"]]
        rows_contrib.append({
            "Ativo":             cfg["name"],
            "Cluster":           cfg["cluster"],
            "Peso HRP+BL":       f"{cfg['w']*100:.1f}%",
            "Contrib. média/mês":f"{c.mean():+.4f}%",
            "Contrib. total":    f"{c.sum():+.2f}%",
            "Meses positivos":   f"{(c>0).sum()} ({(c>0).mean()*100:.0f}%)",
            "Meses negativos":   f"{(c<0).sum()} ({(c<0).mean()*100:.0f}%)",
            "Maior contrib.":    f"{c.max():+.3f}%",
            "Menor contrib.":    f"{c.min():+.3f}%",
        })

    # Linha de total
    total_c = df_contrib["Total"]
    rows_contrib.append({
        "Ativo":             "TOTAL",
        "Cluster":           "—",
        "Peso HRP+BL":       "100%",
        "Contrib. média/mês":f"{total_c.mean():+.4f}%",
        "Contrib. total":    f"{total_c.sum():+.2f}%",
        "Meses positivos":   f"{(total_c>0).sum()} ({(total_c>0).mean()*100:.0f}%)",
        "Meses negativos":   f"{(total_c<0).sum()} ({(total_c<0).mean()*100:.0f}%)",
        "Maior contrib.":    f"{total_c.max():+.3f}%",
        "Menor contrib.":    f"{total_c.min():+.3f}%",
    })

    df_tbl_contrib = pd.DataFrame(rows_contrib).set_index("Ativo")
    st.dataframe(df_tbl_contrib, use_container_width=True)

    st.divider()

    # ── Insight: qual ativo mais contribuiu ───────────────────────────────────
    contribuicoes_totais = {
        cfg["name"]: df_contrib[cfg["name"]].sum()
        for cfg in ASSET_CFG
    }
    maior_contrib = max(contribuicoes_totais, key=contribuicoes_totais.get)
    menor_contrib = min(contribuicoes_totais, key=contribuicoes_totais.get)
    cor_maior = cores_ativos[maior_contrib]
    cor_menor = cores_ativos[menor_contrib]

    col_ins1, col_ins2 = st.columns(2)
    col_ins1.markdown(
        f"<div class='metric-card' style='border-top:3px solid {cor_maior}'>"
        f"<div class='metric-label'>Maior contribuidor no período</div>"
        f"<div class='metric-value pos' style='font-size:22px'>{maior_contrib}</div>"
        f"<div class='metric-sub'>Contribuição total: "
        f"{contribuicoes_totais[maior_contrib]:+.2f}%</div>"
        f"</div>", unsafe_allow_html=True
    )
    col_ins2.markdown(
        f"<div class='metric-card' style='border-top:3px solid {cor_menor}'>"
        f"<div class='metric-label'>Menor contribuidor no período</div>"
        f"<div class='metric-value warn' style='font-size:22px'>{menor_contrib}</div>"
        f"<div class='metric-sub'>Contribuição total: "
        f"{contribuicoes_totais[menor_contrib]:+.2f}%</div>"
        f"</div>", unsafe_allow_html=True
    )


# ── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style='text-align:center;font-size:12px;color:#b4b2a9;padding:.5rem 0'>
    HRP + Black-Litterman Dashboard &nbsp;·&nbsp;
    Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')} &nbsp;·&nbsp;
    CDI rf = {rf_ann*100:.2f}% a.a. ({common_idx[0].strftime('%b/%Y')}–{common_idx[-1].strftime('%b/%Y')})
</div>
""", unsafe_allow_html=True)
