# HRP + Black-Litterman Dashboard

Dashboard interativo de monitoramento de portfólio com dados reais, construído com Streamlit.

## Funcionalidades

- **Retorno acumulado** vs CDI, Ibovespa e 1/N igual
- **Drawdown histórico** comparativo
- **Métricas completas**: Sharpe, Sortino, Calmar, VaR (rf = CDI real via BCB)
- **Rebalanceamento**: detecção automática de drift com banda configurável
- **Cenários macroeconômicos**: Selic, IPCA, PIB, câmbio
- **CDI automático** via API pública do Banco Central do Brasil
- **Upload de dados** para IHFA, IDA Pré, IMA, IDA-Geral, Ibovespa, SPY, TLT

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Acesse em: http://localhost:8501

## Deploy gratuito no Streamlit Cloud

1. Faça fork deste repositório no GitHub
2. Acesse https://share.streamlit.io
3. Clique em "New app" → selecione seu repositório
4. Defina `app.py` como arquivo principal
5. Clique em "Deploy" — pronto, você terá um link público

O CDI é buscado automaticamente via API do Banco Central a cada 24h.

## Deploy no Railway (alternativa)

```bash
# Instale Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
```

## Atualização dos dados

### CDI (automático)
Buscado via `https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados?formato=json`
Atualiza automaticamente a cada 24h com `@st.cache_data(ttl=86400)`.

### Índices (manual via upload)
Na sidebar, faça upload dos arquivos Excel/CSV no formato ANBIMA:
- Coluna 1: código do índice
- Coluna 2: data (DD/MM/AAAA)
- Coluna 3: valor do índice

### Internacional (SPY + TLT)
Faça upload separado de SPY e TLT. O app combina com os pesos configurados (padrão 40%/60%).

## Estrutura dos pesos HRP+BL

| Ativo      | Cluster     | Peso   |
|------------|-------------|--------|
| IDA Pré    | Renda fixa  | 26.8%  |
| IMA        | Renda fixa  | 18.8%  |
| IHFA       | Âncora      | 17.2%  |
| IDA-Geral  | Âncora      | 14.4%  |
| Ibovespa   | Equity      | 14.5%  |
| Internac.  | Equity      |  8.3%  |

## Taxa livre de risco

CDI mensal real (série 4391 do SGS/BCB). Sharpe e Sortino calculados com rf variável mês a mês.
