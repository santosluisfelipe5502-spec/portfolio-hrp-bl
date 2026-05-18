HRP + Black-Litterman Dashboard
Dashboard de gestão de portfólio usando as metodologias Hierarchical Risk Parity (HRP)
de López de Prado (2016) e Black-Litterman (BL) de Black & Litterman (1992).
Acesso
🔗 Dashboard: https://portfolio-hrp-bl.streamlit.app
Metodologia
HRP distribui o risco hierarquicamente entre clusters de ativos correlacionados,
sem precisar inverter a matriz de covariância (mais robusto que Risk Parity clássico).
Black-Litterman combina retornos de equilíbrio históricos com visões táticas do
investidor usando estatística Bayesiana — produzindo retornos esperados mais estáveis.
Ativos
Ativo	Cluster	Peso HRP+BL	Fonte
IRF-M	Renda fixa	26.8%	ANBIMA
IMA-Geral	Renda fixa	18.8%	ANBIMA
IHFA	Âncora	17.2%	ANBIMA
IDA-DI	Âncora	14.4%	ANBIMA
Ibovespa	Equity	14.5%	Yahoo Finance
Internacional (SPY+TLT)	Equity	8.3%	Yahoo Finance
Perfis de Risco
Perfil	Vol alvo	Retorno esperado
Conservador	0.5% – 1.0% a.a.	CDI + 0.3% a 0.8%
Moderado	2.0% – 3.5% a.a.	CDI + 1% a 2%
Agressivo	acima de 5% a.a.	CDI + 2% a 4%
Atualização dos dados
Automático: CDI, PTAX, IPCA via BCB · Ibovespa, SPY, TLT via Yahoo Finance
Manual (mensal): Baixar da ANBIMA e subir no GitHub:
`IRFM.xls` — IRF-M
`IMA.xls` — IMA-Geral
`IHFA.xls` — IHFA
`IDADI.xls` — IDA-DI
Estrutura do código
Todo o código está em `app.py`. O arquivo tem um índice comentado no topo
explicando onde fica cada seção. Principais pontos de manutenção:
Adicionar ativo: `ASSET_CFG` + `REPO_FILES` + `PERFIS`
Adicionar evento de cauda: `TAIL_EVENTS`
Alterar perfis de risco: `PERFIS` (vol_min, vol_max, bandas)
Alterar pesos: botão "Recalcular HRP" na aba Rebalanceamento
Dependências
```
streamlit >= 1.35.0
pandas >= 2.0.0
numpy >= 1.24.0
plotly >= 5.18.0
scipy >= 1.11.0
requests >= 2.31.0
openpyxl >= 3.1.0
xlrd >= 2.0.0
yfinance >= 0.2.40
reportlab >= 4.0.0
kaleido >= 0.2.0
```
Referências
López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample
Black, F. & Litterman, R. (1992). Global Portfolio Optimization
