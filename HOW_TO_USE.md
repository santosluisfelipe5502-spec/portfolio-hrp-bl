Como usar e manter o dashboard
Rotina mensal (15 minutos)
1. Atualizar bases ANBIMA
Acesse anbima.com.br/indices
Baixe os 4 índices: IRF-M, IMA-Geral, IHFA, IDA-DI
Renomeie: `IRFM.xls`, `IMA.xls`, `IHFA.xls`, `IDADI.xls`
Suba no GitHub: clique em cada arquivo → lápis → cola → Commit
2. Verificar dashboard
Abra o dashboard
Confira a data de atualização no header
Verifique se há alertas de rebalanceamento no topo
---
Rotina semestral (30 minutos)
1. Recalcular pesos HRP
Vá na aba 🔄 Rebalanceamento
Clique em "Recalcular pesos HRP com dados recentes"
Selecione janela de 36 meses
Anote os novos pesos sugeridos
Se fizer sentido, atualize `ASSET_CFG` no código com os novos pesos
2. Revisar cenários
Vá na aba 🌐 Cenários
Atualize Selic, IPCA, PIB e câmbio com as expectativas atuais
Rode o BL Dinâmico
Se quiser aplicar os novos pesos, clique em "Aplicar pesos BL ao portfólio"
3. Atualizar câmbio base
Se o dólar mudou significativamente (>10%), atualize no `app.py`:
Busque por `# base = PTAX` e altere o valor
---
Como adicionar um novo ativo
No `app.py`, edite 3 locais:
1. ASSET_CFG (linha ~124)
```python
{"name": "NOME",  "key": "CHAVE",  "color": "#HEXCOR",
 "cluster": "Cluster", "w": 0.XX, "vol": 0.XX},
```
2. REPO_FILES (linha ~700)
```python
"NOME": "NOMEARQUIVO",  # sem extensão
```
3. PERFIS (linha ~448)
Adicione o ativo em cada perfil com as bandas de peso:
```python
"NOME": (peso_min, peso_max),
```
No GitHub:
Suba o arquivo do novo ativo (ex: `NOVOATIVO.xls`)
---
Como adicionar um evento de cauda
No `app.py`, encontre `TAIL_EVENTS` (linha ~1121) e adicione:
```python
{"name": "Nome do evento",
 "start": "YYYY-MM-DD", "end": "YYYY-MM-DD",
 "color": "#HEXCOR",
 "desc": "Descrição do evento.",
 "tipo": "Doméstico"},  # ou "Global"
```
---
Como alterar os perfis de risco
No `app.py`, encontre `PERFIS` (linha ~448):
```python
"Conservador": {
    "vol_min": 0.5,   # volatilidade mínima alvo (% a.a.)
    "vol_max": 1.0,   # volatilidade máxima alvo (% a.a.)
    "bandas": {
        "IRF-M":    (0.0, 8.0),   # (peso_min%, peso_max%)
        "IDA-DI":   (70.0, 95.0),
        ...
    }
}
```
---
Solução de problemas comuns
Dashboard não atualiza após subir arquivo:
Aguarde 30-60 segundos e recarregue a página
Verifique se o nome do arquivo está exatamente correto (maiúsculas/minúsculas)
Dados simulados aparecendo:
O arquivo ANBIMA não foi encontrado no repositório
Verifique o nome do arquivo em `REPO_FILES` no código
Erro ao calcular perfil:
O algoritmo HRP não convergiu com as restrições definidas
Afrouxe as bandas de peso em `PERFIS` para o perfil problemático
CDI não carrega:
API do Banco Central pode estar instável
O sistema usa automaticamente o CDI histórico como fallback
