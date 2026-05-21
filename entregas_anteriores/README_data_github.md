# Projeto Aplicado III — Sistema de Recomendação de Receitas

Este repositório contém a prova de conceito da Etapa 2 do Projeto Aplicado III.

## Organização dos arquivos

Os dados devem ficar dentro da pasta `data` do repositório:

```text
seu-repositorio/
├── data/
│   ├── RAW_recipes.csv
│   ├── RAW_interactions.csv
│   ├── interactions_train.csv
│   ├── interactions_validation.csv
│   ├── interactions_test.csv
│   ├── PP_recipes.csv
│   ├── PP_users.csv
│   └── ingr_map.pkl
├── poc_recomendador_receitas_data_github.ipynb
├── requirements.txt
└── README.md
```

## Importação dos dados no notebook

O notebook usa caminhos relativos com `Path('data')`, então ele funciona no GitHub ou no computador local desde que a pasta `data` esteja no mesmo nível do notebook.

```python
from pathlib import Path

pasta_dados = Path('data')

caminho_receitas = pasta_dados / 'RAW_recipes.csv'
caminho_interacoes = pasta_dados / 'RAW_interactions.csv'
caminho_treino = pasta_dados / 'interactions_train.csv'
caminho_validacao = pasta_dados / 'interactions_validation.csv'
caminho_teste = pasta_dados / 'interactions_test.csv'

receitas = pd.read_csv(caminho_receitas)
interacoes = pd.read_csv(caminho_interacoes)
treino = pd.read_csv(caminho_treino)
validacao = pd.read_csv(caminho_validacao)
teste = pd.read_csv(caminho_teste)
```

## Como executar

```bash
pip install -r requirements.txt
jupyter notebook poc_recomendador_receitas_data_github.ipynb
```

## Observação

Como a base é grande, algumas etapas podem usar amostragem para facilitar a execução em computadores comuns.
