import ast
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PASTA_DADOS = Path('/mnt/data') if Path('/mnt/data/RAW_recipes.csv').exists() else Path('data')
PASTA_SAIDA = Path('outputs')
PASTA_SAIDA.mkdir(exist_ok=True)

receitas = pd.read_csv(PASTA_DADOS / 'RAW_recipes.csv', nrows=50000)
treino = pd.read_csv(PASTA_DADOS / 'interactions_train.csv')
validacao = pd.read_csv(PASTA_DADOS / 'interactions_validation.csv')

# Distribuicao das notas
plt.figure(figsize=(7,4))
validacao['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribuicao das notas na validacao')
plt.xlabel('Nota')
plt.ylabel('Quantidade de interacoes')
plt.tight_layout()
plt.savefig(PASTA_SAIDA / 'grafico_distribuicao_notas.png', dpi=150)
plt.close()

# RMSE e MAE apurados no pipeline
resultados = pd.DataFrame({
    'modelo': ['Media geral', 'Media por receita', 'Media por usuario', 'Media usuario + receita'],
    'RMSE': [1.346776, 1.346776, 1.290494, 1.277964],
    'MAE': [0.858343, 0.858343, 0.796193, 0.811911]
})
plt.figure(figsize=(8,4))
plt.plot(resultados['modelo'], resultados['RMSE'], marker='o', label='RMSE')
plt.plot(resultados['modelo'], resultados['MAE'], marker='o', label='MAE')
plt.title('Comparacao das linhas de base')
plt.xlabel('Modelo')
plt.ylabel('Erro')
plt.xticks(rotation=20, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(PASTA_SAIDA / 'grafico_metricas_erro.png', dpi=150)
plt.close()

# Tempo de preparo
receitas_validas = receitas[receitas['minutes'].fillna(0) > 0].copy()
limite = receitas_validas['minutes'].quantile(0.99)

plt.figure(figsize=(7,4))
receitas_validas['minutes'].clip(upper=limite).hist(bins=40)
plt.title('Distribuicao do tempo de preparo tratado')
plt.xlabel('Minutos')
plt.ylabel('Quantidade de receitas')
plt.tight_layout()
plt.savefig(PASTA_SAIDA / 'grafico_tempo_preparo.png', dpi=150)
plt.close()

# Grafico simples dos resultados Top-K
plt.figure(figsize=(6,4))
plt.bar(['Precision@10', 'Recall@10'], [0.0, 0.0])
plt.ylim(0, 1)
plt.title('Resultado Top-K na avaliacao offline')
plt.ylabel('Valor')
plt.tight_layout()
plt.savefig(PASTA_SAIDA / 'grafico_topk.png', dpi=150)
plt.close()

print('Graficos gerados na pasta outputs')
