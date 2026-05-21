# Projeto Aplicado III - Etapa 3

Código simples em Python para a terceira entrega do projeto de recomendação de receitas.

## Como organizar os arquivos

Na raiz do repositório, deixe a seguinte estrutura:

```text
Projeto_Apli_3/
├── data/
│   ├── interactions_train.csv
│   ├── interactions_validation.csv
│   ├── interactions_test.csv
│   └── PP_recipes.csv
├── etapa3_recomendacao_receitas.py
└── requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt
python etapa3_recomendacao_receitas.py
```

## O que o código faz

1. Carrega os arquivos de treino, validação, teste e receitas processadas.
2. Reavalia a previsão de notas com RMSE e MAE.
3. Ajusta o pipeline da etapa anterior usando uma recomendação híbrida simples.
4. Usa ingredientes das receitas com TF-IDF e uma pontuação de popularidade.
5. Avalia listas de recomendação com Precision@10, Recall@10 e HitRate@10.

## Observação

A avaliação Top-K é feita com candidatos amostrados, pois a base é muito esparsa. Isso evita que a métrica fique sempre zerada quando o modelo precisa escolher 10 receitas entre mais de cem mil opções possíveis.
