# Projeto Aplicado III — Sistema de Recomendação de Receitas

Este repositório contém a prova de conceito da Etapa 2 do Projeto Aplicado III.

O objetivo do projeto é desenvolver um sistema inteligente de recomendação de receitas, alinhado ao ODS 12 — Consumo e Produção Responsáveis. A proposta é recomendar receitas com base nas características dos alimentos, ingredientes e interações dos usuários, contribuindo para o planejamento alimentar e para a redução do desperdício doméstico.

## Tema do projeto

**Sistema inteligente de recomendação para redução do desperdício alimentar alinhado ao ODS 12.**

## Base de dados

A base utilizada é a **Food.com Recipes and Interactions**, disponível publicamente no Kaggle.

Arquivos utilizados na prova de conceito:

- `RAW_recipes.csv`: contém informações das receitas, como nome, ingredientes, tags, tempo de preparo e descrição.
- `RAW_interactions.csv`: contém interações dos usuários com as receitas, incluindo avaliações e comentários.
- `interactions_train.csv`: conjunto de interações para treinamento.
- `interactions_validation.csv`: conjunto de interações para validação.
- `interactions_test.csv`: conjunto de interações para teste.
- `PP_recipes.csv`: dados pré-processados das receitas.
- `PP_users.csv`: dados pré-processados dos usuários.
- `ingr_map.pkl`: mapeamento de ingredientes.

## Objetivo da prova de conceito

A prova de conceito tem como objetivo demonstrar a viabilidade inicial de um sistema de recomendação baseado em conteúdo.

Nesta etapa, o modelo utiliza informações textuais das receitas, principalmente ingredientes e tags, para encontrar receitas semelhantes entre si.

## Técnica utilizada

A técnica inicial utilizada foi:

- **TF-IDF** para transformar ingredientes e tags em vetores numéricos;
- **Similaridade do cosseno** para medir a proximidade entre receitas;
- **KNN / Nearest Neighbors** para encontrar as receitas mais parecidas.

Essa abordagem representa uma recomendação baseada em conteúdo, pois recomenda receitas semelhantes às características de uma receita de referência.

## Etapas do notebook

O notebook foi organizado de forma passo a passo:

1. Importação das bibliotecas;
2. Carregamento dos dados;
3. Análise exploratória inicial;
4. Verificação de valores ausentes;
5. Análise da distribuição das notas;
6. Análise do tempo de preparo;
7. Cálculo da esparsidade da matriz usuário-item;
8. Tratamento e preparação dos dados;
9. Criação do texto com ingredientes e tags;
10. Vetorização com TF-IDF;
11. Treinamento do modelo de recomendação;
12. Geração de recomendações;
13. Avaliação com linha de base;
14. Cálculo de RMSE, MAE, Precision@K e Recall@K.

## Bibliotecas utilizadas

As principais bibliotecas utilizadas são:

- `pandas`: leitura, manipulação e análise dos dados;
- `numpy`: operações numéricas;
- `scikit-learn`: vetorização TF-IDF, modelo de vizinhos mais próximos e métricas;
- `scipy`: suporte para matrizes esparsas;
- `matplotlib` e `seaborn`: visualização dos dados;
- `jupyter`: execução do notebook.

## Como executar o projeto

### 1. Criar um ambiente virtual

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

No Linux ou Mac:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Abrir o Jupyter Notebook

```bash
jupyter notebook
```

Depois, abra o arquivo:

```text
poc_recomendador_receitas_PASSO_A_PASSO_RODADO.ipynb
```

## Observação sobre os caminhos dos arquivos

No notebook, os arquivos CSV são carregados a partir da pasta onde os dados estiverem salvos.

Caso os arquivos estejam em outra pasta, ajuste o caminho nas células de carregamento dos dados.

Exemplo:

```python
caminho_receitas = "RAW_recipes.csv"
caminho_interacoes = "RAW_interactions.csv"
```

ou, se estiverem dentro de uma pasta `dados`:

```python
caminho_receitas = "dados/RAW_recipes.csv"
caminho_interacoes = "dados/RAW_interactions.csv"
```

## Resultados esperados

Ao executar o notebook, espera-se obter:

- estatísticas gerais da base;
- distribuição das avaliações;
- identificação de valores ausentes;
- análise preliminar de tempo de preparo;
- medida de esparsidade da matriz usuário-item;
- recomendações de receitas semelhantes;
- métricas iniciais de avaliação da linha de base.

## Métricas de avaliação

As métricas previstas para avaliação são:

- **RMSE**: mede o erro médio quadrático da previsão de notas;
- **MAE**: mede o erro absoluto médio da previsão de notas;
- **Precision@K**: avalia a proporção de recomendações relevantes entre as K recomendações feitas;
- **Recall@K**: avalia a proporção de itens relevantes recuperados entre os itens relevantes existentes.

## Próximos passos

Como próximos passos do projeto, pretende-se:

- melhorar o tratamento dos ingredientes;
- testar diferentes quantidades de vizinhos no modelo;
- comparar a recomendação baseada em conteúdo com filtragem colaborativa;
- desenvolver uma abordagem híbrida;
- incluir o fator temporal de validade dos alimentos;
- avaliar o modelo em uma amostra maior da base;
- preparar a metodologia completa para a próxima etapa do projeto.

## Integrantes

- Carla de Jesus Dutra Paula
- Gabriel Rezende de Oliveira
- Pedro Henrique Trichez
- Timoteo de Jesus Santana

## Universidade

Universidade Presbiteriana Mackenzie

## Ano

2026
