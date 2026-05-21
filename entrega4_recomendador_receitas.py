# Projeto Aplicado III - Etapa 4
# Sistema simples de recomendacao de receitas
# Codigo em portugues e propositalmente simples para fins academicos.

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors


def achar_pasta_dados():
    opcoes = [Path('data'), Path('../data'), Path('/mnt/data')]
    for pasta in opcoes:
        if (pasta / 'RAW_recipes.csv').exists():
            return pasta
    return Path('data')


PASTA_DADOS = achar_pasta_dados()


def texto_para_lista(valor):
    if pd.isna(valor):
        return []
    try:
        lista = ast.literal_eval(valor)
        if isinstance(lista, list):
            return [str(x).lower().strip() for x in lista]
    except Exception:
        return []
    return []


def carregar_dados(qtd_receitas=50000):
    receitas = pd.read_csv(PASTA_DADOS / 'RAW_recipes.csv', nrows=qtd_receitas)
    treino = pd.read_csv(PASTA_DADOS / 'interactions_train.csv')
    validacao = pd.read_csv(PASTA_DADOS / 'interactions_validation.csv')
    teste = pd.read_csv(PASTA_DADOS / 'interactions_test.csv')
    return receitas, treino, validacao, teste


def preparar_receitas(receitas):
    receitas = receitas.copy()
    receitas = receitas.drop_duplicates(subset='id')
    receitas = receitas.dropna(subset=['id', 'name'])
    receitas = receitas[receitas['minutes'].fillna(0) > 0]

    limite_tempo = receitas['minutes'].quantile(0.99)
    receitas['tempo_tratado'] = receitas['minutes'].clip(upper=limite_tempo)

    receitas['ingredientes_lista'] = receitas['ingredients'].apply(texto_para_lista)
    receitas['tags_lista'] = receitas['tags'].apply(texto_para_lista)

    receitas['texto_receita'] = (
        receitas['name'].fillna('') + ' ' +
        receitas['ingredientes_lista'].apply(lambda x: ' '.join(x)) + ' ' +
        receitas['tags_lista'].apply(lambda x: ' '.join(x))
    )

    return receitas.reset_index(drop=True)


def treinar_modelo_conteudo(receitas_modelo):
    vetorizador = TfidfVectorizer(
        stop_words='english',
        min_df=2,
        max_df=0.90,
        max_features=15000
    )
    matriz = vetorizador.fit_transform(receitas_modelo['texto_receita'])

    modelo = NearestNeighbors(metric='cosine', algorithm='brute')
    modelo.fit(matriz)
    return vetorizador, matriz, modelo


def recomendar_por_receita(id_receita, receitas_modelo, matriz, modelo, quantidade=10):
    posicoes = pd.Series(receitas_modelo.index, index=receitas_modelo['id']).to_dict()
    if id_receita not in posicoes:
        return pd.DataFrame()

    posicao = posicoes[id_receita]
    distancias, indices = modelo.kneighbors(matriz[posicao], n_neighbors=quantidade + 1)

    recomendacoes = receitas_modelo.iloc[indices.flatten()].copy()
    recomendacoes['similaridade'] = 1 - distancias.flatten()
    recomendacoes = recomendacoes[recomendacoes['id'] != id_receita]
    return recomendacoes[['id', 'name', 'minutes', 'n_ingredients', 'similaridade']].head(quantidade)


def recomendar_por_ingredientes(ingredientes_digitados, receitas_modelo, vetorizador, matriz, quantidade=10):
    texto = ingredientes_digitados.lower().replace(',', ' ')
    vetor_usuario = vetorizador.transform([texto])

    modelo_busca = NearestNeighbors(metric='cosine', algorithm='brute')
    modelo_busca.fit(matriz)
    distancias, indices = modelo_busca.kneighbors(vetor_usuario, n_neighbors=quantidade)

    recomendacoes = receitas_modelo.iloc[indices.flatten()].copy()
    recomendacoes['similaridade'] = 1 - distancias.flatten()
    return recomendacoes[['id', 'name', 'minutes', 'n_ingredients', 'similaridade']]


def avaliar_linha_base(treino, validacao):
    media_geral = treino['rating'].mean()
    media_receita = treino.groupby('recipe_id')['rating'].mean()
    media_usuario = treino.groupby('user_id')['rating'].mean()

    real = validacao['rating']

    prev_media_geral = pd.Series(media_geral, index=validacao.index)
    prev_receita = validacao['recipe_id'].map(media_receita).fillna(media_geral)
    prev_usuario = validacao['user_id'].map(media_usuario).fillna(media_geral)
    prev_combinada = (prev_receita + prev_usuario) / 2

    resultados = []
    for nome, prev in [
        ('Media geral', prev_media_geral),
        ('Media por receita', prev_receita),
        ('Media por usuario', prev_usuario),
        ('Media usuario + receita', prev_combinada),
    ]:
        rmse = math.sqrt(mean_squared_error(real, prev))
        mae = mean_absolute_error(real, prev)
        resultados.append({'modelo': nome, 'RMSE': rmse, 'MAE': mae})

    return pd.DataFrame(resultados)


def avaliar_recomendacao_popular(treino, validacao, k=10, max_usuarios=500):
    # Avaliacao simples Top-K: receitas com nota >= 4 sao consideradas relevantes.
    relevantes_validacao = validacao[validacao['rating'] >= 4]
    usuarios = relevantes_validacao['user_id'].drop_duplicates().head(max_usuarios)

    receitas_populares = (
        treino[treino['rating'] >= 4]
        .groupby('recipe_id')
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    precisions = []
    recalls = []

    for usuario in usuarios:
        vistos = set(treino.loc[treino['user_id'] == usuario, 'recipe_id'])
        reais = set(relevantes_validacao.loc[relevantes_validacao['user_id'] == usuario, 'recipe_id'])
        recomendados = [r for r in receitas_populares if r not in vistos][:k]
        acertos = len(set(recomendados) & reais)
        precisions.append(acertos / k)
        recalls.append(acertos / len(reais) if len(reais) > 0 else 0)

    return {
        'modelo': 'Popularidade',
        'Precision@10': float(np.mean(precisions)) if precisions else 0.0,
        'Recall@10': float(np.mean(recalls)) if recalls else 0.0,
        'usuarios_avaliados': len(precisions)
    }


def executar_pipeline():
    receitas, treino, validacao, teste = carregar_dados()
    receitas_modelo = preparar_receitas(receitas)
    vetorizador, matriz, modelo = treinar_modelo_conteudo(receitas_modelo)

    resultados_erro = avaliar_linha_base(treino, validacao)
    resultado_topk = avaliar_recomendacao_popular(treino, validacao)

    return receitas_modelo, vetorizador, matriz, modelo, resultados_erro, resultado_topk


if __name__ == '__main__':
    receitas_modelo, vetorizador, matriz, modelo, resultados_erro, resultado_topk = executar_pipeline()
    print('Receitas usadas no modelo:', len(receitas_modelo))
    print('\nResultados de erro:')
    print(resultados_erro)
    print('\nResultado Top-K:')
    print(resultado_topk)

    id_exemplo = int(receitas_modelo.iloc[0]['id'])
    print('\nExemplo de recomendacoes parecidas:')
    print(recomendar_por_receita(id_exemplo, receitas_modelo, matriz, modelo, quantidade=5))
