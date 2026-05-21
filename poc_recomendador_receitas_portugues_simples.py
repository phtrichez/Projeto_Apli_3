"""
Projeto Aplicado III - Etapa 2
Sistema de recomendação de receitas para redução do desperdício alimentar

Código simples da prova de conceito.
A ideia aqui é mostrar, de forma didática, as etapas principais:
1. carregar os dados;
2. fazer uma análise exploratória inicial;
3. tratar e preparar as receitas;
4. treinar um recomendador baseado em conteúdo;
5. gerar recomendações;
6. avaliar uma linha de base.

Por padrão, o código usa uma amostra para rodar mais rápido.
Para usar mais dados, altere os valores das variáveis abaixo.
"""

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


pasta_dados = Path('/mnt/data')
arquivo_receitas = pasta_dados / 'RAW_recipes.csv'
arquivo_interacoes = pasta_dados / 'RAW_interactions.csv'
arquivo_treino = pasta_dados / 'interactions_train.csv'
arquivo_validacao = pasta_dados / 'interactions_validation.csv'
arquivo_teste = pasta_dados / 'interactions_test.csv'

usar_amostra = True
qtd_receitas = 30000 if usar_amostra else None
qtd_interacoes = 100000 if usar_amostra else None
qtd_treino = 100000 if usar_amostra else None
qtd_validacao = 5000 if usar_amostra else None
qtd_receitas_modelo = 3000
qtd_recomendacoes = 10
nota_relevante = 4


def transformar_texto_em_lista(valor):
    if pd.isna(valor):
        return []

    try:
        lista = ast.literal_eval(valor)
        if isinstance(lista, list):
            return [str(item).lower().strip() for item in lista]
    except:
        return []

    return []


def carregar_dados():
    receitas = pd.read_csv(
        arquivo_receitas,
        usecols=['id', 'name', 'minutes', 'tags', 'ingredients', 'n_ingredients', 'description'],
        nrows=qtd_receitas
    )

    interacoes = pd.read_csv(
        arquivo_interacoes,
        usecols=['user_id', 'recipe_id', 'rating', 'date'],
        nrows=qtd_interacoes
    )

    treino = pd.read_csv(arquivo_treino, nrows=qtd_treino)
    validacao = pd.read_csv(arquivo_validacao, nrows=qtd_validacao)
    teste = pd.read_csv(arquivo_teste, nrows=qtd_validacao)

    return receitas, interacoes, treino, validacao, teste


def analisar_dados(receitas, interacoes):
    usuarios = interacoes['user_id'].nunique()
    receitas_avaliadas = interacoes['recipe_id'].nunique()
    total_interacoes = len(interacoes)

    esparsidade = 1 - (total_interacoes / (usuarios * receitas_avaliadas))

    print('\nANÁLISE EXPLORATÓRIA')
    print('Quantidade de receitas:', receitas['id'].nunique())
    print('Quantidade de usuários:', usuarios)
    print('Quantidade de interações:', total_interacoes)
    print('Quantidade de receitas com avaliação:', receitas_avaliadas)
    print('Esparsidade da matriz usuário-receita:', round(esparsidade, 6))
    print('Média das notas:', round(interacoes['rating'].mean(), 4))
    print('Mediana das notas:', interacoes['rating'].median())

    print('\nDistribuição das notas:')
    print(interacoes['rating'].value_counts().sort_index())

    print('\nValores ausentes nas receitas:')
    print(receitas.isna().sum().sort_values(ascending=False).head(10))

    print('\nResumo do tempo de preparo:')
    print(receitas['minutes'].describe())


def preparar_receitas(receitas, treino):
    receitas_populares = (
        treino.groupby('recipe_id')['rating']
        .count()
        .sort_values(ascending=False)
        .head(qtd_receitas_modelo)
        .index
    )

    receitas_modelo = receitas[receitas['id'].isin(receitas_populares)].copy()

    if len(receitas_modelo) < 500:
        receitas_modelo = receitas.copy()

    receitas_modelo = receitas_modelo.drop_duplicates(subset='id')
    receitas_modelo = receitas_modelo.dropna(subset=['id', 'name'])
    receitas_modelo = receitas_modelo[receitas_modelo['minutes'].fillna(0) > 0]

    limite_tempo = receitas_modelo['minutes'].quantile(0.99)
    receitas_modelo['tempo_tratado'] = receitas_modelo['minutes'].clip(upper=limite_tempo)

    receitas_modelo['ingredientes_lista'] = receitas_modelo['ingredients'].apply(transformar_texto_em_lista)
    receitas_modelo['tags_lista'] = receitas_modelo['tags'].apply(transformar_texto_em_lista)

    receitas_modelo['texto_receita'] = (
        receitas_modelo['name'].fillna('') + ' ' +
        receitas_modelo['ingredientes_lista'].apply(lambda lista: ' '.join(lista)) + ' ' +
        receitas_modelo['tags_lista'].apply(lambda lista: ' '.join(lista))
    )

    colunas = ['id', 'name', 'minutes', 'tempo_tratado', 'n_ingredients', 'texto_receita']
    return receitas_modelo[colunas].reset_index(drop=True)


def treinar_modelo_conteudo(receitas_modelo):
    vetorizador = TfidfVectorizer(
        stop_words='english',
        min_df=2,
        max_df=0.90,
        max_features=15000
    )

    matriz_receitas = vetorizador.fit_transform(receitas_modelo['texto_receita'])

    modelo_vizinhos = NearestNeighbors(metric='cosine', algorithm='brute')
    modelo_vizinhos.fit(matriz_receitas)

    return vetorizador, matriz_receitas, modelo_vizinhos


def recomendar_receitas_parecidas(id_receita, receitas_modelo, matriz_receitas, modelo_vizinhos, quantidade=5):
    posicoes = pd.Series(receitas_modelo.index, index=receitas_modelo['id']).to_dict()

    if id_receita not in posicoes:
        print('Receita não encontrada no conjunto usado pela prova de conceito.')
        return pd.DataFrame()

    posicao_receita = posicoes[id_receita]
    distancias, indices = modelo_vizinhos.kneighbors(
        matriz_receitas[posicao_receita],
        n_neighbors=quantidade + 1
    )

    recomendacoes = receitas_modelo.iloc[indices.flatten()].copy()
    recomendacoes['similaridade'] = 1 - distancias.flatten()
    recomendacoes = recomendacoes[recomendacoes['id'] != id_receita]

    return recomendacoes[['id', 'name', 'minutes', 'n_ingredients', 'similaridade']].head(quantidade)


def avaliar_linha_base(treino, validacao):
    media_geral = treino['rating'].mean()
    media_por_receita = treino.groupby('recipe_id')['rating'].mean()

    notas_reais = validacao['rating']
    notas_previstas = validacao['recipe_id'].map(media_por_receita).fillna(media_geral)

    rmse = math.sqrt(mean_squared_error(notas_reais, notas_previstas))
    mae = mean_absolute_error(notas_reais, notas_previstas)

    print('\nAVALIAÇÃO DA LINHA DE BASE')
    print('RMSE:', round(rmse, 4))
    print('MAE:', round(mae, 4))
    print('Média geral das notas no treino:', round(media_geral, 4))


def criar_perfis_usuarios(treino, receitas_modelo, matriz_receitas):
    receita_para_posicao = pd.Series(receitas_modelo.index, index=receitas_modelo['id']).to_dict()
    treino_filtrado = treino[treino['recipe_id'].isin(receita_para_posicao)].copy()
    treino_bem_avaliado = treino_filtrado[treino_filtrado['rating'] >= nota_relevante]

    perfis_usuarios = {}
    receitas_vistas = {}

    for usuario, grupo in treino_bem_avaliado.groupby('user_id'):
        posicoes = []
        for id_receita in grupo['recipe_id']:
            if id_receita in receita_para_posicao:
                posicoes.append(receita_para_posicao[id_receita])

        if len(posicoes) > 0:
            perfis_usuarios[usuario] = np.asarray(matriz_receitas[posicoes].mean(axis=0))
            receitas_vistas[usuario] = set(treino_filtrado[treino_filtrado['user_id'] == usuario]['recipe_id'])

    return perfis_usuarios, receitas_vistas


def recomendar_para_usuario(usuario, perfis_usuarios, receitas_vistas, receitas_modelo, matriz_receitas, quantidade=10):
    if usuario not in perfis_usuarios:
        return []

    pontuacoes = cosine_similarity(perfis_usuarios[usuario], matriz_receitas).flatten()
    ranking = np.argsort(pontuacoes)[::-1]
    ja_vistas = receitas_vistas.get(usuario, set())

    recomendacoes = []

    for posicao in ranking:
        id_receita = receitas_modelo.iloc[posicao]['id']
        if id_receita not in ja_vistas:
            recomendacoes.append(id_receita)
        if len(recomendacoes) == quantidade:
            break

    return recomendacoes


def avaliar_recomendacoes(treino, validacao, receitas_modelo, matriz_receitas):
    receitas_disponiveis = set(receitas_modelo['id'])

    validacao_relevante = validacao[
        (validacao['rating'] >= nota_relevante) &
        (validacao['recipe_id'].isin(receitas_disponiveis))
    ]

    relevantes_por_usuario = validacao_relevante.groupby('user_id')['recipe_id'].apply(set).to_dict()
    perfis_usuarios, receitas_vistas = criar_perfis_usuarios(treino, receitas_modelo, matriz_receitas)

    precisoes = []
    revocacoes = []

    usuarios = list(relevantes_por_usuario.keys())[:200]

    for usuario in usuarios:
        if usuario not in perfis_usuarios:
            continue

        recomendadas = recomendar_para_usuario(
            usuario,
            perfis_usuarios,
            receitas_vistas,
            receitas_modelo,
            matriz_receitas,
            qtd_recomendacoes
        )

        relevantes = relevantes_por_usuario[usuario]
        acertos = len(set(recomendadas) & relevantes)

        precisoes.append(acertos / qtd_recomendacoes)
        revocacoes.append(acertos / len(relevantes))

    print('\nAVALIAÇÃO DAS RECOMENDAÇÕES')
    if len(precisoes) == 0:
        print('Não houve usuários suficientes para calcular Precision@K e Recall@K nesta amostra.')
    else:
        print('Precision@10:', round(np.mean(precisoes), 4))
        print('Recall@10:', round(np.mean(revocacoes), 4))
        print('Usuários avaliados:', len(precisoes))


def executar_prova_de_conceito():
    print('CARREGANDO DADOS')
    receitas, interacoes, treino, validacao, teste = carregar_dados()

    print('Receitas carregadas:', len(receitas))
    print('Interações carregadas:', len(interacoes))
    print('Registros de treino:', len(treino))
    print('Registros de validação:', len(validacao))
    print('Registros de teste:', len(teste))

    analisar_dados(receitas, interacoes)

    print('\nPREPARANDO RECEITAS')
    receitas_modelo = preparar_receitas(receitas, treino)
    print('Receitas usadas no modelo:', len(receitas_modelo))

    print('\nTREINANDO MODELO BASEADO EM CONTEÚDO')
    vetorizador, matriz_receitas, modelo_vizinhos = treinar_modelo_conteudo(receitas_modelo)
    print('Formato da matriz TF-IDF:', matriz_receitas.shape)

    print('\nEXEMPLO DE RECOMENDAÇÃO')
    id_exemplo = receitas_modelo.iloc[0]['id']
    nome_exemplo = receitas_modelo.iloc[0]['name']
    print('Receita base:', id_exemplo, '-', nome_exemplo)

    recomendacoes = recomendar_receitas_parecidas(
        id_exemplo,
        receitas_modelo,
        matriz_receitas,
        modelo_vizinhos,
        quantidade=5
    )
    print(recomendacoes.to_string(index=False))

    avaliar_linha_base(treino, validacao)
    avaliar_recomendacoes(treino, validacao, receitas_modelo, matriz_receitas)


if __name__ == '__main__':
    executar_prova_de_conceito()
