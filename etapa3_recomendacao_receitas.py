# Projeto Aplicado III - Etapa 3
# Sistema de recomendacao de receitas - codigo simples em portugues
#
# Este arquivo ajusta a prova de conceito da etapa 2.
# Principais ajustes:
# 1. usa corretamente os arquivos de treino, validacao e teste;
# 2. cria uma linha de base por media do usuario e media da receita;
# 3. cria um recomendador hibrido simples usando ingredientes + popularidade;
# 4. reavalia o desempenho com RMSE, MAE, Precision@10, Recall@10 e HitRate@10.

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Altere para Path('/mnt/data') quando estiver usando no ChatGPT.
# No GitHub/Colab, deixe a pasta data na raiz do projeto.
pasta_dados = Path('data')

arquivo_treino = pasta_dados / 'interactions_train.csv'
arquivo_validacao = pasta_dados / 'interactions_validation.csv'
arquivo_teste = pasta_dados / 'interactions_test.csv'
arquivo_receitas_processadas = pasta_dados / 'PP_recipes.csv'

# Para deixar a execucao mais leve em computador comum.
# Na apresentacao final, pode aumentar os valores abaixo.
qtd_usuarios_avaliacao = 1000
qtd_receitas_populares = 5000
qtd_negativos_por_usuario = 100
k_recomendacoes = 10


def carregar_interacoes():
    treino = pd.read_csv(arquivo_treino, usecols=['user_id', 'recipe_id', 'rating', 'i'])
    validacao = pd.read_csv(arquivo_validacao, usecols=['user_id', 'recipe_id', 'rating', 'i'])
    teste = pd.read_csv(arquivo_teste, usecols=['user_id', 'recipe_id', 'rating', 'i'])
    return treino, validacao, teste


def avaliar_previsao_notas(treino, dados_avaliacao):
    media_geral = treino['rating'].mean()
    media_usuario = treino.groupby('user_id')['rating'].mean()
    media_receita = treino.groupby('recipe_id')['rating'].mean()

    previsao_usuario = dados_avaliacao['user_id'].map(media_usuario).fillna(media_geral)
    previsao_receita = dados_avaliacao['recipe_id'].map(media_receita).fillna(media_geral)

    # Ajuste simples em relacao a etapa 2: combina media do usuario e media da receita.
    previsao_final = (previsao_usuario + previsao_receita) / 2

    rmse = math.sqrt(mean_squared_error(dados_avaliacao['rating'], previsao_final))
    mae = mean_absolute_error(dados_avaliacao['rating'], previsao_final)

    return rmse, mae


def transformar_lista_em_texto(valor):
    try:
        lista = ast.literal_eval(valor)
    except Exception:
        return ''

    if isinstance(lista, list):
        return ' '.join('ingrediente_' + str(item) for item in lista)

    return ''


def preparar_base_conteudo(treino, validacao, teste):
    usuarios_avaliacao = set(validacao['user_id']) | set(teste['user_id'])

    curtidas_treino = treino[(treino['rating'] >= 4) & (treino['user_id'].isin(usuarios_avaliacao))]

    estatisticas_receita = treino.groupby('i')['rating'].agg(['mean', 'count'])
    estatisticas_receita['popularidade'] = estatisticas_receita['mean'] * np.log1p(estatisticas_receita['count'])

    receitas_populares = set(
        estatisticas_receita.sort_values('popularidade', ascending=False)
        .head(qtd_receitas_populares)
        .index
    )

    receitas_candidatas = set(validacao['i']) | set(teste['i']) | set(curtidas_treino['i']) | receitas_populares

    receitas = pd.read_csv(
        arquivo_receitas_processadas,
        usecols=['i', 'ingredient_ids', 'calorie_level']
    )

    receitas = receitas[receitas['i'].isin(receitas_candidatas)].drop_duplicates('i').reset_index(drop=True)

    receitas['texto_receita'] = (
        receitas['ingredient_ids'].apply(transformar_lista_em_texto)
        + ' nivel_caloria_' + receitas['calorie_level'].astype(str)
    )

    vetorizador = TfidfVectorizer(min_df=1)
    matriz_receitas = vetorizador.fit_transform(receitas['texto_receita'])

    posicao_receita = {int(codigo): posicao for posicao, codigo in enumerate(receitas['i'])}

    popularidade = np.zeros(len(receitas))
    for codigo_receita, valor in estatisticas_receita['popularidade'].items():
        if codigo_receita in posicao_receita:
            popularidade[posicao_receita[codigo_receita]] = valor

    if popularidade.max() > 0:
        popularidade = popularidade / popularidade.max()

    curtidas_por_usuario = curtidas_treino.groupby('user_id')['i'].apply(list).to_dict()
    vistas_por_usuario = treino[treino['user_id'].isin(usuarios_avaliacao)].groupby('user_id')['i'].apply(set).to_dict()

    return receitas, matriz_receitas, posicao_receita, popularidade, curtidas_por_usuario, vistas_por_usuario, estatisticas_receita


def pontuar_receitas_usuario(usuario, candidatos, matriz_receitas, posicao_receita, popularidade, curtidas_por_usuario, vistas_por_usuario):
    candidatos_validos = []
    for receita in candidatos:
        if receita in posicao_receita and receita not in vistas_por_usuario.get(usuario, set()):
            candidatos_validos.append(receita)

    if len(candidatos_validos) == 0:
        return []

    posicoes_candidatas = [posicao_receita[receita] for receita in candidatos_validos]

    receitas_curtidas = []
    for receita in curtidas_por_usuario.get(usuario, [])[-30:]:
        if receita in posicao_receita:
            receitas_curtidas.append(receita)

    if len(receitas_curtidas) > 0:
        posicoes_curtidas = [posicao_receita[receita] for receita in receitas_curtidas]
        perfil_usuario = matriz_receitas[posicoes_curtidas].mean(axis=0)
        pontuacao_conteudo = np.asarray(perfil_usuario @ matriz_receitas[posicoes_candidatas].T).ravel()

        if pontuacao_conteudo.max() > 0:
            pontuacao_conteudo = pontuacao_conteudo / pontuacao_conteudo.max()

        # Recomendacao hibrida simples: conteudo dos ingredientes + popularidade da receita.
        pontuacao_final = 0.65 * pontuacao_conteudo + 0.35 * popularidade[posicoes_candidatas]
    else:
        pontuacao_final = popularidade[posicoes_candidatas]

    ordem = np.argsort(-pontuacao_final)
    receitas_ordenadas = [candidatos_validos[posicao] for posicao in ordem]

    return receitas_ordenadas


def avaliar_top_k(dados_avaliacao, matriz_receitas, posicao_receita, popularidade, curtidas_por_usuario, vistas_por_usuario, estatisticas_receita):
    gerador = np.random.default_rng(42)

    receitas_relevantes = dados_avaliacao[dados_avaliacao['rating'] >= 4]
    relevantes_por_usuario = receitas_relevantes.groupby('user_id')['i'].apply(set).to_dict()

    usuarios = []
    for usuario in relevantes_por_usuario.keys():
        if usuario in curtidas_por_usuario:
            usuarios.append(usuario)

    usuarios = usuarios[:qtd_usuarios_avaliacao]

    receitas_populares = list(estatisticas_receita.sort_values('popularidade', ascending=False).head(qtd_receitas_populares).index)

    precisoes = []
    revocacoes = []
    acertos_usuarios = 0

    for usuario in usuarios:
        relevantes = set()
        for receita in relevantes_por_usuario[usuario]:
            if receita in posicao_receita:
                relevantes.add(receita)

        if len(relevantes) == 0:
            continue

        negativos = []
        for receita in receitas_populares:
            if receita not in relevantes and receita not in vistas_por_usuario.get(usuario, set()) and receita in posicao_receita:
                negativos.append(receita)

        if len(negativos) > qtd_negativos_por_usuario:
            negativos = list(gerador.choice(negativos, size=qtd_negativos_por_usuario, replace=False))

        candidatos = list(relevantes) + negativos
        recomendadas = pontuar_receitas_usuario(
            usuario,
            candidatos,
            matriz_receitas,
            posicao_receita,
            popularidade,
            curtidas_por_usuario,
            vistas_por_usuario
        )[:k_recomendacoes]

        acertos = len(set(recomendadas) & relevantes)
        precisoes.append(acertos / k_recomendacoes)
        revocacoes.append(acertos / len(relevantes))

        if acertos > 0:
            acertos_usuarios += 1

    if len(precisoes) == 0:
        return 0, 0, 0, 0

    precision_k = float(np.mean(precisoes))
    recall_k = float(np.mean(revocacoes))
    hit_rate_k = acertos_usuarios / len(precisoes)

    return len(precisoes), precision_k, recall_k, hit_rate_k


def executar_etapa3():
    treino, validacao, teste = carregar_interacoes()

    print('Quantidade de interacoes de treino:', len(treino))
    print('Quantidade de interacoes de validacao:', len(validacao))
    print('Quantidade de interacoes de teste:', len(teste))

    rmse_validacao, mae_validacao = avaliar_previsao_notas(treino, validacao)
    rmse_teste, mae_teste = avaliar_previsao_notas(treino, teste)

    print('\nAvaliacao de previsao de notas')
    print('Validacao - RMSE:', round(rmse_validacao, 4), 'MAE:', round(mae_validacao, 4))
    print('Teste     - RMSE:', round(rmse_teste, 4), 'MAE:', round(mae_teste, 4))

    receitas, matriz_receitas, posicao_receita, popularidade, curtidas_por_usuario, vistas_por_usuario, estatisticas_receita = preparar_base_conteudo(
        treino,
        validacao,
        teste
    )

    print('\nBase de conteudo preparada')
    print('Receitas candidatas:', len(receitas))
    print('Formato da matriz de conteudo:', matriz_receitas.shape)

    n_validacao, p_validacao, r_validacao, h_validacao = avaliar_top_k(
        validacao,
        matriz_receitas,
        posicao_receita,
        popularidade,
        curtidas_por_usuario,
        vistas_por_usuario,
        estatisticas_receita
    )

    n_teste, p_teste, r_teste, h_teste = avaliar_top_k(
        teste,
        matriz_receitas,
        posicao_receita,
        popularidade,
        curtidas_por_usuario,
        vistas_por_usuario,
        estatisticas_receita
    )

    print('\nAvaliacao Top-K com candidatos amostrados')
    print('Validacao - usuarios:', n_validacao, 'Precision@10:', round(p_validacao, 4), 'Recall@10:', round(r_validacao, 4), 'HitRate@10:', round(h_validacao, 4))
    print('Teste     - usuarios:', n_teste, 'Precision@10:', round(p_teste, 4), 'Recall@10:', round(r_teste, 4), 'HitRate@10:', round(h_teste, 4))


if __name__ == '__main__':
    executar_etapa3()
