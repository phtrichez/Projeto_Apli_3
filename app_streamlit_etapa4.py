# Aplicacao Streamlit simples para testar o projeto
# Para rodar: streamlit run app/app_streamlit.py

import ast
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


st.set_page_config(page_title='Recomendador de Receitas', layout='wide')


def achar_pasta_dados():
    opcoes = [Path('data'), Path('../data'), Path('/mnt/data')]
    for pasta in opcoes:
        if (pasta / 'RAW_recipes.csv').exists():
            return pasta
    return Path('data')


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


@st.cache_data
def carregar_receitas(qtd_receitas):
    pasta = achar_pasta_dados()
    receitas = pd.read_csv(pasta / 'RAW_recipes.csv', nrows=qtd_receitas)
    receitas = receitas.drop_duplicates(subset='id')
    receitas = receitas.dropna(subset=['id', 'name'])
    receitas = receitas[receitas['minutes'].fillna(0) > 0].copy()
    receitas['ingredientes_lista'] = receitas['ingredients'].apply(texto_para_lista)
    receitas['tags_lista'] = receitas['tags'].apply(texto_para_lista)
    receitas['texto_receita'] = (
        receitas['name'].fillna('') + ' ' +
        receitas['ingredientes_lista'].apply(lambda x: ' '.join(x)) + ' ' +
        receitas['tags_lista'].apply(lambda x: ' '.join(x))
    )
    return receitas.reset_index(drop=True)


@st.cache_resource
def treinar_modelo(textos):
    vetorizador = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.90, max_features=15000)
    matriz = vetorizador.fit_transform(textos)
    modelo = NearestNeighbors(metric='cosine', algorithm='brute')
    modelo.fit(matriz)
    return vetorizador, matriz, modelo


st.title('Sistema de Recomendação de Receitas')
st.write('Aplicação simples da Etapa 4 para testar recomendações por ingredientes ou por uma receita de referência.')

qtd_receitas = st.sidebar.slider('Quantidade de receitas carregadas', 5000, 50000, 20000, step=5000)
qtd_recomendacoes = st.sidebar.slider('Quantidade de recomendações', 5, 20, 10)

receitas = carregar_receitas(qtd_receitas)
vetorizador, matriz, modelo = treinar_modelo(receitas['texto_receita'])

aba1, aba2, aba3 = st.tabs(['Recomendar por ingredientes', 'Receitas parecidas', 'Sobre o projeto'])

with aba1:
    st.subheader('Informe os ingredientes disponíveis')
    ingredientes = st.text_input('Exemplo: chicken, rice, tomato, cheese', value='chicken, rice, tomato')
    dias_validade = st.slider('Prioridade para aproveitar alimentos próximos da validade', 1, 30, 5)

    if st.button('Gerar recomendações por ingredientes'):
        texto = ingredientes.lower().replace(',', ' ')
        vetor_usuario = vetorizador.transform([texto])
        distancias, indices = modelo.kneighbors(vetor_usuario, n_neighbors=qtd_recomendacoes)

        recomendacoes = receitas.iloc[indices.flatten()].copy()
        recomendacoes['similaridade'] = 1 - distancias.flatten()
        recomendacoes['prioridade_validade'] = 1 / dias_validade
        recomendacoes['score_final'] = recomendacoes['similaridade'] + recomendacoes['prioridade_validade']
        recomendacoes = recomendacoes.sort_values('score_final', ascending=False)

        st.dataframe(
            recomendacoes[['id', 'name', 'minutes', 'n_ingredients', 'similaridade', 'score_final']],
            use_container_width=True
        )

with aba2:
    st.subheader('Escolha uma receita e encontre receitas semelhantes')
    nomes = receitas['name'].head(1000).tolist()
    nome_escolhido = st.selectbox('Receita de referência', nomes)

    if st.button('Buscar receitas parecidas'):
        posicao = receitas[receitas['name'] == nome_escolhido].index[0]
        distancias, indices = modelo.kneighbors(matriz[posicao], n_neighbors=qtd_recomendacoes + 1)
        recomendacoes = receitas.iloc[indices.flatten()].copy()
        recomendacoes['similaridade'] = 1 - distancias.flatten()
        recomendacoes = recomendacoes[recomendacoes['name'] != nome_escolhido]
        st.dataframe(
            recomendacoes[['id', 'name', 'minutes', 'n_ingredients', 'similaridade']],
            use_container_width=True
        )

with aba3:
    st.subheader('Objetivo')
    st.write('O projeto recomenda receitas usando ingredientes, tags e similaridade de conteúdo, apoiando o consumo consciente e o aproveitamento de alimentos disponíveis.')
    st.subheader('Como interpretar')
    st.write('Quanto maior a similaridade, mais a receita encontrada se parece com os ingredientes ou com a receita usada como referência.')
    st.write('A prioridade de validade é uma regra simples para representar a ideia de dar mais importância aos alimentos que precisam ser usados primeiro.')
