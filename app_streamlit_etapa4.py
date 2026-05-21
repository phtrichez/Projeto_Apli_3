# Aplicacao Streamlit simples para testar o projeto
# Para rodar: streamlit run app_streamlit_etapa4.py

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


@st.cache_data
def listar_ingredientes(receitas):
    ingredientes = set()
    for lista in receitas['ingredientes_lista']:
        for item in lista:
            if item:
                ingredientes.add(item)
    return sorted(ingredientes)


@st.cache_resource
def treinar_modelo(textos):
    vetorizador = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.90, max_features=15000)
    matriz = vetorizador.fit_transform(textos)
    modelo = NearestNeighbors(metric='cosine', algorithm='brute')
    modelo.fit(matriz)
    return vetorizador, matriz, modelo


def montar_texto_ingredientes(ingredientes_escolhidos, dias_validade):
    # Quanto menor a validade, mais vezes o ingrediente aparece no texto.
    # Isso aumenta o peso dele no vetor TF-IDF de consulta, de forma simples.
    partes = []
    for ingrediente, dias in zip(ingredientes_escolhidos, dias_validade):
        if ingrediente == '':
            continue
        if dias <= 3:
            peso = 5
        elif dias <= 7:
            peso = 3
        else:
            peso = 1
        partes.extend([ingrediente] * peso)
    return ' '.join(partes)


def mostrar_card_receita(linha, numero):
    with st.container(border=True):
        st.markdown(f'### {numero}. {linha["name"]}')
        col1, col2, col3 = st.columns(3)
        col1.metric('Similaridade', f'{linha["similaridade"]:.3f}')
        col2.metric('Tempo', f'{int(linha["minutes"])} min')
        col3.metric('Ingredientes', int(linha['n_ingredients']))
        st.caption(f'ID da receita: {int(linha["id"])}')


st.title('Sistema de Recomendação de Receitas')
st.write('Aplicação simples da Etapa 4 para testar recomendações por ingredientes ou por uma receita de referência.')

qtd_receitas = st.sidebar.slider('Quantidade de receitas carregadas', 5000, 50000, 20000, step=5000)
qtd_recomendacoes = st.sidebar.slider('Quantidade de recomendações', 5, 20, 10)

receitas = carregar_receitas(qtd_receitas)
ingredientes_disponiveis = listar_ingredientes(receitas)
vetorizador, matriz, modelo = treinar_modelo(receitas['texto_receita'])

aba1, aba2, aba3 = st.tabs(['Recomendar por alimentos disponíveis', 'Receitas parecidas', 'Sobre o projeto'])

with aba1:
    st.subheader('Informe os alimentos disponíveis')
    st.write('Cada alimento fica em um bloco. Comece a digitar no campo e escolha uma opção da lista para evitar erro de escrita.')

    qtd_alimentos = st.slider('Quantidade de alimentos que você quer informar', 1, 8, 4)

    ingredientes_escolhidos = []
    dias_validade = []

    for i in range(qtd_alimentos):
        with st.container(border=True):
            st.markdown(f'**Alimento {i + 1}**')
            col1, col2 = st.columns([2, 1])
            ingrediente = col1.selectbox(
                'Comece a escrever e escolha o alimento',
                [''] + ingredientes_disponiveis,
                key=f'ingrediente_{i}'
            )
            dias = col2.slider(
                'Dias até vencer',
                1,
                30,
                7,
                key=f'dias_{i}'
            )
            ingredientes_escolhidos.append(ingrediente)
            dias_validade.append(dias)

    if st.button('Gerar recomendações por alimentos'):
        texto = montar_texto_ingredientes(ingredientes_escolhidos, dias_validade)

        if texto.strip() == '':
            st.warning('Escolha pelo menos um alimento para gerar recomendações.')
        else:
            vetor_usuario = vetorizador.transform([texto])
            distancias, indices = modelo.kneighbors(vetor_usuario, n_neighbors=qtd_recomendacoes)

            recomendacoes = receitas.iloc[indices.flatten()].copy()
            recomendacoes['similaridade'] = 1 - distancias.flatten()
            recomendacoes = recomendacoes.sort_values('similaridade', ascending=False)

            st.info('A validade foi usada como uma regra simples: alimentos com menos dias até vencer recebem maior peso na busca. A base Food.com não possui data real de validade dos alimentos.')

            for numero, (_, linha) in enumerate(recomendacoes.iterrows(), start=1):
                mostrar_card_receita(linha, numero)

with aba2:
    st.subheader('Escolha uma receita e encontre receitas semelhantes')
    st.write('O campo abaixo também permite começar a digitar o nome da receita para escolher sem erro de escrita.')
    nomes = receitas['name'].head(3000).tolist()
    nome_escolhido = st.selectbox('Receita de referência', nomes)

    if st.button('Buscar receitas parecidas'):
        posicao = receitas[receitas['name'] == nome_escolhido].index[0]
        distancias, indices = modelo.kneighbors(matriz[posicao], n_neighbors=qtd_recomendacoes + 1)
        recomendacoes = receitas.iloc[indices.flatten()].copy()
        recomendacoes['similaridade'] = 1 - distancias.flatten()
        recomendacoes = recomendacoes[recomendacoes['name'] != nome_escolhido]

        for numero, (_, linha) in enumerate(recomendacoes.iterrows(), start=1):
            mostrar_card_receita(linha, numero)

with aba3:
    st.subheader('Objetivo')
    st.write('O projeto recomenda receitas usando ingredientes, tags e similaridade de conteúdo, apoiando o consumo consciente e o aproveitamento de alimentos disponíveis.')
    st.subheader('Como interpretar')
    st.write('Quanto maior a similaridade, mais a receita encontrada se parece com os ingredientes ou com a receita usada como referência.')
    st.subheader('Como a validade foi considerada')
    st.write('A base Food.com não possui a data de validade dos alimentos do usuário. Por isso, na aplicação, a validade é uma informação simulada pelo usuário. Quando o usuário informa que um alimento está perto de vencer, esse alimento recebe maior peso na busca das receitas.')
    st.subheader('Limitação de idioma')
    st.write('As receitas da base podem conter nomes, tags e ingredientes em diferentes idiomas ou com diferentes formas de escrita. Nesta versão, ainda não foi feito tratamento multilíngue. Essa melhoria fica como trabalho futuro.')
