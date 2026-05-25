# Projeto Aplicado III - Sistema de Recomendação de Receitas

Sistema inteligente de recomendação de receitas desenvolvido para o componente **Projeto Aplicado III**, com foco na redução do desperdício alimentar e alinhamento ao **ODS 12 - Consumo e Produção Responsáveis**.

O projeto utiliza dados da base **Food.com Recipes and Interactions** para construir uma prova de conceito de recomendação de receitas. A proposta é sugerir receitas com base em ingredientes, características das receitas e similaridade de conteúdo, apoiando o usuário no melhor aproveitamento dos alimentos disponíveis.

## Links do projeto

- **Vídeo de apresentação:** https://youtu.be/svtluxm1PSA
- **Canvas / apresentação do projeto:** https://www.canva.com/design/DAHKfoZxPX8/bunYgpBztbmqEtNMErhxMA/edit
- **Base de dados:** https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

## Integrantes

- Carla de Jesus Dutra Paula
- Gabriel Rezende de Oliveira
- Pedro Henrique Trichez
- Timoteo de Jesus Santana

## Objetivo do projeto

Desenvolver e avaliar uma prova de conceito de um sistema de recomendação de receitas capaz de sugerir opções úteis ao usuário, considerando características das receitas e ingredientes disponíveis.

Além da recomendação em si, o projeto busca relacionar Ciência de Dados, Aprendizado de Máquina e sustentabilidade, demonstrando como sistemas de recomendação podem apoiar decisões de consumo mais conscientes.

## Problema tratado

O desperdício alimentar é um problema ambiental, econômico e social. Muitas vezes, alimentos disponíveis em casa deixam de ser utilizados por falta de planejamento ou por dificuldade em pensar em receitas adequadas aos ingredientes existentes.

Neste contexto, um sistema de recomendação pode auxiliar o usuário a encontrar receitas compatíveis com os alimentos disponíveis, contribuindo para o aproveitamento de ingredientes e para a redução do desperdício doméstico.

## Abordagem utilizada

A solução utiliza principalmente **recomendação baseada em conteúdo**, considerando informações textuais das receitas, como:

- nome da receita;
- ingredientes;
- tags/categorias;
- similaridade entre receitas.

A representação textual das receitas é transformada em vetores por meio de **TF-IDF**. Em seguida, é utilizado o algoritmo **Nearest Neighbors** com distância do cosseno para encontrar receitas semelhantes.

Também foram utilizadas linhas de base para avaliação de notas, com métricas como **RMSE** e **MAE**, além de métricas Top-K, como **Precision@10** e **Recall@10**.

## Observação sobre validade dos alimentos

A base Food.com não possui informações reais sobre a validade dos alimentos disponíveis na casa do usuário. Por isso, a validade não foi tratada como uma variável real do dataset.

Na aplicação, a validade foi considerada como uma **regra de negócio simulada**: o usuário pode informar quantos dias faltam para determinado alimento vencer, e alimentos mais próximos do vencimento recebem maior prioridade na busca por receitas.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Jupyter Notebook
- Git/GitHub

## Estrutura do projeto

```text
Projeto_Apli_3/
│
├── .idea/
├── data/
├── outputs/
│
├── .gitignore
├── app_streamlit_etapa4.py
├── entrega4_recomendador_receitas.py
├── gerar_graficos_resultados.py
├── Projeto_Aplicado_III_Entrega4_Notebook.ipynb
├── README.md
└── requirements.txt