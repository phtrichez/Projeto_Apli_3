# Projeto Aplicado III - Etapa 4

Sistema inteligente de recomendação de receitas alinhado ao ODS 12.

## Arquivos principais

- `src/entrega4_recomendador_receitas.py`: pipeline simples do recomendador.
- `app/app_streamlit.py`: aplicação Streamlit para testar recomendações.
- `notebooks/Projeto_Aplicado_III_Entrega4_Notebook.ipynb`: notebook passo a passo.
- `docs/Projeto_Aplicado_III_Modulo4_Resultados_Conclusao_Recomendacao_Receitas.docx`: documentação final.

## Como executar

Crie um ambiente virtual e instale as dependências:

```bash
pip install -r requirements.txt
```

Coloque os arquivos da base Food.com dentro da pasta `data/`:

- `RAW_recipes.csv`
- `interactions_train.csv`
- `interactions_validation.csv`
- `interactions_test.csv`

Execute o pipeline:

```bash
python src/entrega4_recomendador_receitas.py
```

Execute a aplicação:

```bash
streamlit run app/app_streamlit.py
```

## Observação

O código foi escrito de forma simples, com nomes em português e sem sofisticação desnecessária, para facilitar a apresentação acadêmica e a manutenção pelo grupo.
