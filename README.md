##### Projeto 1 - Prever se uma determinada pessoa recebeu, ou não, o Auxílio Emergencial do Governo Federal (BR) durante a pandemia do novo coronavírus (COVID-19), dadas as suas características socioeconômicas.



##### Estrutura do Projeto:



Projeto1_PNAD-COVID/
├── dados/
│   ├── NOVOS_DADOS_N_PADRONIZADOS.csv
│   ├── PNAD_COVID_062020.csv
├── deploy/
│   ├── deploy.py
│   ├── help_deploy_na_AWS.txt
│   ├── Previsao_Auxilio_Emergencial.bat
├── documentation/
│   ├── _Link de acesso aos microdados.txt
│   ├── de-para_PNAD_COVID_062020 (jun).xlsx
│   ├── NOVOS_DADOS_N_PADRONIZADOS (colunas esperadas no deploy).xlsx
│   ├── Perguntas Streamlit.xlsx
│   ├── Tabela_Descrição_Variáveis_PNAD-COVID.pdf
│   ├── Tabela_Descrição_Variáveis_PNAD-COVID.xlsx
├── legacy/
│   ├── PNAD-COVID (ÍNDICE).ipynb
│   ├── PNAD-COVID_v1.ipynb
│   ├── Projeto1_PNAD-COVID_v1.pdf
├── logo/
│   ├── logo.png
├── modelos/
│   ├── dsa_modelo_v1.pkl
│   ├── dsa_padronizador.pkl
├── src/
│   ├── **init**.py
├── _Abordagem ML.docx
├── _Roteiro PNAD-COVID (previsão aux emergencial).odt
├── PNAD-COVID_vX.ipynb
├── README.md
├── requirements.txt





\# Abra o terminal ou prompt de comando, navegue até a pasta com os arquivos e execute:



pip install -r requirements.txt





\# Navegue agora até a pasta "deploy" e execute:



streamlit run deploy.py

