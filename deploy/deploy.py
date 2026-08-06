# Projeto 1 - Prever se uma determinada pessoa recebeu, ou não, o Auxílio Emergencial do Governo Federal (BR) 
# durante a pandemia do novo coronavírus (COVID-19), dadas as suas características socioeconômicas

# Instale o streamlit: pip install streamlit

# Imports
from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Carregar o modelo e o scaler
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "modelos" / "dsa_modelo_v1.pkl"
MODEL_PATH2 = BASE_DIR.parent / "modelos" / "dsa_padronizador.pkl"

modelo = joblib.load(MODEL_PATH)
scaler = joblib.load(MODEL_PATH2)

# Função para pré-processar os dados de entrada
# As colunas devem ser exatamente as mesmas usadas durante o treinamento
def preprocess_input(salario,
                     valor_ticket_cesta,
                     soma_aposentadorias_mais_pensao,
                     d0023,
                     soma_bolsas_familia,
                     soma_bpc_loas,
                     soma_seguros_desemprego,
                     d0073,
                     uf_ac,
                     uf_al,
                     uf_am,
                     uf_ap,
                     uf_ba,
                     uf_ce,
                     uf_df,
                     uf_es,
                     uf_go,
                     uf_ma,
                     uf_mg,
                     uf_ms,
                     uf_mt,
                     uf_pa,
                     uf_pb,
                     uf_pe,
                     uf_pi,
                     uf_pr,
                     uf_rj,
                     uf_rn,
                     uf_ro,
                     uf_rr,
                     uf_rs,
                     uf_sc,
                     uf_se,
                     uf_sp,
                     uf_to,
                     v1022_1,
                     v1022_2,
                     v1023_1,
                     v1023_2,
                     v1023_3,
                     v1023_4,
                     idade_menor_5,
                     idade_12_17,
                     idade_18_49,
                     idade_5_11,
                     idade_50_64,
                     idade_65_mais,
                     sexo_homem,
                     sexo_mulher,
                     cor_amarela,
                     cor_branca,
                     cor_ignorado,
                     cor_indigena,
                     cor_parda,
                     cor_preta,
                     escolaridade_fundamental_completo_ou_medio_incompleto,
                     escolaridade_medio_completo_ou_superior_incompleto,
                     escolaridade_pos_graduacao,
                     escolaridade_sem_instrucao_ou_fundamental_incompleto,
                     escolaridade_superior_completo,
                     estava_trabalhando_na,
                     estava_trabalhando_nao,
                     estava_trabalhando_sim,
                     tipo_emprego_autonomo__conta_propria_,
                     tipo_emprego_empregado_do_setor_privado,
                     tipo_emprego_empregado_do_setor_publico,
                     tipo_emprego_empregador,
                     tipo_emprego_estava_fora_do_mercado_de_trabalho,
                     tipo_emprego_militar,
                     tipo_emprego_na,
                     tipo_emprego_policial_militar_ou_bombeiro_militar,
                     tipo_emprego_trabalhador_domestico__empregado_domestico_cuidados_baba_,
                     tipo_emprego_trabalhador_familiar_nao_remunerado_em_ajuda_a_parente,
                     carteira_assinada_na,
                     carteira_assinada_nao,
                     carteira_assinada_sim_carteira_de_trab_assinada,
                     carteira_assinada_sim_servidor_pub_estatutario,
                     c007c_1,
                     c007c_2,
                     c007c_3,
                     c007c_4,
                     c007c_5,
                     c007c_6,
                     c007c_7,
                     c007c_8,
                     c007c_9,
                     c007c_10,
                     c007c_11,
                     c007c_12,
                     c007c_13,
                     c007c_14,
                     c007c_15,
                     c007c_16,
                     c007c_17,
                     c007c_18,
                     c007c_19,
                     c007c_20,
                     c007c_21,
                     c007c_22,
                     c007c_23,
                     c007c_24,
                     c007c_25,
                     c007c_26,
                     c007c_27,
                     c007c_28,
                     c007c_29,
                     c007c_30,
                     c007c_31,
                     c007c_32,
                     c007c_33,
                     c007c_34,
                     c007c_35,
                     c007c_36,
                     c007c_99,
                     c007d_1,
                     c007d_2,
                     c007d_3,
                     c007d_4,
                     c007d_5,
                     c007d_6,
                     c007d_7,
                     c007d_8,
                     c007d_9,
                     c007d_10,
                     c007d_11,
                     c007d_12,
                     c007d_13,
                     c007d_14,
                     c007d_15,
                     c007d_16,
                     c007d_17,
                     c007d_18,
                     c007d_19,
                     c007d_20,
                     c007d_21,
                     c007d_22,
                     c007d_23,
                     c007d_24,
                     c007d_25,
                     c007d_99,
                     c010_1,
                     c010_99,
                     somente_beneficio_na,
                     somente_beneficio_sim,
                     nao_remunerado_na,
                     nao_remunerado_sim,
                     c014_1,
                     c014_2,
                     c014_99):
    
    # Dataframe
    data = pd.DataFrame({
        'SALARIO': [salario],
        'VALOR_TICKET-CESTA': [valor_ticket_cesta],
        'SOMA_APOSENTADORIAS+PENSAO': [soma_aposentadorias_mais_pensao],
        'D0023': [d0023],
        'SOMA_BOLSAS_FAMILIA': [soma_bolsas_familia],
        'SOMA_BPC-LOAS': [soma_bpc_loas],
        'SOMA_SEGUROS_DESEMPREGO': [soma_seguros_desemprego],
        'D0073': [d0073],
        'UF_AC': [uf_ac],
        'UF_AL': [uf_al],
        'UF_AM': [uf_am],
        'UF_AP': [uf_ap],
        'UF_BA': [uf_ba],
        'UF_CE': [uf_ce],
        'UF_DF': [uf_df],
        'UF_ES': [uf_es],
        'UF_GO': [uf_go],
        'UF_MA': [uf_ma],
        'UF_MG': [uf_mg],
        'UF_MS': [uf_ms],
        'UF_MT': [uf_mt],
        'UF_PA': [uf_pa],
        'UF_PB': [uf_pb],
        'UF_PE': [uf_pe],
        'UF_PI': [uf_pi],
        'UF_PR': [uf_pr],
        'UF_RJ': [uf_rj],
        'UF_RN': [uf_rn],
        'UF_RO': [uf_ro],
        'UF_RR': [uf_rr],
        'UF_RS': [uf_rs],
        'UF_SC': [uf_sc],
        'UF_SE': [uf_se],
        'UF_SP': [uf_sp],
        'UF_TO': [uf_to],
        'V1022_1': [v1022_1],
        'V1022_2': [v1022_2],
        'V1023_1': [v1023_1],
        'V1023_2': [v1023_2],
        'V1023_3': [v1023_3],
        'V1023_4': [v1023_4],
        'IDADE_-5': [idade_menor_5],
        'IDADE_12-17': [idade_12_17],
        'IDADE_18-49': [idade_18_49],
        'IDADE_5-11': [idade_5_11],
        'IDADE_50-64': [idade_50_64],
        'IDADE_65+': [idade_65_mais],
        'SEXO_Homem': [sexo_homem],
        'SEXO_Mulher': [sexo_mulher],
        'COR_Amarela': [cor_amarela],
        'COR_Branca': [cor_branca],
        'COR_Ignorado': [cor_ignorado],
        'COR_Indígena': [cor_indigena],
        'COR_Parda': [cor_parda],
        'COR_Preta': [cor_preta],
        'ESCOLARIDADE_Fundamental Completo ou Médio Incompleto': [escolaridade_fundamental_completo_ou_medio_incompleto],
        'ESCOLARIDADE_Médio Completo ou Superior Incompleto': [escolaridade_medio_completo_ou_superior_incompleto],
        'ESCOLARIDADE_Pós-graduação': [escolaridade_pos_graduacao],
        'ESCOLARIDADE_Sem instrução ou Fundamental Incompleto': [escolaridade_sem_instrucao_ou_fundamental_incompleto],
        'ESCOLARIDADE_Superior Completo': [escolaridade_superior_completo],
        'ESTAVA_TRABALHANDO_N/A': [estava_trabalhando_na],
        'ESTAVA_TRABALHANDO_Não': [estava_trabalhando_nao],
        'ESTAVA_TRABALHANDO_Sim': [estava_trabalhando_sim],
        'TIPO_EMPREGO_Autônomo (Conta própria)': [tipo_emprego_autonomo__conta_propria_],
        'TIPO_EMPREGO_Empregado do setor privado': [tipo_emprego_empregado_do_setor_privado],
        'TIPO_EMPREGO_Empregado do setor público': [tipo_emprego_empregado_do_setor_publico],
        'TIPO_EMPREGO_Empregador': [tipo_emprego_empregador],
        'TIPO_EMPREGO_Estava fora do mercado de trabalho': [tipo_emprego_estava_fora_do_mercado_de_trabalho],
        'TIPO_EMPREGO_Militar': [tipo_emprego_militar],
        'TIPO_EMPREGO_N/A': [tipo_emprego_na],
        'TIPO_EMPREGO_Policial militar ou bombeiro militar': [tipo_emprego_policial_militar_ou_bombeiro_militar],
        'TIPO_EMPREGO_Trabalhador doméstico (empregado doméstico, cuidados, babá)': [tipo_emprego_trabalhador_domestico__empregado_domestico_cuidados_baba_],
        'TIPO_EMPREGO_Trabalhador familiar não remunerado em ajuda a parente': [tipo_emprego_trabalhador_familiar_nao_remunerado_em_ajuda_a_parente],
        'CARTEIRA_ASSINADA_N/A': [carteira_assinada_na],
        'CARTEIRA_ASSINADA_Não': [carteira_assinada_nao],
        'CARTEIRA_ASSINADA_Sim, carteira de trab assinada': [carteira_assinada_sim_carteira_de_trab_assinada],
        'CARTEIRA_ASSINADA_Sim, servidor púb estatutário': [carteira_assinada_sim_servidor_pub_estatutario],
        'C007C_1.0': [c007c_1],
        'C007C_2.0': [c007c_2],
        'C007C_3.0': [c007c_3],
        'C007C_4.0': [c007c_4],
        'C007C_5.0': [c007c_5],
        'C007C_6.0': [c007c_6],
        'C007C_7.0': [c007c_7],
        'C007C_8.0': [c007c_8],
        'C007C_9.0': [c007c_9],
        'C007C_10.0': [c007c_10],
        'C007C_11.0': [c007c_11],
        'C007C_12.0': [c007c_12],
        'C007C_13.0': [c007c_13],
        'C007C_14.0': [c007c_14],
        'C007C_15.0': [c007c_15],
        'C007C_16.0': [c007c_16],
        'C007C_17.0': [c007c_17],
        'C007C_18.0': [c007c_18],
        'C007C_19.0': [c007c_19],
        'C007C_20.0': [c007c_20],
        'C007C_21.0': [c007c_21],
        'C007C_22.0': [c007c_22],
        'C007C_23.0': [c007c_23],
        'C007C_24.0': [c007c_24],
        'C007C_25.0': [c007c_25],
        'C007C_26.0': [c007c_26],
        'C007C_27.0': [c007c_27],
        'C007C_28.0': [c007c_28],
        'C007C_29.0': [c007c_29],
        'C007C_30.0': [c007c_30],
        'C007C_31.0': [c007c_31],
        'C007C_32.0': [c007c_32],
        'C007C_33.0': [c007c_33],
        'C007C_34.0': [c007c_34],
        'C007C_35.0': [c007c_35],
        'C007C_36.0': [c007c_36],
        'C007C_99.0': [c007c_99],
        'C007D_1.0': [c007d_1],
        'C007D_2.0': [c007d_2],
        'C007D_3.0': [c007d_3],
        'C007D_4.0': [c007d_4],
        'C007D_5.0': [c007d_5],
        'C007D_6.0': [c007d_6],
        'C007D_7.0': [c007d_7],
        'C007D_8.0': [c007d_8],
        'C007D_9.0': [c007d_9],
        'C007D_10.0': [c007d_10],
        'C007D_11.0': [c007d_11],
        'C007D_12.0': [c007d_12],
        'C007D_13.0': [c007d_13],
        'C007D_14.0': [c007d_14],
        'C007D_15.0': [c007d_15],
        'C007D_16.0': [c007d_16],
        'C007D_17.0': [c007d_17],
        'C007D_18.0': [c007d_18],
        'C007D_19.0': [c007d_19],
        'C007D_20.0': [c007d_20],
        'C007D_21.0': [c007d_21],
        'C007D_22.0': [c007d_22],
        'C007D_23.0': [c007d_23],
        'C007D_24.0': [c007d_24],
        'C007D_25.0': [c007d_25],
        'C007D_99.0': [c007d_99],
        'C010_1.0': [c010_1],
        'C010_99.0': [c010_99],
        'SOMENTE_BENEFICIO_N/A': [somente_beneficio_na],
        'SOMENTE_BENEFICIO_Sim': [somente_beneficio_sim],
        'NAO_REMUNERADO_N/A': [nao_remunerado_na],
        'NAO_REMUNERADO_Sim': [nao_remunerado_sim],
        'C014_1.0': [c014_1],
        'C014_2.0': [c014_2],
        'C014_99.0': [c014_99]
    })

    # Lista de colunas
    numeric_cols = ['SALARIO', 
                    'VALOR_TICKET-CESTA',
                    'SOMA_APOSENTADORIAS+PENSAO',
                    'D0023',
                    'SOMA_BOLSAS_FAMILIA',
                    'SOMA_BPC-LOAS',
                    'SOMA_SEGUROS_DESEMPREGO',
                    'D0073',
                    'UF_AC',
                    'UF_AL',
                    'UF_AM',
                    'UF_AP',
                    'UF_BA',
                    'UF_CE',
                    'UF_DF',
                    'UF_ES',
                    'UF_GO',
                    'UF_MA',
                    'UF_MG',
                    'UF_MS',
                    'UF_MT',
                    'UF_PA',
                    'UF_PB',
                    'UF_PE',
                    'UF_PI',
                    'UF_PR',
                    'UF_RJ',
                    'UF_RN',
                    'UF_RO',
                    'UF_RR',
                    'UF_RS',
                    'UF_SC',
                    'UF_SE',
                    'UF_SP',
                    'UF_TO',
                    'V1022_1',
                    'V1022_2',
                    'V1023_1',
                    'V1023_2',
                    'V1023_3',
                    'V1023_4',
                    'IDADE_-5',
                    'IDADE_12-17',
                    'IDADE_18-49',
                    'IDADE_5-11',
                    'IDADE_50-64',
                    'IDADE_65+',
                    'SEXO_Homem',
                    'SEXO_Mulher',
                    'COR_Amarela',
                    'COR_Branca',
                    'COR_Ignorado',
                    'COR_Indígena',
                    'COR_Parda',
                    'COR_Preta',
                    'ESCOLARIDADE_Fundamental Completo ou Médio Incompleto',
                    'ESCOLARIDADE_Médio Completo ou Superior Incompleto',
                    'ESCOLARIDADE_Pós-graduação',
                    'ESCOLARIDADE_Sem instrução ou Fundamental Incompleto',
                    'ESCOLARIDADE_Superior Completo',
                    'ESTAVA_TRABALHANDO_N/A',
                    'ESTAVA_TRABALHANDO_Não',
                    'ESTAVA_TRABALHANDO_Sim',
                    'TIPO_EMPREGO_Autônomo (Conta própria)',
                    'TIPO_EMPREGO_Empregado do setor privado',
                    'TIPO_EMPREGO_Empregado do setor público',
                    'TIPO_EMPREGO_Empregador',
                    'TIPO_EMPREGO_Estava fora do mercado de trabalho',
                    'TIPO_EMPREGO_Militar',
                    'TIPO_EMPREGO_N/A',
                    'TIPO_EMPREGO_Policial militar ou bombeiro militar',
                    'TIPO_EMPREGO_Trabalhador doméstico (empregado doméstico, cuidados, babá)',
                    'TIPO_EMPREGO_Trabalhador familiar não remunerado em ajuda a parente',
                    'CARTEIRA_ASSINADA_N/A',
                    'CARTEIRA_ASSINADA_Não',
                    'CARTEIRA_ASSINADA_Sim, carteira de trab assinada',
                    'CARTEIRA_ASSINADA_Sim, servidor púb estatutário',
                    'C007C_1.0',
                    'C007C_2.0',
                    'C007C_3.0',
                    'C007C_4.0',
                    'C007C_5.0',
                    'C007C_6.0',
                    'C007C_7.0',
                    'C007C_8.0',
                    'C007C_9.0',
                    'C007C_10.0',
                    'C007C_11.0',
                    'C007C_12.0',
                    'C007C_13.0',
                    'C007C_14.0',
                    'C007C_15.0',
                    'C007C_16.0',
                    'C007C_17.0',
                    'C007C_18.0',
                    'C007C_19.0',
                    'C007C_20.0',
                    'C007C_21.0',
                    'C007C_22.0',
                    'C007C_23.0',
                    'C007C_24.0',
                    'C007C_25.0',
                    'C007C_26.0',
                    'C007C_27.0',
                    'C007C_28.0',
                    'C007C_29.0',
                    'C007C_30.0',
                    'C007C_31.0',
                    'C007C_32.0',
                    'C007C_33.0',
                    'C007C_34.0',
                    'C007C_35.0',
                    'C007C_36.0',
                    'C007C_99.0',
                    'C007D_1.0',
                    'C007D_2.0',
                    'C007D_3.0',
                    'C007D_4.0',
                    'C007D_5.0',
                    'C007D_6.0',
                    'C007D_7.0',
                    'C007D_8.0',
                    'C007D_9.0',
                    'C007D_10.0',
                    'C007D_11.0',
                    'C007D_12.0',
                    'C007D_13.0',
                    'C007D_14.0',
                    'C007D_15.0',
                    'C007D_16.0',
                    'C007D_17.0',
                    'C007D_18.0',
                    'C007D_19.0',
                    'C007D_20.0',
                    'C007D_21.0',
                    'C007D_22.0',
                    'C007D_23.0',
                    'C007D_24.0',
                    'C007D_25.0',
                    'C007D_99.0',
                    'C010_1.0',
                    'C010_99.0',
                    'SOMENTE_BENEFICIO_N/A',
                    'SOMENTE_BENEFICIO_Sim',
                    'NAO_REMUNERADO_N/A',
                    'NAO_REMUNERADO_Sim',
                    'C014_1.0',
                    'C014_2.0',
                    'C014_99.0']

    # Aplicando padronização
    data[numeric_cols] = scaler.transform(data[numeric_cols])

    return data

#########################################################################################################################

# Interface do Streamlit
# Configuração da aba do Streamlit
st.set_page_config(page_title="Projeto 1 - PNAD-COVID", page_icon="📊", layout="wide", initial_sidebar_state= "expanded")

# Exibe o logo no canto superior esquerdo e na sidebar
st.logo("../logo/logo.png")
st.image("../logo/logo.png")


# Título centralizado
st.title("Data Science Academy - Pós-Graduação em Engenharia de Machine Learning")
st.title("Projeto PNAD-COVID - Auxlílio Emergencial")
st.caption("Este app tentará prever se uma determinada pessoa recebeu, ou não, o Auxílio Emergencial do Governo Federal durante a pandemia do COVID-19")

# Barra lateral com instruções
with st.sidebar:
    
    # Define o título da barra lateral
    st.title("Instruções")
    
    # Mostra um texto explicativo
    st.markdown("Este app tentará prever se uma determinada pessoa recebeu, ou não, o **Auxílio Emergencial do Governo Federal** durante a pandemia do novo coronavírus (COVID-19), dadas as suas características socioeconômicas.")
    
    # Mostra um texto explicativo
    st.markdown("---")
    st.markdown("Fonte de dados para o treinamento do modelo: ")
    st.markdown("🔗 [Microdados da PNAD-COVID](https://www.ibge.gov.br/estatisticas/investigacoes-experimentais/estatisticas-experimentais/27946-divulgacao-semanal-pnadcovid1.html?=&t=microdados)")

    # Adiciona linhas divisórias e explicações extras na barra lateral
    st.markdown("---")
    st.markdown("Preencha os campos ao lado e clique no botão **'Prever'**, ao final da página.   =========== >  ")
 
    # Botão de link para enviar e-mail ao suporte
    st.link_button("✉️ Contacte-me", "mailto:chris.francischetti@gmail.com")


# Criação de campos para entrada de dados
st.markdown("---")
st.write("**QUESITOS  DO  QUESTIONÁRIO:**")
st.write("")

# Campos Categóricos:
st.write("")
idade = st.selectbox('**(1/24) - QUAL ERA SUA FAIXA ETÁRIA DURANTE A PANDEMIA?**', ['5-11', '12-17', '18-49', '50-64', '65+', '-5'])

st.write("")
cor = st.selectbox('**(2/24) - QUAL SUA COR/RAÇA?**', ['Branca', 'Preta', 'Amarela', 'Parda', 'Indígena', 'Ignorado'])

st.write("")
escolaridade = st.selectbox('**(3/24) - QUAL SUA ESCOLARIDADE?**', ['Sem instrução ou Fundamental Incompleto', \
            'Fundamental Completo ou Médio Incompleto', \
                'Médio Completo ou Superior Incompleto', \
                 'Superior Completo', \
                    'Pós-graduação'])

st.write("")
sexo = st.selectbox('**(4/24) - QUAL É SUA ORIENTAÇÃO SEXUAL?**', ['Homem', 'Mulher'])

st.write("")
uf = st.selectbox('**(5/24) - UNIDADE DA FEDERAÇÃO (UF) ONDE RESIDIA/RESIDE:**', ['RO', 'AC', 'AM', 'RR', 'PA', 'AP', 'TO', 'MA','PI', 'CE', 'RN', 'PB', \
                                            'PE', 'AL', 'SE', 'BA', 'MG', 'ES', 'RJ', 'SP', 'PR', 'SC', 'RS', 'MS', \
                                                'MT', 'GO', 'DF'])

st.write("")
v1023 = st.selectbox('**(6/24) - TIPO DE ÁREA:**', ['1- Capital', '2- Resto Região Metropolitana', '3- Resto da RIDE', '4- Resto da UF'])

st.write("")
v1022 = st.selectbox('**(7/24) - SITUAÇÃO DO DOMICÍLIO:**', ['1- Urbano', '2- Rural'])

st.write("")
c014 = st.selectbox('**(8/24) - CONTRIBUIA COM O INSS?**', ['1- Sim', '2- Não', '99- N/A'])

st.write("")
estava_trabalhando = st.selectbox('**(9/24) - ESTAVA TRABALHANDO OU FAZENDO ALGUM BICO?**' , ['Sim', 'Não', 'N/A'])

st.write("")
carteira_assinada = st.selectbox('**(10/24) - ESTAVA COM CARTEIRA DE TRABALHO ASSINADA OU ERA FUNCIONÁRIO PÚBLICO/ ESTATUTÁRIO?**', ['Sim, carteira de trab assinada', 'Sim, servidor púb estatutário', 'Não', 'N/A'])

st.write("")
tipo_emprego = st.selectbox('**(11/24) - QUE TIPO DE EMPREGO TINHA NA ÉPOCA?**', ['Trabalhador doméstico (empregado doméstico, cuidados, babá)', \
     'Militar', \
        'Policial militar ou bombeiro militar', \
            'Empregado do setor privado', \
                'Empregado do setor público', \
                    'Empregador', \
                        'Autônomo (Conta própria)', \
                            'Trabalhador familiar não remunerado em ajuda a parente', \
                                'Estava fora do mercado de trabalho', 'N/A'])

st.write("")
c007c = st.selectbox('**(12/24) - QUE TIPO DE TRABALHO, CARGO OU FUNÇÃO ESTAVA EXERCENDO NO SEU TRABALHO (ÚNICO OU PRINCIPAL)?**', ['01- Empregado doméstico, diarista, cozinheiro (em domicílios particulares)', \
    '02- Faxineiro, auxiliar de limpeza etc. (em empresa pública ou privada)', \
        '03- Auxiliar de escritório, escriturário', \
            '04- Secretária, recepcionista', \
                '05- Operador de Telemarketing', \
                    '06- Comerciante (dono do bar, da loja etc.)', \
                        '07- Balconista, vendedor de loja', \
                            '08- Vendedor a domicílio, representante de vendas, vendedor de catálogo (Avon, Natura etc.)', \
                                '09- Vendedor ambulante (feirante, camelô, comerciante de rua, quiosque)', \
                                    '10- Cozinheiro e garçom (de restaurantes, empresas)', \
                                        '11- Padeiro, açougueiro e doceiro', \
                                            '12- Agricultor, criador de animais, pescador, silvicultor e jardineiro', \
                                                '13- Auxiliar da agropecuária (colhedor de frutas, boia fria, etc.)', \
                                                    '14- Motorista (de aplicativo, de taxi, de van, de mototáxi, de ônibus)', \
                                                        '15- Motorista de caminhão (caminhoneiro)', \
                                                            '16- Motoboy', \
                                                                '17- Entregador de mercadorias (de restaurante, de farmácia, de loja, Uber Eats, IFood, Rappy etc.)', \
                                                                    '18- Pedreiro, servente de pedreiro, pintor, eletricista, marceneiro', \
                                                                        '19- Mecânico de veículos, máquinas industriais etc.', \
                                                                            '20- Artesão, costureiro e sapateiro', \
                                                                                '21- Cabeleireiro, manicure e afins', \
                                                                                    '22- Operador de máquinas, montador na indústria', \
                                                                                        '23- Auxiliar de produção, de carga e descarga', \
                                                                                            '24- Professor da educação infantil, de ensino fundamental, médio ou superior', \
                                                                                                '25- Pedagogo, professor de idiomas, música, arte e reforço escolar', \
                                                                                                    '26- Médico, enfermeiro, profissionais de saúde de nível superior', \
                                                                                                        '27- Técnico, profissional da saúde de nível médio', \
                                                                                                            '28- Cuidador de crianças, doentes ou idosos', \
                                                                                                                '29- Segurança, vigilante, outro trabalhador dos serviços de proteção', \
                                                                                                                    '30- Policial civil', \
                                                                                                                        '31- Porteiro, zelador', \
                                                                                                                            '32- Artista, religioso (padre, pastor etc.)', \
                                                                                                                                '33- Diretor, gerente, cargo político ou comissionado', \
                                                                                                                                    '34- Outra profissão de nível superior (advogado, engenheiro, contador, jornalista etc.)', \
                                                                                                                                        '35- Outro técnico ou profissional de nível médio', \
                                                                                                                                            '36- Outros', \
                                                                                                                                                'N/A'])

st.write("")
c007d = st.selectbox('**(13/24) - QUAL ERA A PRINCIPAL ATIVIDADE (RAMO) DO LOCAL/EMPRESA EM QUE TRABALHAVA?**', ['01- Agricultura, pecuária, produção florestal e pesca', \
     '02- Extração de petróleo, carvão mineral, minerais metálicos, pedra, areia, sal etc.', \
        '03- Indústria da transformação (inclusive confecção e fabricação caseira)', \
            '04- Fornecimento de eletricidade e gás, água, esgoto e coleta de lixo', \
                '05- Construção', \
                    '06- Comércio no atacado e varejo', \
                        '07- Reparação de veículos automotores e motocicletas', \
                            '08- Transporte de passageiros', \
                                '09- Transporte de mercadorias', \
                                    '10- Armazenamento, correios e serviços de entregas', \
                                        '11- Hospedagem (hotéis, pousadas etc.)', \
                                            '12- Serviço de alimentação (bares, restaurantes, ambulantes de alimentação)', \
                                                '13- Informação e comunicação (jornais, rádio e televisão, telecomunicações e informática)', \
                                                    '14- Bancos, atividades financeiras e de seguros', \
                                                        '15- Atividades imobiliárias', \
                                                            '16- Escritórios de advocacia, engenharia, publicidade e veterinária (Atividades profissionais, científicas e técnicas)', \
                                                                '17- Atividades de locação de mão de obra, segurança, limpeza, paisagismo e teleatendimento', \
                                                                    '18- Administração pública (governo federal, estadual e municipal)', \
                                                                        '19- Educação', \
                                                                            '20- Saúde humana e assistência social', \
                                                                                '21- Organizações religiosas, sindicatos e associações', \
                                                                                    '22- Atividade artísticas, esportivas e de recreação', \
                                                                                        '23- Cabeleireiros, tratamento de beleza e serviços pessoais', \
                                                                                            '24- Serviço doméstico remunerado (será imputado da posição na ocupação)', \
                                                                                                '25- Outro', \
                                                                                                    'N/A'])

st.write("")
nao_remunerado = st.selectbox('**(14/24) - NÃO ERA REMUNERADO / NÃO TRABALHAVA?**', ['N/A', 'Sim, nem trabalhava e nem era remunerado'])

st.write("")
somente_beneficio = st.selectbox('**(15/24) - RECEBIA SOMENTE EM BENEFÍCIOS?**', ['N/A', 'Sim'])

st.write("")
c010 = st.selectbox('**(16/24) - PODE RESPONDER A RESPEITO DA RENDA PROVENIENTE DE SEU(S) TRABALHO(S)?**', ['1- Sim', '99- N/A'])
# Campos Numéricos:
st.write("")
salario = st.number_input('**(17/24) - QUANTO RECEBIA/RETIRAVA DE SALÁRIO, EM TODOS OS TRABALHOS, DURANTE A PANDEMIDA DO COVID-19?**', min_value = 0, max_value = 500000, value = 1045)

st.write("")
valor_ticket_cesta = st.number_input('**(18/24) - QUANTO RECEBIA EM PRODUTOS/MERCADORIAS (TICKET-ALIMENTAÇÃO)?**', min_value = 0, max_value = 50000, value = 0)

st.write("")
soma_bolsas_familia = st.number_input('**(19/24) - QUANTO RECEBIA DE RENDA PROVENIENTE DO PROGRAMA BOLSA FAMÍLIA?**', min_value = 0, max_value = 1000, value = 0)

st.write("")
soma_bpc_loas = st.number_input('**(20/24) - QUANTO RECEBIA DE RENDA PROVENIENTE DO PROGRAMA BPC-LOAS?**', min_value = 0, max_value = 5000, value = 0)

st.write("")
soma_seguros_desemprego = st.number_input('**(21/24) - QUANTO RECEBIA DE RENDA PROVENIENTE DO SEGURO DESEMPREGO?**', min_value = 0, max_value = 9000, value = 0)

st.write("")
soma_aposentadorias_mais_pensao = st.number_input('**(22/24) - QUAL ERA O SOMATÓRIO DE TODOS OS RENDIMENTOS PROVENIENTES DE APOSENTADORIA/PENSÃO NO DOMICÍLIO?**', min_value = 0, max_value = 50000, value = 0)

st.write("")
d0023 = st.number_input('**(23/24) - QUAL O SOMATÓRIO DE PENSÃO ALIMENTÍCIA/DOAÇÃO OU MESADA DE ALGUÉM DE FORA DO DOMIÍCIO, EM SUA RESIDÊNCIA?**', min_value = 0, max_value = 20000, value = 0)

st.write("")
d0073 = st.number_input('**(24/24) - INFORME A SOMA DOS VALORES DE OUTRAS RENDAS DO DOMICÍLIO (EX: ALUGUEL, PREVIDÊNCIA PRIVADA, RENDIMENTOS DE POUPANÇA)**', min_value = 0, max_value = 50000, value = 0)



# Botão para realizar previsões
st.write("")
st.write("")
if st.button('Prever'):

    # Ajusta as variáveis pré-processadas com One-Hot Encoding

    # Idade
    idade_menor_5 =    1 if idade == '-5' else 0
    idade_12_17 = 1 if idade == '12-17' else 0
    idade_18_49 = 1 if idade == '18-49' else 0
    idade_5_11 =  1 if idade == '5-11' else 0
    idade_50_64 = 1 if idade == '50-64' else 0
    idade_65_mais =   1 if idade == '65+' else 0

    # Cor
    cor_amarela = 1 if cor == 'Amarela' else 0
    cor_branca = 1 if cor == 'Branca' else 0
    cor_ignorado = 1 if cor == 'Ignorado' else 0
    cor_indigena = 1 if cor == 'Indígena' else 0
    cor_parda = 1 if cor == 'Parda' else 0
    cor_preta = 1 if cor == 'Preta' else 0

    # Escolaridade
    escolaridade_fundamental_completo_ou_medio_incompleto = 1 if escolaridade == 'Fundamental Completo ou Médio Incompleto' else 0
    escolaridade_medio_completo_ou_superior_incompleto = 1 if escolaridade == 'Médio Completo ou Superior Incompleto' else 0
    escolaridade_pos_graduacao = 1 if escolaridade == 'Pós-graduação' else 0
    escolaridade_sem_instrucao_ou_fundamental_incompleto = 1 if escolaridade == 'Sem instrução ou Fundamental Incompleto' else 0
    escolaridade_superior_completo = 1 if escolaridade == 'Superior Completo' else 0  

    # Sexo
    sexo_homem = 1 if sexo == 'Homem' else 0
    sexo_mulher = 1 if sexo == 'Mulher' else 0

    # UF
    uf_ac = 1 if uf == 'AC' else 0
    uf_al = 1 if uf == 'AL' else 0
    uf_am = 1 if uf == 'AM' else 0
    uf_ap = 1 if uf == 'AP' else 0
    uf_ba = 1 if uf == 'BA' else 0
    uf_ce = 1 if uf == 'CE' else 0
    uf_df = 1 if uf == 'DF' else 0
    uf_es = 1 if uf == 'ES' else 0
    uf_go = 1 if uf == 'GO' else 0
    uf_ma = 1 if uf == 'MA' else 0
    uf_mg = 1 if uf == 'MG' else 0
    uf_ms = 1 if uf == 'MS' else 0
    uf_mt = 1 if uf == 'MT' else 0
    uf_pa = 1 if uf == 'PA' else 0
    uf_pb = 1 if uf == 'PB' else 0
    uf_pe = 1 if uf == 'PE' else 0
    uf_pi = 1 if uf == 'PI' else 0
    uf_pr = 1 if uf == 'PR' else 0
    uf_rj = 1 if uf == 'RJ' else 0
    uf_rn = 1 if uf == 'RN' else 0
    uf_ro = 1 if uf == 'RO' else 0
    uf_rr = 1 if uf == 'RR' else 0
    uf_rs = 1 if uf == 'RS' else 0
    uf_sc = 1 if uf == 'SC' else 0
    uf_se = 1 if uf == 'SE' else 0
    uf_sp = 1 if uf == 'SP' else 0
    uf_to = 1 if uf == 'TO' else 0

    # V1023 - Tipo de Área
    v1023_1 = 1 if v1023 == '1- Capital' else 0
    v1023_2 = 1 if v1023 == '2- Resto Região Metropolitana' else 0
    v1023_3 = 1 if v1023 == '3- Resto da RIDE' else 0
    v1023_4 = 1 if v1023 == '4- Resto da UF' else 0

    # V1022 - Situação do Domicílio
    v1022_1 = 1 if v1022 == '1- Urbano' else 0
    v1022_2 = 1 if v1022 == '2- Rural' else 0

    # C014 - Contribui com o INSS
    c014_1 = 1 if c014 == '1- Sim' else 0
    c014_2 = 1 if c014 == '2- Não' else 0
    c014_99 = 1 if c014 == '99- N/A' else 0

    # Estava Trabalhando
    estava_trabalhando_na = 1 if estava_trabalhando == 'N/A' else 0
    estava_trabalhando_nao = 1 if estava_trabalhando == 'Não' else 0
    estava_trabalhando_sim = 1 if estava_trabalhando == 'Sim' else 0

    # Carteira Assinada
    carteira_assinada_na = 1 if carteira_assinada == 'N/A' else 0
    carteira_assinada_nao = 1 if carteira_assinada == 'Não' else 0
    carteira_assinada_sim_carteira_de_trab_assinada = 1 if carteira_assinada == 'Sim, carteira de trab assinada' else 0
    carteira_assinada_sim_servidor_pub_estatutario = 1 if carteira_assinada == 'Sim, servidor púb estatutário' else 0

    # Tipo de Emprego
    tipo_emprego_autonomo__conta_propria_ = 1 if tipo_emprego == 'Autônomo (Conta própria)' else 0
    tipo_emprego_empregado_do_setor_privado = 1 if tipo_emprego == 'Empregado do setor privado' else 0
    tipo_emprego_empregado_do_setor_publico = 1 if tipo_emprego == 'Empregado do setor público' else 0
    tipo_emprego_empregador = 1 if tipo_emprego == 'Empregador' else 0
    tipo_emprego_estava_fora_do_mercado_de_trabalho = 1 if tipo_emprego == 'Estava fora do mercado de trabalho' else 0
    tipo_emprego_militar = 1 if tipo_emprego == 'Militar' else 0
    tipo_emprego_na = 1 if tipo_emprego == 'N/A' else 0
    tipo_emprego_policial_militar_ou_bombeiro_militar = 1 if tipo_emprego == 'Policial militar ou bombeiro militar' else 0
    tipo_emprego_trabalhador_domestico__empregado_domestico_cuidados_baba_ = 1 if tipo_emprego == 'Trabalhador doméstico (empregado doméstico, cuidados, babá)' else 0
    tipo_emprego_trabalhador_familiar_nao_remunerado_em_ajuda_a_parente = 1 if tipo_emprego == 'Trabalhador familiar não remunerado em ajuda a parente' else 0

    # C007C - Função no Emprego
    c007c_1 = 1 if c007c == '01- Empregado doméstico, diarista, cozinheiro (em domicílios particulares)' else 0
    c007c_2 = 1 if c007c == '02- Faxineiro, auxiliar de limpeza etc. (em empresa pública ou privada)' else 0
    c007c_3 = 1 if c007c == '03- Auxiliar de escritório, escriturário' else 0
    c007c_4 = 1 if c007c == '04- Secretária, recepcionista' else 0
    c007c_5 = 1 if c007c == '05- Operador de Telemarketing' else 0
    c007c_6 = 1 if c007c == '06- Comerciante (dono do bar, da loja etc.)' else 0
    c007c_7 = 1 if c007c == '07- Balconista, vendedor de loja' else 0
    c007c_8 = 1 if c007c == '08- Vendedor a domicílio, representante de vendas, vendedor de catálogo (Avon, Natura etc.)' else 0
    c007c_9 = 1 if c007c == '09- Vendedor ambulante (feirante, camelô, comerciante de rua, quiosque)' else 0
    c007c_10 = 1 if c007c == '10- Cozinheiro e garçom (de restaurantes, empresas)' else 0
    c007c_11 = 1 if c007c == '11- Padeiro, açougueiro e doceiro' else 0
    c007c_12 = 1 if c007c == '12- Agricultor, criador de animais, pescador, silvicultor e jardineiro' else 0
    c007c_13 = 1 if c007c == '13- Auxiliar da agropecuária (colhedor de frutas, boia fria, etc.)' else 0
    c007c_14 = 1 if c007c == '14- Motorista (de aplicativo, de taxi, de van, de mototáxi, de ônibus)' else 0
    c007c_15 = 1 if c007c == '15- Motorista de caminhão (caminhoneiro)' else 0
    c007c_16 = 1 if c007c == '16- Motoboy' else 0
    c007c_17 = 1 if c007c == '17- Entregador de mercadorias (de restaurante, de farmácia, de loja, Uber Eats, IFood, Rappy etc.)' else 0
    c007c_18 = 1 if c007c == '18- Pedreiro, servente de pedreiro, pintor, eletricista, marceneiro' else 0
    c007c_19 = 1 if c007c == '19- Mecânico de veículos, máquinas industriais etc.' else 0
    c007c_20 = 1 if c007c == '20- Artesão, costureiro e sapateiro' else 0
    c007c_21 = 1 if c007c == '21- Cabeleireiro, manicure e afins' else 0
    c007c_22 = 1 if c007c == '22- Operador de máquinas, montador na indústria' else 0
    c007c_23 = 1 if c007c == '23- Auxiliar de produção, de carga e descarga' else 0
    c007c_24 = 1 if c007c == '24- Professor da educação infantil, de ensino fundamental, médio ou superior' else 0
    c007c_25 = 1 if c007c == '25- Pedagogo, professor de idiomas, música, arte e reforço escolar' else 0
    c007c_26 = 1 if c007c == '26- Médico, enfermeiro, profissionais de saúde de nível superior' else 0
    c007c_27 = 1 if c007c == '27- Técnico, profissional da saúde de nível médio' else 0
    c007c_28 = 1 if c007c == '28- Cuidador de crianças, doentes ou idosos' else 0
    c007c_29 = 1 if c007c == '29- Segurança, vigilante, outro trabalhador dos serviços de proteção' else 0
    c007c_30 = 1 if c007c == '30- Policial civil' else 0
    c007c_31 = 1 if c007c == '31- Porteiro, zelador' else 0
    c007c_32 = 1 if c007c == '32- Artista, religioso (padre, pastor etc.)' else 0
    c007c_33 = 1 if c007c == '33- Diretor, gerente, cargo político ou comissionado' else 0
    c007c_34 = 1 if c007c == '34- Outra profissão de nível superior (advogado, engenheiro, contador, jornalista etc.)' else 0
    c007c_35 = 1 if c007c == '35- Outro técnico ou profissional de nível médio' else 0
    c007c_36 = 1 if c007c == '36- Outros' else 0
    c007c_99 = 1 if c007c == 'N/A' else 0

    # c007d - Atividade da empresa
    c007d_1 = 1 if c007d == '01- Agricultura, pecuária, produção florestal e pesca' else 0
    c007d_2 = 1 if c007d == '02- Extração de petróleo, carvão mineral, minerais metálicos, pedra, areia, sal etc.' else 0
    c007d_3 = 1 if c007d == '03- Indústria da transformação (inclusive confecção e fabricação caseira)' else 0
    c007d_4 = 1 if c007d == '04- Fornecimento de eletricidade e gás, água, esgoto e coleta de lixo' else 0
    c007d_5 = 1 if c007d == '05- Construção' else 0
    c007d_6 = 1 if c007d == '06- Comércio no atacado e varejo' else 0
    c007d_7 = 1 if c007d == '07- Reparação de veículos automotores e motocicletas' else 0
    c007d_8 = 1 if c007d == '08- Transporte de passageiros' else 0
    c007d_9 = 1 if c007d == '09- Transporte de mercadorias' else 0
    c007d_10 = 1 if c007d == '10- Armazenamento, correios e serviços de entregas' else 0
    c007d_11 = 1 if c007d == '11- Hospedagem (hotéis, pousadas etc.)' else 0
    c007d_12 = 1 if c007d == '12- Serviço de alimentação (bares, restaurantes, ambulantes de alimentação)' else 0
    c007d_13 = 1 if c007d == '13- Informação e comunicação (jornais, rádio e televisão, telecomunicações e informática)' else 0
    c007d_14 = 1 if c007d == '14- Bancos, atividades financeiras e de seguros' else 0
    c007d_15 = 1 if c007d == '15- Atividades imobiliárias' else 0
    c007d_16 = 1 if c007d == '16- Escritórios de advocacia, engenharia, publicidade e veterinária (Atividades profissionais, científicas e técnicas)' else 0
    c007d_17 = 1 if c007d == '17- Atividades de locação de mão de obra, segurança, limpeza, paisagismo e teleatendimento' else 0
    c007d_18 = 1 if c007d == '18- Administração pública (governo federal, estadual e municipal)' else 0
    c007d_19 = 1 if c007d == '19- Educação' else 0
    c007d_20 = 1 if c007d == '20- Saúde humana e assistência social' else 0
    c007d_21 = 1 if c007d == '21- Organizações religiosas, sindicatos e associações' else 0
    c007d_22 = 1 if c007d == '22- Atividade artísticas, esportivas e de recreação' else 0
    c007d_23 = 1 if c007d == '23- Cabeleireiros, tratamento de beleza e serviços pessoais' else 0
    c007d_24 = 1 if c007d == '24- Serviço doméstico remunerado (será imputado da posição na ocupação)' else 0
    c007d_25 = 1 if c007d == '25- Outro' else 0
    c007d_99 = 1 if c007d == 'N/A' else 0

    # Não Remunerado
    nao_remunerado_na = 1 if nao_remunerado == 'N/A' else 0
    nao_remunerado_sim = 1 if nao_remunerado == 'Sim, nem trabalhava e nem era remunerado' else 0

    # Somente Benefícios
    somente_beneficio_na = 1 if somente_beneficio == 'N/A' else 0
    somente_beneficio_sim = 1 if somente_beneficio == 'Sim' else 0

    # C010 - Respondeu quanto recebia?
    c010_1 = 1 if c010 == '1- Sim' else 0
    c010_99 = 1 if c010 == '99- N/A' else 0


     #################################################################################################################

    # Executa a função de pré-processamento de dados
    input_data = preprocess_input(salario, 
                                  valor_ticket_cesta,
                                  soma_aposentadorias_mais_pensao,
                                  d0023,
                                  soma_bolsas_familia,
                                  soma_bpc_loas,
                                  soma_seguros_desemprego,
                                  d0073,
                                  uf_ac,
                                  uf_al,
                                  uf_am,
                                  uf_ap,
                                  uf_ba,
                                  uf_ce,
                                  uf_df,
                                  uf_es,
                                  uf_go,
                                  uf_ma,
                                  uf_mg,
                                  uf_ms,
                                  uf_mt,
                                  uf_pa,
                                  uf_pb,
                                  uf_pe,
                                  uf_pi,
                                  uf_pr,
                                  uf_rj,
                                  uf_rn,
                                  uf_ro,
                                  uf_rr,
                                  uf_rs,
                                  uf_sc,
                                  uf_se,
                                  uf_sp,
                                  uf_to,
                                  v1022_1,
                                  v1022_2,
                                  v1023_1,
                                  v1023_2,
                                  v1023_3,
                                  v1023_4,
                                  idade_menor_5,
                                  idade_12_17,
                                  idade_18_49,
                                  idade_5_11,
                                  idade_50_64,
                                  idade_65_mais,
                                  sexo_homem,
                                  sexo_mulher,
                                  cor_amarela,
                                  cor_branca,
                                  cor_ignorado,
                                  cor_indigena,
                                  cor_parda,
                                  cor_preta,
                                  escolaridade_fundamental_completo_ou_medio_incompleto,
                                  escolaridade_medio_completo_ou_superior_incompleto,
                                  escolaridade_pos_graduacao,
                                  escolaridade_sem_instrucao_ou_fundamental_incompleto,
                                  escolaridade_superior_completo,
                                  estava_trabalhando_na,
                                  estava_trabalhando_nao,
                                  estava_trabalhando_sim,
                                  tipo_emprego_autonomo__conta_propria_,
                                  tipo_emprego_empregado_do_setor_privado,
                                  tipo_emprego_empregado_do_setor_publico,
                                  tipo_emprego_empregador,
                                  tipo_emprego_estava_fora_do_mercado_de_trabalho,
                                  tipo_emprego_militar,
                                  tipo_emprego_na,
                                  tipo_emprego_policial_militar_ou_bombeiro_militar,
                                  tipo_emprego_trabalhador_domestico__empregado_domestico_cuidados_baba_,
                                  tipo_emprego_trabalhador_familiar_nao_remunerado_em_ajuda_a_parente,
                                  carteira_assinada_na,
                                  carteira_assinada_nao,
                                  carteira_assinada_sim_carteira_de_trab_assinada,
                                  carteira_assinada_sim_servidor_pub_estatutario,
                                  c007c_1,
                                  c007c_2,
                                  c007c_3,
                                  c007c_4,
                                  c007c_5,
                                  c007c_6,
                                  c007c_7,
                                  c007c_8,
                                  c007c_9,
                                  c007c_10,
                                  c007c_11,
                                  c007c_12,
                                  c007c_13,
                                  c007c_14,
                                  c007c_15,
                                  c007c_16,
                                  c007c_17,
                                  c007c_18,
                                  c007c_19,
                                  c007c_20,
                                  c007c_21,
                                  c007c_22,
                                  c007c_23,
                                  c007c_24,
                                  c007c_25,
                                  c007c_26,
                                  c007c_27,
                                  c007c_28,
                                  c007c_29,
                                  c007c_30,
                                  c007c_31,
                                  c007c_32,
                                  c007c_33,
                                  c007c_34,
                                  c007c_35,
                                  c007c_36,
                                  c007c_99,
                                  c007d_1,
                                  c007d_2,
                                  c007d_3,
                                  c007d_4,
                                  c007d_5,
                                  c007d_6,
                                  c007d_7,
                                  c007d_8,
                                  c007d_9,
                                  c007d_10,
                                  c007d_11,
                                  c007d_12,
                                  c007d_13,
                                  c007d_14,
                                  c007d_15,
                                  c007d_16,
                                  c007d_17,
                                  c007d_18,
                                  c007d_19,
                                  c007d_20,
                                  c007d_21,
                                  c007d_22,
                                  c007d_23,
                                  c007d_24,
                                  c007d_25,
                                  c007d_99,
                                  c010_1,
                                  c010_99,
                                  somente_beneficio_na,
                                  somente_beneficio_sim,
                                  nao_remunerado_na,
                                  nao_remunerado_sim,
                                  c014_1,
                                  c014_2,
                                  c014_99)

    # Faz a previsão com o modelo
    prediction = modelo.predict(input_data)
    prediction_proba = modelo.predict_proba(input_data)[0]

    #st.write('Previsão:' , 'Sim' if prediction[0] == 1 else 'Não')
    st.write("")
    if prediction[0] == 1:
        st.success("**Sim, esta pessoa provavelmente recebeu o Auxílio Emergencial durante a pandemia do COVID 19**")
    else:
        st.warning("**Não, provavelmente esta pessoa não recebeu o Auxílio Emergencial**")

    st.markdown("---")
    st.subheader("Probabilidades por classe:")
    st.write(f"- **Probabilidade de ter recebido:** {prediction_proba[0]*100:.2f}%")
    st.write(f"- **Probabilidade de não ter recebido:** {prediction_proba[1]*100:.2f}%")

    st.write("")
    st.write("")
    st.write("")
    st.write('Obrigado por testar!')
    st.caption('By Christopherson B. Francischetti')
    st.markdown(
    """
    <div style="text-align: center; color: gray;">
        <hr>
        <p>Atenção: Este modelo está em fase beta! Modelo de Machine Learning criado com base nos microdados do IBGE disponibilizados na PNAD-COVID19.</p>
    </div>
    """,
    unsafe_allow_html=True
    )

# FIM


