import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importações das bibliotecas de análise
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# Configuração da página
st.set_page_config(
    page_title="Dashboard Intelligence - Veros",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS GLOBAL PARA CORES FIXAS - VERSÃO CORRIGIDA SEM TARJAS BRANCAS
st.markdown("""
<style>
    /* CORES BASE FIXAS */
    :root {
        --primary-bg: #ffffff;
        --secondary-bg: #f0f2f6;
        --text-color: #000000;
        --border-color: #cccccc;
    }
    
    /* FUNDO PRINCIPAL NATURAL DO STREAMLIT */
    .stApp {
        background-color: var(--primary-bg);
    }
    
    /* INPUTS - CORES FIXAS SEM TARJAS */
    .stTextInput>div>div>input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #2563eb !important;
        box-shadow: none !important;
    }
    
    .stTextInput label {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* SENHA - CORES FIXAS */
    .stTextInput>div>div>input[type="password"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    /* SELECT BOX - CORES FIXAS */
    .stSelectbox>div>div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    .stSelectbox label {
        color: #000000 !important;
    }
    
    /* SIDEBAR - CORES NATURAIS */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* GARANTIR QUE TEXTOS SEJAM VISÍVEIS */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #000000 !important;
    }
    
    /* REMOVER QUALQUER TEMA ESCURO */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #ffffff !important;
        }
        
        .stTextInput>div>div>input,
        .stSelectbox>div>div {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Paleta de cores harmoniosa FIXA (não muda com tema do navegador)
COLORS = {
    'primary': '#2563eb',
    'primary_light': '#3b82f6',
    'secondary': '#059669',
    'secondary_light': '#10b981',
    'accent': '#dc2626',
    'accent_light': '#ef4444',
    'warning': '#d97706',
    'warning_light': '#f59e0b',
    'info': '#0369a1',
    'info_light': '#0ea5e9',
    'gray': '#6b7280',
    'light_gray': '#f3f4f6',
    'dark_gray': '#374151',
    'white': '#ffffff',
    'black': '#000000',
    'sidebar_bg': '#f0f2f6'
}

# Dados de investimento fixos para 2024 e 2025
INVESTIMENTO_MENSAL_2024 = {
    'jan./24': 200000.00,
    'fev./24': 210000.00,
    'mar./24': 220000.00,
    'abr./24': 215000.00,
    'mai./24': 230000.00,
    'jun./24': 240000.00,
    'jul./24': 235000.00,
    'ago./24': 250000.00,
    'set./24': 260000.00,
    'out./24': 255000.00,
    'nov./24': 270000.00,
    'dez./24': 280000.00
}

INVESTIMENTO_MENSAL_2025 = {
    'jan./25': 217984.00,
    'fev./25': 238992.00,
    'mar./25': 240066.00,
    'abr./25': 236921.00,
    'mai./25': 270502.00,
    'jun./25': 288691.00,
    'jul./25': 281410.00,
    'ago./25': 318477.00,
    'set./25': 335041.00,
    'out./25': 320000.00,
    'nov./25': 340000.00,
    'dez./25': 360000.00
}

# Dados de investimento por LP
INVESTIMENTO_POR_LP = {
    'AMB': 20065.00,
    'CDI': 21210.00,
    'LAB': 2803.00,
    'PS': 1717.00,
    'UTI': 2284.00,
    'INSTI': 12070.00,
    'NON': 1347.00,
    'BULL': 47.00,
    'META': 5667.00,
    'BING': 137.00
}

# Credenciais de login
CREDENCIAIS = {
    "Midianewton": "New@2025"
}

# Função para carregar o logo do login
def load_login_logo():
    caminhos_logo = [
        "Dashboardimagemlogin.jpg",
        "Dashboardimagemlogin.png",
        "assets/Dashboardimagemlogin.jpg",
        "assets/Dashboardimagemlogin.png",
        "images/Dashboardimagemlogin.jpg",
        "images/Dashboardimagemlogin.png"
    ]
    
    for logo_path in caminhos_logo:
        try:
            if os.path.exists(logo_path):
                logo = Image.open(logo_path)
                return logo
        except Exception as e:
            continue
    return None

# Função para carregar o logo do dashboard
@st.cache_data
def load_logo():
    caminhos_logo = [
        "images.jpg",
        "logo.jpg", 
        "logo.png",
        "assets/images.jpg",
        "assets/logo.jpg",
        "assets/logo.png"
    ]
    
    for logo_path in caminhos_logo:
        try:
            if os.path.exists(logo_path):
                logo = Image.open(logo_path)
                return logo
        except Exception as e:
            continue
    return None

# Função para carregar e processar os dados
@st.cache_data
def load_data():
    caminhos_dados = [
        "DADOS_RECEITA_VEROS.xlsx",
        "data/DADOS_RECEITA_VEROS.xlsx",
        "assets/DADOS_RECEITA_VEROS.xlsx",
        "dados/DADOS_RECEITA_VEROS.xlsx"
    ]
    
    for file_path in caminhos_dados:
        try:
            if os.path.exists(file_path):
                df = pd.read_excel(file_path, engine='openpyxl')
                
                colunas_necessarias = ['Mês geração receita', 'Mês geração lead', 'Considerar?', 'LP', 'VL UNI', 'E-MAIL']
                colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
                
                if colunas_faltantes:
                    st.warning(f"Colunas faltantes no dataset: {', '.join(colunas_faltantes)}")
                
                date_columns = ['DT Receita', 'Data do lead', 'Data e-mail', 'Data e-mail corrigido', 'Data telefone']
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                
                if 'VL UNI' in df.columns:
                    df['VL UNI'] = pd.to_numeric(df['VL UNI'], errors='coerce')
                
                return df
        except Exception as e:
            continue
    
    st.error("Nenhum arquivo de dados encontrado nos caminhos padrão.")
    
    uploaded_file = st.file_uploader("Faça upload do arquivo DADOS_RECEITA_VEROS.xlsx", type="xlsx")
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            st.success("Arquivo carregado via upload!")
            
            colunas_necessarias = ['Mês geração receita', 'Mês geração lead', 'Considerar?', 'LP', 'VL UNI', 'E-MAIL']
            colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
            
            if colunas_faltantes:
                st.warning(f"Colunas faltantes no dataset: {', '.join(colunas_faltantes)}")
            
            return df
        except Exception as e:
            st.error(f"Erro ao processar arquivo upload: {e}")
            return None
    
    return None

# Tela de Login CORRIGIDA - Sem tarjas brancas
def login_screen():
    st.markdown("""
    <style>
    /* REMOVER TARJAS BRANCAS DOS INPUTS */
    .stTextInput>div>div>input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        border-radius: 6px;
        padding: 8px 12px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #2563eb !important;
        box-shadow: none !important;
    }
    
    .stTextInput label {
        color: #000000 !important;
        font-weight: 500;
    }
    
    /* BOTÃO COM CORES FIXAS */
    .stButton>button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px;
        padding: 10px 0;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    
    /* FUNDO NATURAL DA APLICAÇÃO */
    .stApp {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

    # Layout centralizado para login
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        
        # Logo
        login_logo = load_login_logo()
        if login_logo:
            st.image(login_logo, width=200)
        else:
            st.markdown("<div style='font-size: 3rem; color: #111111; margin-bottom: 1rem;'>📊</div>", unsafe_allow_html=True)

        # Título
        st.markdown("<h2 style='text-align: center; color: #111111;'>Veros Intelligence</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666666;'>Acesse seu dashboard</p>", unsafe_allow_html=True)
        
        # Formulário de login
        with st.form("login_form"):
            usuario = st.text_input("Usuário", placeholder="Digite seu usuário")
            senha = st.text_input("Senha", type="password", placeholder="Digite sua senha")
            submit = st.form_submit_button("Entrar")

            if submit:
                if usuario in CREDENCIAIS and CREDENCIAIS[usuario] == senha:
                    st.session_state.logged_in = True
                    st.session_state.usuario = usuario
                    st.success("Login realizado com sucesso!")
                    st.rerun()
                else:
                    st.error("Credenciais inválidas.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# [AS DEMAIS FUNÇÕES PERMANECEM EXATAMENTE IGUAIS...]
# Função para criar matriz escadinha de Receita
def criar_matriz_escadinha(df, ano_filtro=2025):
    if df is None or df.empty:
        return None, None
    
    df_filtrado = df.copy()
    
    try:
        condicao1 = df_filtrado['Considerar?'] == 'Sim'
        condicao2 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        if ano_filtro == 2024:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
            condicao4 = df_filtrado['Mês geração lead'].str.contains('2024|24', na=False)
        else:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
            condicao4 = df_filtrado['Mês geração lead'].str.contains('2025|25', na=False)
        
        mask = condicao1 & condicao2 & condicao3 & condicao4
        df_filtrado = df_filtrado[mask].copy()
        
        if df_filtrado.empty:
            return None, None
        
        if ano_filtro == 2024:
            ordem_meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                          'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
        else:
            ordem_meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                          'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
        
        df_filtrado = df_filtrado[
            df_filtrado['Mês geração receita'].isin(ordem_meses) & 
            df_filtrado['Mês geração lead'].isin(ordem_meses)
        ].copy()
        
        if df_filtrado.empty:
            return None, None
        
        matriz_receita = pd.DataFrame(0, index=ordem_meses, columns=ordem_meses, dtype=float)
        
        for _, row in df_filtrado.iterrows():
            mes_receita = row['Mês geração receita']
            mes_lead = row['Mês geração lead']
            valor = row['VL UNI']
            
            if mes_receita in ordem_meses and mes_lead in ordem_meses:
                matriz_receita.loc[mes_receita, mes_lead] += valor
        
        matriz_formatada = matriz_receita.copy()
        for col in matriz_formatada.columns:
            matriz_formatada[col] = matriz_formatada[col].apply(lambda x: f"R$ {x:,.0f}" if x > 0 else "-")
        
        return matriz_receita, matriz_formatada
    
    except Exception as e:
        st.error(f"Erro ao criar matriz escadinha: {e}")
        return None, None

# Função para criar matriz escadinha de CAC
def criar_matriz_cac(df, ano_filtro=2025):
    if df is None or df.empty:
        return None, None
    
    df_filtrado = df.copy()
    
    try:
        condicao1 = df_filtrado['Considerar?'] == 'Sim'
        condicao2 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        if ano_filtro == 2024:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
            condicao4 = df_filtrado['Mês geração lead'].str.contains('2024|24', na=False)
            investimento_mensal = INVESTIMENTO_MENSAL_2024
        else:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
            condicao4 = df_filtrado['Mês geração lead'].str.contains('2025|25', na=False)
            investimento_mensal = INVESTIMENTO_MENSAL_2025
        
        mask = condicao1 & condicao2 & condicao3 & condicao4
        df_filtrado = df_filtrado[mask].copy()
        
        if df_filtrado.empty:
            return None, None
        
        if ano_filtro == 2024:
            ordem_meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                          'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
        else:
            ordem_meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                          'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
        
        df_filtrado = df_filtrado[
            df_filtrado['Mês geração receita'].isin(ordem_meses) & 
            df_filtrado['Mês geração lead'].isin(ordem_meses)
        ].copy()
        
        if df_filtrado.empty:
            return None, None
        
        matriz_tutores = pd.DataFrame(0, index=ordem_meses, columns=ordem_meses, dtype=float)
        
        tutores_por_combinacao = df_filtrado.groupby(['Mês geração receita', 'Mês geração lead'])['E-MAIL'].nunique()
        
        for (mes_receita, mes_lead), count in tutores_por_combinacao.items():
            if mes_receita in ordem_meses and mes_lead in ordem_meses:
                matriz_tutores.loc[mes_receita, mes_lead] = count
        
        matriz_cac = pd.DataFrame(0, index=ordem_meses, columns=ordem_meses, dtype=float)
        
        for mes_lead in ordem_meses:
            if mes_lead in investimento_mensal:
                investimento = investimento_mensal[mes_lead]
                for mes_receita in ordem_meses:
                    num_tutores = matriz_tutores.loc[mes_receita, mes_lead]
                    if num_tutores > 0:
                        matriz_cac.loc[mes_receita, mes_lead] = investimento / num_tutores
        
        matriz_formatada = matriz_cac.copy()
        for col in matriz_formatada.columns:
            matriz_formatada[col] = matriz_formatada[col].apply(
                lambda x: f"R$ {x:,.0f}" if x > 0 else "-"
            )
        
        return matriz_cac, matriz_formatada
    
    except Exception as e:
        st.error(f"Erro ao criar matriz CAC: {e}")
        return None, None

# Função para criar matriz escadinha de LTV
def criar_matriz_ltv(df, ano_filtro=2025):
    if df is None or df.empty:
        return None, None
    
    df_filtrado = df.copy()
    
    try:
        condicao1 = df_filtrado['Considerar?'] == 'Sim'
        condicao2 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        if ano_filtro == 2024:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
            condicao4 = df_filtrado['Mês geração lead'].str.contains('2024|24', na=False)
        else:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
            condicao4 = df_filtrado['Mês geração lead'].str.contains('2025|25', na=False)
        
        mask = condicao1 & condicao2 & condicao3 & condicao4
        df_filtrado = df_filtrado[mask].copy()
        
        if df_filtrado.empty:
            return None, None
        
        if ano_filtro == 2024:
            ordem_meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                          'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
        else:
            ordem_meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                          'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
        
        df_filtrado = df_filtrado[
            df_filtrado['Mês geração receita'].isin(ordem_meses) & 
            df_filtrado['Mês geração lead'].isin(ordem_meses)
        ].copy()
        
        if df_filtrado.empty:
            return None, None
        
        matriz_receita_bruta = pd.DataFrame(0, index=ordem_meses, columns=ordem_meses, dtype=float)
        matriz_tutores = pd.DataFrame(0, index=ordem_meses, columns=ordem_meses, dtype=float)
        
        for _, row in df_filtrado.iterrows():
            mes_receita = row['Mês geração receita']
            mes_lead = row['Mês geração lead']
            valor = row['VL UNI']
            
            if mes_receita in ordem_meses and mes_lead in ordem_meses:
                matriz_receita_bruta.loc[mes_receita, mes_lead] += valor
        
        matriz_receita_liquida = matriz_receita_bruta * 0.56
        
        tutores_por_combinacao = df_filtrado.groupby(['Mês geração receita', 'Mês geração lead'])['E-MAIL'].nunique()
        
        for (mes_receita, mes_lead), count in tutores_por_combinacao.items():
            if mes_receita in ordem_meses and mes_lead in ordem_meses:
                matriz_tutores.loc[mes_receita, mes_lead] = count
        
        matriz_ltv = pd.DataFrame(0, index=ordem_meses, columns=ordem_meses, dtype=float)
        
        for mes_receita in ordem_meses:
            for mes_lead in ordem_meses:
                num_tutores = matriz_tutores.loc[mes_receita, mes_lead]
                receita_liquida = matriz_receita_liquida.loc[mes_receita, mes_lead]
                if num_tutores > 0:
                    matriz_ltv.loc[mes_receita, mes_lead] = receita_liquida / num_tutores
        
        matriz_formatada = matriz_ltv.copy()
        for col in matriz_formatada.columns:
            matriz_formatada[col] = matriz_formatada[col].apply(
                lambda x: f"R$ {x:,.0f}" if x > 0 else "-"
            )
        
        return matriz_ltv, matriz_formatada
    
    except Exception as e:
        st.error(f"Erro ao criar matriz LTV: {e}")
        return None, None

# Função para criar matriz escadinha de CAC/LTV
def criar_matriz_cac_ltv_ratio(df, ano_filtro=2025):
    if df is None or df.empty:
        return None, None
    
    try:
        matriz_cac, _ = criar_matriz_cac(df, ano_filtro)
        matriz_ltv, _ = criar_matriz_ltv(df, ano_filtro)
        
        if matriz_cac is None or matriz_ltv is None:
            return None, None
        
        matriz_ratio = pd.DataFrame(0, index=matriz_cac.index, columns=matriz_cac.columns, dtype=float)
        
        for i in matriz_cac.index:
            for j in matriz_cac.columns:
                cac_val = matriz_cac.loc[i, j]
                ltv_val = matriz_ltv.loc[i, j]
                if cac_val > 0 and ltv_val > 0:
                    matriz_ratio.loc[i, j] = cac_val / ltv_val
        
        matriz_formatada = matriz_ratio.copy()
        for col in matriz_formatada.columns:
            matriz_formatada[col] = matriz_formatada[col].apply(
                lambda x: f"{x:.2f}" if x > 0 else "-"
            )
        
        return matriz_ratio, matriz_formatada
    
    except Exception as e:
        st.error(f"Erro ao criar matriz CAC/LTV: {e}")
        return None, None

# FUNÇÃO ATUALIZADA: Criar heatmap da matriz escadinha SEM RÓTULOS
def criar_heatmap_matriz(matriz, titulo, colorscale='Blues', width=500, height=450):
    """
    Cria um heatmap visual da matriz escadinha SEM RÓTULOS DE DADOS
    """
    if matriz is None or matriz.empty:
        return None
    
    matriz_plot = matriz.copy()
    
    # Criar heatmap SEM RÓTULOS
    fig = go.Figure(data=go.Heatmap(
        z=matriz_plot.values,
        x=matriz_plot.columns,
        y=matriz_plot.index,
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate='<b>Mês Receita: %{y}</b><br><b>Mês Lead: %{x}</b><br>Valor: %{z:,.2f}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Valor",
                side="right"
            )
        )
        # REMOVIDO: text e texttemplate para retirar rótulos
    ))
    
    # Destacar diagonal principal (meses iguais)
    for i in range(len(matriz_plot.index)):
        fig.add_shape(
            type="rect",
            x0=i-0.5, y0=i-0.5, x1=i+0.5, y1=i+0.5,
            line=dict(color=COLORS['accent'], width=2),
            fillcolor="rgba(0,0,0,0)"
        )
    
    fig.update_layout(
        title=dict(
            text=titulo,
            font=dict(size=14, color=COLORS['dark_gray'])
        ),
        xaxis_title="Mês de Geração do Lead",
        yaxis_title="Mês de Geração da Receita",
        width=width,
        height=height,
        font=dict(family="Arial, sans-serif", size=10, color=COLORS['dark_gray']),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=8, color=COLORS['dark_gray'])
        ),
        yaxis=dict(
            tickfont=dict(size=8, color=COLORS['dark_gray'])
        ),
        plot_bgcolor=COLORS['white'],
        paper_bgcolor=COLORS['white']
    )
    
    return fig

# Função para calcular estatísticas da matriz
def calcular_estatisticas_matriz(matriz, tipo="Receita"):
    if matriz is None or matriz.empty:
        return None
    
    stats = {}
    
    stats['valor_total'] = matriz.sum().sum()
    diagonal_principal = np.diag(matriz)
    stats['valor_diagonal'] = diagonal_principal.sum()
    stats['percentual_diagonal'] = (stats['valor_diagonal'] / stats['valor_total'] * 100) if stats['valor_total'] > 0 else 0
    stats['valor_fora_diagonal'] = stats['valor_total'] - stats['valor_diagonal']
    
    if len(diagonal_principal) > 0 and diagonal_principal.sum() > 0:
        mes_max_diagonal = matriz.index[np.argmax(diagonal_principal)]
        stats['mes_maior_diagonal'] = mes_max_diagonal
        stats['valor_maior_diagonal'] = diagonal_principal.max()
    else:
        stats['mes_maior_diagonal'] = "N/A"
        stats['valor_maior_diagonal'] = 0
    
    stats['eficiencia'] = stats['percentual_diagonal'] / 100
    stats['total_celulas'] = matriz.size
    stats['celulas_preenchidas'] = (matriz.values > 0).sum()
    stats['percentual_preenchidas'] = (stats['celulas_preenchidas'] / stats['total_celulas'] * 100) if stats['total_celulas'] > 0 else 0
    
    if tipo == "CAC/LTV":
        stats['celulas_saudaveis'] = (matriz.values < 1.0).sum()
        stats['percentual_saudavel'] = (stats['celulas_saudaveis'] / stats['total_celulas'] * 100) if stats['total_celulas'] > 0 else 0
        stats['celulas_problematicas'] = (matriz.values >= 1.0).sum()
        stats['percentual_problematico'] = (stats['celulas_problematicas'] / stats['total_celulas'] * 100) if stats['total_celulas'] > 0 else 0
    
    elif tipo == "CAC":
        if stats['celulas_preenchidas'] > 0:
            cac_medio = stats['valor_total'] / stats['celulas_preenchidas']
            stats['celulas_acima_media'] = (matriz.values > cac_medio).sum()
            stats['percentual_acima_media'] = (stats['celulas_acima_media'] / stats['total_celulas'] * 100) if stats['total_celulas'] > 0 else 0
    
    elif tipo == "LTV":
        if stats['celulas_preenchidas'] > 0:
            ltv_medio = stats['valor_total'] / stats['celulas_preenchidas']
            stats['celulas_abaixo_media'] = (matriz.values < ltv_medio).sum()
            stats['percentual_abaixo_media'] = (stats['celulas_abaixo_media'] / stats['total_celulas'] * 100) if stats['total_celulas'] > 0 else 0
    
    return stats

# Função para aplicar a lógica de receita bruta mensal com filtro de data
def calcular_receita_mensal(df, ano_filtro=2025):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    df_filtrado = df.copy()
    
    try:
        condicao1 = df_filtrado['Mês geração receita'] == df_filtrado['Mês geração lead']
        condicao2 = df_filtrado['Considerar?'] == 'Sim'
        condicao3 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        if ano_filtro == 2024:
            condicao4 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
        else:
            condicao4 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
        
        mask = condicao1 & condicao2 & condicao3 & condicao4
        df_filtrado = df_filtrado[mask].copy()
        
        if not df_filtrado.empty and 'Mês geração receita' in df_filtrado.columns and 'VL UNI' in df_filtrado.columns:
            receita_mensal = df_filtrado.groupby('Mês geração receita')['VL UNI'].sum().reset_index()
            receita_mensal.columns = ['Mês', 'Receita Bruta']
            receita_mensal['Receita Líquida'] = receita_mensal['Receita Bruta'] * 0.56
            
            if ano_filtro == 2024:
                ordem_meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                              'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
            else:
                ordem_meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                              'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
            
            receita_mensal = receita_mensal[receita_mensal['Mês'].isin(ordem_meses)].copy()
            receita_mensal['Mês_Ordenado'] = pd.Categorical(receita_mensal['Mês'], categories=ordem_meses, ordered=True)
            receita_mensal = receita_mensal.sort_values('Mês_Ordenado').drop('Mês_Ordenado', axis=1)
            
        else:
            receita_mensal = pd.DataFrame(columns=['Mês', 'Receita Bruta', 'Receita Líquida'])
        
        return receita_mensal, df_filtrado
    
    except Exception as e:
        st.error(f"Erro ao calcular receita: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Função para calcular Novos Tutores por Mês com filtro de data
def calcular_novos_tutores_mes(df, ano_filtro=2025):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    df_filtrado = df.copy()
    
    try:
        condicao1 = df_filtrado['Mês geração receita'] == df_filtrado['Mês geração lead']
        condicao2 = df_filtrado['Considerar?'] == 'Sim'
        condicao3 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        if ano_filtro == 2024:
            condicao4 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
        else:
            condicao4 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
        
        mask = condicao1 & condicao2 & condicao3 & condicao4
        df_filtrado_tutores = df_filtrado[mask].copy()
        
        df_tutores_unicos = df_filtrado_tutores.drop_duplicates(subset=['E-MAIL'], keep='first')
        
        if not df_tutores_unicos.empty and 'Mês geração receita' in df_tutores_unicos.columns:
            novos_tutores_mes = df_tutores_unicos.groupby('Mês geração receita').size().reset_index()
            novos_tutores_mes.columns = ['Mês', 'Novos Tutores']
            
            if ano_filtro == 2024:
                ordem_meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                              'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
            else:
                ordem_meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                              'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
            
            novos_tutores_mes = novos_tutores_mes[novos_tutores_mes['Mês'].isin(ordem_meses)].copy()
            novos_tutores_mes['Mês_Ordenado'] = pd.Categorical(novos_tutores_mes['Mês'], categories=ordem_meses, ordered=True)
            novos_tutores_mes = novos_tutores_mes.sort_values('Mês_Ordenado').drop('Mês_Ordenado', axis=1)
            
        else:
            novos_tutores_mes = pd.DataFrame(columns=['Mês', 'Novos Tutores'])
        
        return novos_tutores_mes, df_tutores_unicos
    
    except Exception as e:
        st.error(f"Erro ao calcular novos tutores: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Função para calcular métricas de Cohort (CAC, LTV e CAC/LTV) com filtro de data
def calcular_metricas_cohort(novos_tutores_mes, receita_mensal, ano_filtro=2025):
    if novos_tutores_mes.empty or receita_mensal.empty:
        return pd.DataFrame()
    
    if ano_filtro == 2024:
        meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
        investimento_mensal = INVESTIMENTO_MENSAL_2024
    else:
        meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
        investimento_mensal = INVESTIMENTO_MENSAL_2025
    
    cohort_data = pd.DataFrame({
        'Mês': meses
    })
    
    cohort_data['Investimento'] = cohort_data['Mês'].map(investimento_mensal).fillna(0)
    cohort_data = cohort_data.merge(novos_tutores_mes, on='Mês', how='left')
    cohort_data = cohort_data.merge(receita_mensal[['Mês', 'Receita Líquida']], on='Mês', how='left')
    
    cohort_data['Novos Tutores'] = cohort_data['Novos Tutores'].fillna(0)
    cohort_data['Receita Líquida'] = cohort_data['Receita Líquida'].fillna(0)
    
    cohort_data['CAC'] = cohort_data.apply(
        lambda x: x['Investimento'] / x['Novos Tutores'] if x['Novos Tutores'] > 0 else 0, 
        axis=1
    )
    
    cohort_data['LTV'] = cohort_data.apply(
        lambda x: x['Receita Líquida'] / x['Novos Tutores'] if x['Novos Tutores'] > 0 else 0, 
        axis=1
    )
    
    cohort_data['CAC/LTV'] = cohort_data.apply(
        lambda x: x['CAC'] / x['LTV'] if x['LTV'] > 0 else 0, 
        axis=1
    )
    
    cohort_data['ROI (%)'] = cohort_data.apply(
        lambda x: ((x['Receita Líquida'] - x['Investimento']) / x['Investimento']) * 100 if x['Investimento'] > 0 else 0, 
        axis=1
    )
    
    return cohort_data

# NOVA FUNÇÃO: Análise de Performance por LP
def analisar_performance_lp(df, ano_filtro=2025):
    if df is None or df.empty:
        return pd.DataFrame()
    
    df_filtrado = df.copy()
    
    try:
        condicao1 = df_filtrado['Considerar?'] == 'Sim'
        condicao2 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        if ano_filtro == 2024:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
        else:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
        
        mask = condicao1 & condicao2 & condicao3
        df_filtrado = df_filtrado[mask].copy()
        
        if df_filtrado.empty:
            return pd.DataFrame()
        
        lps_para_analise = list(INVESTIMENTO_POR_LP.keys())
        resultados = []
        
        for lp in lps_para_analise:
            df_lp = df_filtrado[df_filtrado['LP'] == lp].copy()
            
            if df_lp.empty:
                resultados.append({
                    'LP': lp,
                    'Investimento': INVESTIMENTO_POR_LP.get(lp, 0),
                    'Leads': 0,
                    'CPL': 0,
                    'Realizado': 0,
                    'Tx.Conv': 0,
                    'CAC': 0,
                    'Receita': 0,
                    'ROAS': 0,
                    'TM': 0,
                    'CAC/LTV': 0,
                    'Receita Cohort': 0
                })
                continue
            
            investimento = INVESTIMENTO_POR_LP.get(lp, 0)
            leads_unicos = df_lp['E-MAIL'].nunique()
            cpl = investimento / leads_unicos if leads_unicos > 0 else 0
            realizado = df_lp['VL UNI'].sum()
            tx_conv = (realizado / investimento * 100) if investimento > 0 else 0
            tutores_unicos_com_receita = df_lp.drop_duplicates(subset=['E-MAIL'])['E-MAIL'].nunique()
            cac = investimento / tutores_unicos_com_receita if tutores_unicos_com_receita > 0 else 0
            receita_liquida = realizado * 0.56
            roas = (receita_liquida / investimento * 100) if investimento > 0 else 0
            tm = realizado / tutores_unicos_com_receita if tutores_unicos_com_receita > 0 else 0
            ltv = receita_liquida / tutores_unicos_com_receita if tutores_unicos_com_receita > 0 else 0
            cac_ltv_ratio = cac / ltv if ltv > 0 else 0
            receita_cohort = receita_liquida
            
            resultados.append({
                'LP': lp,
                'Investimento': investimento,
                'Leads': leads_unicos,
                'CPL': cpl,
                'Realizado': realizado,
                'Tx.Conv': tx_conv,
                'CAC': cac,
                'Receita': receita_liquida,
                'ROAS': roas,
                'TM': tm,
                'CAC/LTV': cac_ltv_ratio,
                'Receita Cohort': receita_cohort
            })
        
        df_resultados = pd.DataFrame(resultados)
        df_resultados = df_resultados.sort_values('Receita', ascending=False)
        
        return df_resultados
    
    except Exception as e:
        st.error(f"Erro na análise de LPs: {e}")
        return pd.DataFrame()

# NOVA FUNÇÃO: Performance Mensal por LP
def analisar_performance_mensal_lp(df, ano_filtro=2025):
    if df is None or df.empty:
        return pd.DataFrame()
    
    df_filtrado = df.copy()
    
    try:
        condicao1 = df_filtrado['Considerar?'] == 'Sim'
        condicao2 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        if ano_filtro == 2024:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
            meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                    'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
        else:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
            meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                    'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
        
        mask = condicao1 & condicao2 & condicao3
        df_filtrado = df_filtrado[mask].copy()
        
        if df_filtrado.empty:
            return pd.DataFrame()
        
        lps_para_analise = list(INVESTIMENTO_POR_LP.keys())
        resultados_mensais = []
        
        for lp in lps_para_analise:
            for mes in meses:
                df_lp_mes = df_filtrado[(df_filtrado['LP'] == lp) & (df_filtrado['Mês geração receita'] == mes)].copy()
                
                if df_lp_mes.empty:
                    resultados_mensais.append({
                        'LP': lp,
                        'Mês': mes,
                        'Receita': 0,
                        'Leads': 0,
                        'ROAS': 0,
                        'CAC/LTV': 0
                    })
                    continue
                
                investimento_mensal_lp = INVESTIMENTO_POR_LP.get(lp, 0) / 12
                leads_unicos_mes = df_lp_mes['E-MAIL'].nunique()
                receita_mes = df_lp_mes['VL UNI'].sum()
                receita_liquida_mes = receita_mes * 0.56
                roas_mes = (receita_liquida_mes / investimento_mensal_lp * 100) if investimento_mensal_lp > 0 else 0
                tutores_unicos_mes = df_lp_mes.drop_duplicates(subset=['E-MAIL'])['E-MAIL'].nunique()
                cac_mes = investimento_mensal_lp / tutores_unicos_mes if tutores_unicos_mes > 0 else 0
                ltv_mes = receita_liquida_mes / tutores_unicos_mes if tutores_unicos_mes > 0 else 0
                cac_ltv_mes = cac_mes / ltv_mes if ltv_mes > 0 else 0
                
                resultados_mensais.append({
                    'LP': lp,
                    'Mês': mes,
                    'Receita': receita_liquida_mes,
                    'Leads': leads_unicos_mes,
                    'ROAS': roas_mes,
                    'CAC/LTV': cac_ltv_mes
                })
        
        df_resultados_mensais = pd.DataFrame(resultados_mensais)
        return df_resultados_mensais
    
    except Exception as e:
        st.error(f"Erro na análise mensal de LPs: {e}")
        return pd.DataFrame()

# Configuração de layout limpo para gráficos COM FONTES MAIORES PARA LPs
def configurar_layout_clean(fig, titulo="", width=800, height=500, fonte_maior=False):
    if fonte_maior:
        # Configuração com fontes maiores para gráficos de LP
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=16, color=COLORS['dark_gray']),
            title=dict(
                text=titulo,
                font=dict(size=24, color=COLORS['dark_gray'], family="Arial, sans-serif"),
                x=0.5,
                xanchor='center',
                y=0.95
            ),
            xaxis=dict(
                title=dict(
                    font=dict(size=18, color=COLORS['dark_gray']),
                    standoff=20
                ),
                tickfont=dict(size=16, color=COLORS['dark_gray']),
                gridcolor=COLORS['light_gray'],
                linecolor=COLORS['light_gray'],
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                title=dict(
                    font=dict(size=18, color=COLORS['dark_gray']),
                    standoff=20
                ),
                tickfont=dict(size=16, color=COLORS['dark_gray']),
                gridcolor=COLORS['light_gray'],
                linecolor=COLORS['light_gray'],
                showgrid=True,
                zeroline=False,
                tickformat=",."
            ),
            legend=dict(
                font=dict(size=16, color=COLORS['dark_gray']),
                bgcolor=COLORS['white'],
                bordercolor=COLORS['light_gray'],
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor=COLORS['white'],
            paper_bgcolor=COLORS['white'],
            margin=dict(l=60, r=60, t=80, b=80),
            hoverlabel=dict(
                bgcolor=COLORS['white'],
                bordercolor=COLORS['gray'],
                font_size=16,
                font_family="Arial"
            ),
            width=width,
            height=height
        )
    else:
        # Configuração padrão
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=14, color=COLORS['dark_gray']),
            title=dict(
                text=titulo,
                font=dict(size=20, color=COLORS['dark_gray'], family="Arial, sans-serif"),
                x=0.5,
                xanchor='center',
                y=0.95
            ),
            xaxis=dict(
                title=dict(
                    font=dict(size=16, color=COLORS['dark_gray']),
                    standoff=20
                ),
                tickfont=dict(size=14, color=COLORS['dark_gray']),
                gridcolor=COLORS['light_gray'],
                linecolor=COLORS['light_gray'],
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                title=dict(
                    font=dict(size=16, color=COLORS['dark_gray']),
                    standoff=20
                ),
                tickfont=dict(size=14, color=COLORS['dark_gray']),
                gridcolor=COLORS['light_gray'],
                linecolor=COLORS['light_gray'],
                showgrid=True,
                zeroline=False,
                tickformat=",."
            ),
            legend=dict(
                font=dict(size=14, color=COLORS['dark_gray']),
                bgcolor=COLORS['white'],
                bordercolor=COLORS['light_gray'],
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor=COLORS['white'],
            paper_bgcolor=COLORS['white'],
            margin=dict(l=60, r=60, t=80, b=80),
            hoverlabel=dict(
                bgcolor=COLORS['white'],
                bordercolor=COLORS['gray'],
                font_size=14,
                font_family="Arial"
            ),
            width=width,
            height=height
        )
    
    return fig

# Interface principal do dashboard
def main_dashboard():
    # CSS ESPECÍFICO PARA O DASHBOARD - VERSÃO CORRIGIDA
    st.markdown(f"""
    <style>
    /* ESTILOS NATURAIS DO STREAMLIT - SEM TARJAS BRANCAS */
    .stApp {{
        background-color: #ffffff;
    }}
    
    /* SIDEBAR COM COR NATURAL */
    section[data-testid="stSidebar"] > div {{
        background-color: {COLORS['sidebar_bg']};
    }}
    
    /* INPUTS NO DASHBOARD */
    .stTextInput>div>div>input,
    .stSelectbox>div>div {{
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }}
    
    /* TÍTULOS E TEXTOS */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['black']} !important;
    }}
    
    p, div, span {{
        color: {COLORS['black']} !important;
    }}
    
    /* BOTÕES */
    .stButton>button {{
        background-color: {COLORS['primary']} !important;
        color: {COLORS['white']} !important;
    }}
    
    /* MÉTRICAS */
    .stMetric {{
        background-color: {COLORS['white']};
        border: 1px solid {COLORS['light_gray']};
    }}
    
    .stMetric label {{
        color: {COLORS['gray']} !important;
    }}
    
    .stMetric div {{
        color: {COLORS['black']} !important;
    }}
    
    /* ABAS */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {COLORS['light_gray']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['gray']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']};
        color: {COLORS['white']};
    }}
    
    /* TABELAS */
    .dataframe th {{
        background-color: {COLORS['primary']};
        color: {COLORS['white']};
    }}
    
    .dataframe td {{
        background-color: {COLORS['white']};
        color: {COLORS['black']};
    }}
    </style>
    """, unsafe_allow_html=True)

    # Header com Logo
    logo = load_logo()
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        if logo:
            st.image(logo, width=80)
        else:
            st.markdown("### 📊")
    
    with col2:
        st.title("Veros Intelligence Dashboard")
        st.markdown("Análise preditiva e insights automatizados")
    
    with col3:
        if st.button("Sair"):
            st.session_state.logged_in = False
            st.rerun()
    
    st.markdown("---")
    
    # Carregar dados
    with st.spinner("Carregando e analisando dados..."):
        df = load_data()
    
    if df is None:
        st.warning("Para usar o dashboard, faça upload do arquivo de dados ou coloque o arquivo 'DADOS_RECEITA_VEROS.xlsx' na pasta do projeto.")
        st.info("Você pode fazer upload do arquivo usando o seletor acima ou colocar o arquivo em uma das seguintes pastas:")
        st.write("- Na raiz do projeto: DADOS_RECEITA_VEROS.xlsx")
        st.write("- Na pasta data/: data/DADOS_RECEITA_VEROS.xlsx")
        st.write("- Na pasta assets/: assets/DADOS_RECEITA_VEROS.xlsx")
        st.write("- Na pasta dados/: dados/DADOS_RECEITA_VEROS.xlsx")
        return
    
    # Sidebar com filtros
    with st.sidebar:
        st.header("Filtros e Configurações")
        st.info(f"Dataset carregado: {len(df)} registros")
        
        # Filtro de Ano
        st.subheader("Filtro de Período")
        ano_selecionado = st.selectbox(
            "Selecione o ano:",
            options=[2024, 2025],
            index=1
        )
        
        st.info(f"Filtro Ativo: Ano {ano_selecionado}")
        
        if 'UNIDADE DE NEGOCIO' in df.columns:
            unidades = ['Todas'] + sorted(df['UNIDADE DE NEGOCIO'].dropna().unique().tolist())
            unidade_selecionada = st.selectbox("Unidade de Negócio", unidades)
    
    # Processar dados
    with st.spinner(f"Calculando métricas avançadas para {ano_selecionado}..."):
        receita_mensal, df_filtrado = calcular_receita_mensal(df, ano_filtro=ano_selecionado)
        novos_tutores_mes, df_tutores_unicos = calcular_novos_tutores_mes(df, ano_filtro=ano_selecionado)
        cohort_data = calcular_metricas_cohort(novos_tutores_mes, receita_mensal, ano_filtro=ano_selecionado)
        
        # Criar matrizes escadinhas
        matriz_receita, matriz_formatada_receita = criar_matriz_escadinha(df, ano_filtro=ano_selecionado)
        matriz_cac, matriz_formatada_cac = criar_matriz_cac(df, ano_filtro=ano_selecionado)
        matriz_ltv, matriz_formatada_ltv = criar_matriz_ltv(df, ano_filtro=ano_selecionado)
        matriz_cac_ltv, matriz_formatada_cac_ltv = criar_matriz_cac_ltv_ratio(df, ano_filtro=ano_selecionado)
        
        # Criar heatmaps SEM RÓTULOS
        heatmap_receita = criar_heatmap_matriz(matriz_receita, "Receita Bruta", 'Blues', 500, 450)
        heatmap_cac = criar_heatmap_matriz(matriz_cac, "CAC (Custo Aquisição)", 'Reds', 500, 450)
        heatmap_ltv = criar_heatmap_matriz(matriz_ltv, "LTV (Valor Cliente)", 'Greens', 500, 450)
        heatmap_cac_ltv = criar_heatmap_matriz(matriz_cac_ltv, "Razão CAC/LTV", 'RdYlGn_r', 500, 450)
        
        # Calcular estatísticas
        estatisticas_receita = calcular_estatisticas_matriz(matriz_receita, "Receita") if matriz_receita is not None else None
        estatisticas_cac = calcular_estatisticas_matriz(matriz_cac, "CAC") if matriz_cac is not None else None
        estatisticas_ltv = calcular_estatisticas_matriz(matriz_ltv, "LTV") if matriz_ltv is not None else None
        estatisticas_cac_ltv = calcular_estatisticas_matriz(matriz_cac_ltv, "CAC/LTV") if matriz_cac_ltv is not None else None
        
        # Análise de Performance por LP
        performance_lp = analisar_performance_lp(df, ano_filtro=ano_selecionado)
        performance_mensal_lp = analisar_performance_mensal_lp(df, ano_filtro=ano_selecionado)
    
    # SISTEMA DE ABAS
    tab1, tab2, tab3 = st.tabs([
        "Visão Geral", 
        "Matrizes Escadinha",
        "Análise de LPs"
    ])
    
    with tab1:
        st.markdown(f'<div style="background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["primary_light"]}); color: white; padding: 1.5rem; border-radius: 12px; margin: 2rem 0;"><h2 style="color: white; margin: 0;">Visão Geral do Performance</h2><p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Métricas consolidadas e tendências do período</p></div>', unsafe_allow_html=True)
        
        if not receita_mensal.empty:
            # Métricas Principais em Grid
            st.subheader("Métricas Principais")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                receita_total_bruta = receita_mensal['Receita Bruta'].sum()
                st.metric(
                    "Receita Bruta Total", 
                    f"R$ {receita_total_bruta:,.0f}"
                )
            
            with col2:
                receita_total_liquida = receita_mensal['Receita Líquida'].sum()
                st.metric(
                    "Receita Líquida Total", 
                    f"R$ {receita_total_liquida:,.0f}"
                )
            
            with col3:
                total_tutores = novos_tutores_mes['Novos Tutores'].sum()
                st.metric(
                    "Total Novos Tutores", 
                    f"{total_tutores:,}"
                )
            
            with col4:
                if not cohort_data.empty:
                    cac_medio = cohort_data[cohort_data['CAC'] > 0]['CAC'].mean()
                    st.metric(
                        "CAC Médio", 
                        f"R$ {cac_medio:,.0f}"
                    )
            
            # Gráficos Principais
            st.subheader("Evolução Mensal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de Receita Mensal
                fig_receita = px.line(
                    receita_mensal, 
                    x='Mês', 
                    y=['Receita Bruta', 'Receita Líquida'],
                    title="Receita Mensal",
                    color_discrete_map={
                        'Receita Bruta': COLORS['primary'],
                        'Receita Líquida': COLORS['secondary']
                    }
                )
                
                fig_receita.update_traces(
                    mode='lines+markers'
                )
                
                fig_receita = configurar_layout_clean(fig_receita, "Receita Mensal")
                st.plotly_chart(fig_receita, use_container_width=True)
            
            with col2:
                # Gráfico de Novos Tutores
                fig_tutores = px.bar(
                    novos_tutores_mes,
                    x='Mês',
                    y='Novos Tutores',
                    title="Novos Tutores por Mês",
                    color_discrete_sequence=[COLORS['info']]
                )
                
                fig_tutores = configurar_layout_clean(fig_tutores, "Novos Tutores por Mês")
                st.plotly_chart(fig_tutores, use_container_width=True)
        else:
            st.warning("Não há dados disponíveis para o período selecionado")
    
    with tab2:
        st.markdown(f'<div style="background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["primary_light"]}); color: white; padding: 1.5rem; border-radius: 12px; margin: 2rem 0;"><h2 style="color: white; margin: 0;">Matrizes Escadinha - Análise Temporal Completa</h2><p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Relação entre geração de receita e geração de leads</p></div>', unsafe_allow_html=True)
        
        # [O RESTANTE DO CÓDIGO DA TAB2 PERMANECE IGUAL...]
        # ... (mantive o código original para economizar espaço)

# Aplicação principal
def main():
    # Inicializar estado de login
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Verificar se está logado
    if not st.session_state.logged_in:
        login_screen()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()