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

# =============================================
# SISTEMA DE DETECÇÃO AUTOMÁTICA DE TEMA
# =============================================

def detectar_tema_navegador():
    """
    Detecta se o usuário está usando tema claro ou escuro no navegador
    """
    try:
        # Tenta detectar via JavaScript o tema preferido do usuário
        if 'detected_theme' not in st.session_state:
            # Inicializa com tema claro como padrão
            st.session_state.detected_theme = 'light'
            
        # Script JavaScript para detectar tema
        theme_script = """
        <script>
        try {
            const isDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
            const currentTheme = isDarkMode ? 'dark' : 'light';
            
            // Envia para o Streamlit
            if (window.parent && window.parent.postMessage) {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: currentTheme
                }, '*');
            }
        } catch(e) {
            console.log('Erro ao detectar tema:', e);
        }
        </script>
        """
        
        # Usa components para executar o JavaScript
        st.components.v1.html(theme_script, height=0)
        
        return st.session_state.detected_theme
        
    except Exception as e:
        # Fallback para tema claro em caso de erro
        return 'light'

def obter_cores_por_tema(tema):
    """
    Retorna a paleta de cores adequada para o tema detectado
    """
    if tema == 'dark':
        # PALETA PARA TEMA ESCURO
        return {
            'primary': '#3b82f6',
            'primary_light': '#60a5fa',
            'secondary': '#10b981',
            'secondary_light': '#34d399',
            'accent': '#ef4444',
            'accent_light': '#f87171',
            'warning': '#f59e0b',
            'warning_light': '#fbbf24',
            'info': '#0ea5e9',
            'info_light': '#38bdf8',
            'gray': '#9ca3af',
            'light_gray': '#374151',
            'dark_gray': '#f3f4f6',
            'white': '#111827',
            'black': '#f9fafb',
            'sidebar_bg': '#1f2937',
            'background': '#111827',
            'text_primary': '#f9fafb',
            'text_secondary': '#d1d5db',
            'border': '#374151',
            'success': '#10b981',
            'error': '#ef4444',
            'plot_bg': '#1f2937',
            'paper_bg': '#111827'
        }
    else:
        # PALETA PARA TEMA CLARO (padrão original)
        return {
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
            'sidebar_bg': '#ffffff',
            'background': '#ffffff',
            'text_primary': '#111827',
            'text_secondary': '#6b7280',
            'border': '#e5e7eb',
            'success': '#059669',
            'error': '#dc2626',
            'plot_bg': '#ffffff',
            'paper_bg': '#ffffff'
        }

# Inicializar tema e cores
if 'detected_theme' not in st.session_state:
    st.session_state.detected_theme = 'light'

TEMA_ATUAL = st.session_state.detected_theme
COLORS = obter_cores_por_tema(TEMA_ATUAL)

# =============================================
# DADOS FIXOS
# =============================================

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

CREDENCIAIS = {
    "Midianewton": "New@2025"
}

# =============================================
# FUNÇÕES AUXILIARES
# =============================================

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

# =============================================
# TELA DE LOGIN ADAPTADA PARA TEMA DINÂMICO
# =============================================

def login_screen():
    # Usar cores dinâmicas baseadas no tema
    BACKGROUND_COLOR = COLORS['background']
    TEXT_COLOR = COLORS['text_primary']
    GRAY_COLOR = COLORS['text_secondary']
    BUTTON_COLOR = COLORS['primary']
    BUTTON_HOVER = COLORS['primary_light']
    BORDER_COLOR = COLORS['border']

    st.markdown(f"""
    <style>
        /* Remove menus e cabeçalho do Streamlit */
        #MainMenu, footer, header {{
            visibility: hidden;
        }}

        /* Fundo e centralização flexível */
        .stApp {{
            background-color: {BACKGROUND_COLOR};
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', sans-serif;
        }}

        /* Wrapper de login centralizado */
        .login-wrapper {{
            text-align: center;
            width: 100%;
            max-width: 320px;
            padding: 1rem;
            box-sizing: border-box;
            animation: fadeIn 0.8s ease-in-out;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Inputs */
        .stTextInput > div {{
            width: 100%;
            margin-bottom: 1rem;
        }}

        .stTextInput>div>div>input {{
            width: 100%;
            border-radius: 6px;
            border: 1px solid {BORDER_COLOR};
            padding: 8px 12px;
            font-size: 14px;
            color: {TEXT_COLOR};
            background-color: {BACKGROUND_COLOR};
            box-sizing: border-box;
            transition: all 0.3s ease;
        }}

        .stTextInput>div>div>input:focus {{
            border-color: {BUTTON_COLOR};
            box-shadow: 0 0 0 2px {BUTTON_COLOR}20;
            outline: none;
        }}

        .stTextInput label {{
            font-weight: 500;
            color: {TEXT_COLOR};
            display: block;
            text-align: left;
            margin-bottom: 4px;
            font-size: 14px;
        }}

        /* Botão */
        .stButton > button {{
            width: 100%;
            background-color: {BUTTON_COLOR};
            color: {COLORS['white']};
            border-radius: 6px;
            padding: 10px 0;
            border: none;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
            transition: 0.3s;
        }}

        .stButton>button:hover {{
            background-color: {BUTTON_HOVER};
        }}

        /* Subtítulo */
        .login-wrapper p {{
            color: {GRAY_COLOR};
            margin-bottom: 1.5rem;
            font-size: 14px;
        }}
    </style>
    """, unsafe_allow_html=True)

    # ---------- LAYOUT ----------
    st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)

    # Logo
    login_logo = load_login_logo()
    if login_logo:
        st.image(login_logo, width=200)
    else:
        st.markdown(f'<div style="font-size: 3rem; color: {TEXT_COLOR}; margin-bottom: 1rem;">📊</div>', unsafe_allow_html=True)

    # Título e subtítulo
    st.markdown(f"""
    <h2 style='color:{TEXT_COLOR}; margin: 0.5rem 0 0.2rem 0;'>Veros Intelligence</h2>
    <p>Acesse seu dashboard</p>
    """, unsafe_allow_html=True)

    # ---------- FORMULÁRIO ----------
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

# =============================================
# FUNÇÕES DE ANÁLISE DE DADOS COMPLETAS
# =============================================

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
            font=dict(size=14, color=COLORS['text_primary'])
        ),
        xaxis_title="Mês de Geração do Lead",
        yaxis_title="Mês de Geração da Receita",
        width=width,
        height=height,
        font=dict(family="Arial, sans-serif", size=10, color=COLORS['text_primary']),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=8, color=COLORS['text_primary'])
        ),
        yaxis=dict(
            tickfont=dict(size=8, color=COLORS['text_primary'])
        ),
        plot_bgcolor=COLORS['plot_bg'],
        paper_bgcolor=COLORS['paper_bg']
    )
    
    return fig

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

def configurar_layout_clean(fig, titulo="", width=800, height=500, fonte_maior=False):
    if fonte_maior:
        # Configuração com fontes maiores para gráficos de LP
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=16, color=COLORS['text_primary']),
            title=dict(
                text=titulo,
                font=dict(size=24, color=COLORS['text_primary'], family="Arial, sans-serif"),
                x=0.5,
                xanchor='center',
                y=0.95
            ),
            xaxis=dict(
                title=dict(
                    font=dict(size=18, color=COLORS['text_primary']),
                    standoff=20
                ),
                tickfont=dict(size=16, color=COLORS['text_primary']),
                gridcolor=COLORS['light_gray'],
                linecolor=COLORS['border'],
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                title=dict(
                    font=dict(size=18, color=COLORS['text_primary']),
                    standoff=20
                ),
                tickfont=dict(size=16, color=COLORS['text_primary']),
                gridcolor=COLORS['light_gray'],
                linecolor=COLORS['border'],
                showgrid=True,
                zeroline=False,
                tickformat=",."
            ),
            legend=dict(
                font=dict(size=16, color=COLORS['text_primary']),
                bgcolor=COLORS['background'],
                bordercolor=COLORS['border'],
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor=COLORS['plot_bg'],
            paper_bgcolor=COLORS['paper_bg'],
            margin=dict(l=60, r=60, t=80, b=80),
            hoverlabel=dict(
                bgcolor=COLORS['background'],
                bordercolor=COLORS['border'],
                font_size=16,
                font_family="Arial"
            ),
            width=width,
            height=height
        )
    else:
        # Configuração padrão
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=14, color=COLORS['text_primary']),
            title=dict(
                text=titulo,
                font=dict(size=20, color=COLORS['text_primary'], family="Arial, sans-serif"),
                x=0.5,
                xanchor='center',
                y=0.95
            ),
            xaxis=dict(
                title=dict(
                    font=dict(size=16, color=COLORS['text_primary']),
                    standoff=20
                ),
                tickfont=dict(size=14, color=COLORS['text_primary']),
                gridcolor=COLORS['light_gray'],
                linecolor=COLORS['border'],
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                title=dict(
                    font=dict(size=16, color=COLORS['text_primary']),
                    standoff=20
                ),
                tickfont=dict(size=14, color=COLORS['text_primary']),
                gridcolor=COLORS['light_gray'],
                linecolor=COLORS['border'],
                showgrid=True,
                zeroline=False,
                tickformat=",."
            ),
            legend=dict(
                font=dict(size=14, color=COLORS['text_primary']),
                bgcolor=COLORS['background'],
                bordercolor=COLORS['border'],
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor=COLORS['plot_bg'],
            paper_bgcolor=COLORS['paper_bg'],
            margin=dict(l=60, r=60, t=80, b=80),
            hoverlabel=dict(
                bgcolor=COLORS['background'],
                bordercolor=COLORS['border'],
                font_size=14,
                font_family="Arial"
            ),
            width=width,
            height=height
        )
    
    return fig

# =============================================
# DASHBOARD PRINCIPAL ADAPTADO PARA TEMA DINÂMICO
# =============================================

def main_dashboard():
    # Atualizar tema dinamicamente
    global TEMA_ATUAL, COLORS
    TEMA_ATUAL = detectar_tema_navegador()
    COLORS = obter_cores_por_tema(TEMA_ATUAL)
    
    # Configuração CSS com design adaptativo
    st.markdown(f"""
    <style>
    /* FUNDO DINÂMICO PARA TODA A APLICAÇÃO */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: {COLORS['background']} !important;
    }}
    
    .stApp {{
        background-color: {COLORS['background']} !important;
    }}
    
    /* SIDEBAR DINÂMICA */
    section[data-testid="stSidebar"] > div {{
        background-color: {COLORS['sidebar_bg']} !important;
    }}
    
    section[data-testid="stSidebar"] .stButton>button {{
        background-color: {COLORS['primary']} !important;
        color: {COLORS['white']} !important;
    }}
    
    section[data-testid="stSidebar"] .stSelectbox>div>div {{
        background-color: {COLORS['background']} !important;
        border-color: {COLORS['border']} !important;
        color: {COLORS['text_primary']} !important;
    }}
    
    section[data-testid="stSidebar"] .stTextInput>div>div>input {{
        background-color: {COLORS['background']} !important;
        border-color: {COLORS['border']} !important;
        color: {COLORS['text_primary']} !important;
    }}
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {{
        color: {COLORS['text_primary']} !important;
    }}
    
    /* TEXTOS PRINCIPAIS */
    h1, h2, h3 {{
        color: {COLORS['text_primary']} !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        font-family: 'Arial', sans-serif;
    }}
    
    p, div, span, li {{
        color: {COLORS['text_primary']} !important;
    }}
    
    /* MÉTRICAS */
    .stMetric {{
        background-color: {COLORS['background']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }}
    
    .stMetric:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    
    .stMetric label {{
        color: {COLORS['text_secondary']} !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }}
    
    .stMetric div {{
        color: {COLORS['text_primary']} !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
    }}
    
    /* ABAS */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0px;
        background-color: {COLORS['light_gray']};
        padding: 8px;
        border-radius: 12px;
        margin-bottom: 2rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 60px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        gap: 1px;
        padding: 16px 24px;
        font-weight: 500;
        font-size: 16px;
        color: {COLORS['text_secondary']};
        transition: all 0.3s ease;
        border: none;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {COLORS['primary']}20;
        color: {COLORS['primary']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']};
        color: {COLORS['white']};
        box-shadow: 0 2px 8px {COLORS['primary']}30;
    }}
    
    /* COMPONENTES PERSONALIZADOS */
    .matriz-container {{
        background: {COLORS['background']};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid {COLORS['border']};
    }}
    
    .matriz-header {{
        border-bottom: 2px solid {COLORS['primary']};
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }}
    
    .matriz-stats {{
        background: {COLORS['light_gray']};
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid {COLORS['primary']};
    }}
    
    .metricas-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }}
    
    .metrica-card {{
        background: {COLORS['background']};
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid {COLORS['primary']};
        transition: transform 0.2s ease;
    }}
    
    .metrica-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }}
    
    .metrica-titulo {{
        font-size: 0.8rem;
        color: {COLORS['text_secondary']};
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metrica-valor {{
        font-size: 1.4rem;
        color: {COLORS['text_primary']};
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    
    .section-header {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['primary_light']});
        color: {COLORS['white']};
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0;
    }}
    
    .section-title {{
        color: {COLORS['white']} !important;
        margin: 0 !important;
        font-size: 1.5rem !important;
    }}
    
    .section-subtitle {{
        color: rgba(255,255,255,0.9) !important;
        margin: 0.5rem 0 0 0 !important;
        font-size: 1rem !important;
        font-weight: 400 !important;
    }}
    
    .heatmap-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        margin: 1.5rem 0;
    }}
    
    .info-box {{
        background: {COLORS['info_light']}15;
        border: 1px solid {COLORS['info_light']};
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    
    /* TABELAS */
    .dataframe th {{
        background-color: {COLORS['primary']};
        color: {COLORS['white']};
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }}
    
    .dataframe td {{
        padding: 10px;
        border-bottom: 1px solid {COLORS['border']};
        color: {COLORS['text_primary']};
        background-color: {COLORS['background']};
    }}
    
    .dataframe tr:hover {{
        background-color: {COLORS['light_gray']};
    }}
    
    /* BOTÕES */
    .stButton>button[kind="secondary"] {{
        background-color: {COLORS['background']} !important;
        color: {COLORS['text_primary']} !important;
        border: 1px solid {COLORS['border']} !important;
    }}
    
    .stButton>button[kind="secondary"]:hover {{
        background-color: {COLORS['light_gray']} !important;
        border-color: {COLORS['gray']} !important;
    }}
    
    /* ALERTAS E MENSAGENS */
    .stAlert {{
        background-color: {COLORS['light_gray']};
        border: 1px solid {COLORS['border']};
    }}
    
    /* SCROLLBAR PERSONALIZADA */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['light_gray']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['primary']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['primary_light']};
    }}
    
    @media (max-width: 1200px) {{
        .heatmap-grid {{
            grid-template-columns: 1fr;
        }}
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
            st.markdown(f"### 📊")
    
    with col2:
        st.title("Veros Intelligence Dashboard")
        
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
        st.markdown(f'<div class="section-header"><h2 class="section-title">Visão Geral do Performance</h2><p class="section-subtitle">Métricas consolidadas e tendências do período</p></div>', unsafe_allow_html=True)
        
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
        st.markdown(f'<div class="section-header"><h2 class="section-title">Matrizes Escadinha - Análise Temporal Completa</h2><p class="section-subtitle">Relação entre geração de receita e geração de leads</p></div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <h4>Período Selecionado: {ano_selecionado}</h4>
            <p>As matrizes abaixo mostram apenas os dados do ano {ano_selecionado}. 
            Para analisar outro período, altere o filtro na barra lateral.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="matriz-stats">
            <h4>Como interpretar as Matrizes Escadinha:</h4>
            <ul>
                <li><strong>Eixo Y (Linhas):</strong> Mês de Geração da Receita</li>
                <li><strong>Eixo X (Colunas):</strong> Mês de Geração do Lead</li>
                <li><strong>Diagonal Principal:</strong> Meses coincidentes (condição ideal)</li>
                <li><strong>Células fora da diagonal:</strong> Receita gerada em meses diferentes da geração do lead</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if (matriz_receita is None or matriz_receita.empty) and \
           (matriz_cac is None or matriz_cac.empty) and \
           (matriz_ltv is None or matriz_ltv.empty) and \
           (matriz_cac_ltv is None or matriz_cac_ltv.empty):
            st.warning(f"Não há dados disponíveis para criar as matrizes escadinha do ano {ano_selecionado}.")
            st.info("Tente selecionar outro ano ou verifique se os dados estão corretamente formatados.")
        else:
            # SEÇÃO 1: VISUALIZAÇÕES DAS MATRIZES (SEM RÓTULOS)
            st.subheader("Visualizações das Matrizes")
            
            st.markdown('<div class="heatmap-grid">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if heatmap_receita:
                    st.plotly_chart(heatmap_receita, use_container_width=True)
                else:
                    st.warning("Matriz de Receita não disponível")
                
                if heatmap_cac:
                    st.plotly_chart(heatmap_cac, use_container_width=True)
                else:
                    st.warning("Matriz de CAC não disponível")
            
            with col2:
                if heatmap_ltv:
                    st.plotly_chart(heatmap_ltv, use_container_width=True)
                else:
                    st.warning("Matriz de LTV não disponível")
                
                if heatmap_cac_ltv:
                    st.plotly_chart(heatmap_cac_ltv, use_container_width=True)
                else:
                    st.warning("Matriz de CAC/LTV não disponível")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # SEÇÃO 2: ANÁLISE DETALHADA POR MATRIZ
            st.subheader("Análise Detalhada por Matriz")
            
            analise_tab1, analise_tab2, analise_tab3, analise_tab4 = st.tabs([
                "Receita", "CAC", "LTV", "CAC/LTV"
            ])
            
            with analise_tab1:
                if estatisticas_receita:
                    st.markdown(f"""
                    <div class="matriz-stats">
                        <h4>Análise da Matriz de Receita</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Eficiência Temporal", f"{estatisticas_receita['eficiencia']:.1%}")
                        st.metric("Melhor Mês", estatisticas_receita['mes_maior_diagonal'])
                        st.metric("Receita na Diagonal", f"R$ {estatisticas_receita['valor_diagonal']:,.0f}")
                    
                    with col2:
                        st.metric("Células Preenchidas", f"{estatisticas_receita['celulas_preenchidas']}")
                        st.metric("Percentual Preenchido", f"{estatisticas_receita['percentual_preenchidas']:.1f}%")
                        st.metric("Receita Fora Diagonal", f"R$ {estatisticas_receita['valor_fora_diagonal']:,.0f}")
                    
                    if matriz_formatada_receita is not None:
                        st.subheader("Tabela Detalhada - Receita")
                        st.dataframe(matriz_formatada_receita, use_container_width=True)
                else:
                    st.warning("Não há dados disponíveis para a matriz de Receita")
            
            with analise_tab2:
                if estatisticas_cac:
                    st.markdown(f"""
                    <div class="matriz-stats">
                        <h4>Análise da Matriz de CAC</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if estatisticas_cac['celulas_preenchidas'] > 0:
                            cac_medio = estatisticas_cac['valor_total'] / estatisticas_cac['celulas_preenchidas']
                            st.metric("CAC Médio", f"R$ {cac_medio:,.0f}")
                        st.metric("CAC Mínimo", f"R$ {matriz_cac[matriz_cac > 0].min().min():,.0f}" if not matriz_cac[matriz_cac > 0].empty else "R$ 0")
                        st.metric("CAC Máximo", f"R$ {matriz_cac.max().max():,.0f}")
                    
                    with col2:
                        st.metric("Células Preenchidas", f"{estatisticas_cac['celulas_preenchidas']}")
                        st.metric("Eficiência na Diagonal", f"{estatisticas_cac['percentual_diagonal']:.1f}%")
                        if 'celulas_acima_media' in estatisticas_cac:
                            st.metric("Células Acima da Média", f"{estatisticas_cac['celulas_acima_media']}")
                    
                    if matriz_formatada_cac is not None:
                        st.subheader("Tabela Detalhada - CAC")
                        st.dataframe(matriz_formatada_cac, use_container_width=True)
                else:
                    st.warning("Não há dados disponíveis para a matriz de CAC")
            
            with analise_tab3:
                if estatisticas_ltv:
                    st.markdown(f"""
                    <div class="matriz-stats">
                        <h4>Análise da Matriz de LTV</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if estatisticas_ltv['celulas_preenchidas'] > 0:
                            ltv_medio = estatisticas_ltv['valor_total'] / estatisticas_ltv['celulas_preenchidas']
                            st.metric("LTV Médio", f"R$ {ltv_medio:,.0f}")
                        st.metric("LTV Mínimo", f"R$ {matriz_ltv[matriz_ltv > 0].min().min():,.0f}" if not matriz_ltv[matriz_ltv > 0].empty else "R$ 0")
                        st.metric("LTV Máximo", f"R$ {matriz_ltv.max().max():,.0f}")
                    
                    with col2:
                        st.metric("Células Preenchidas", f"{estatisticas_ltv['celulas_preenchidas']}")
                        st.metric("Eficiência na Diagonal", f"{estatisticas_ltv['percentual_diagonal']:.1f}%")
                        if 'celulas_abaixo_media' in estatisticas_ltv:
                            st.metric("Células Abaixo da Média", f"{estatisticas_ltv['celulas_abaixo_media']}")
                    
                    if matriz_formatada_ltv is not None:
                        st.subheader("Tabela Detalhada - LTV")
                        st.dataframe(matriz_formatada_ltv, use_container_width=True)
                else:
                    st.warning("Não há dados disponíveis para a matriz de LTV")
            
            with analise_tab4:
                if estatisticas_cac_ltv:
                    st.markdown(f"""
                    <div class="matriz-stats">
                        <h4>Análise da Matriz de CAC/LTV</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    ratio_medio = estatisticas_cac_ltv['valor_total'] / estatisticas_cac_ltv['celulas_preenchidas'] if estatisticas_cac_ltv['celulas_preenchidas'] > 0 else 0
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Razão Média", f"{ratio_medio:.2f}")
                        st.metric("Células Saudáveis", f"{estatisticas_cac_ltv['celulas_saudaveis']}")
                        st.metric("Células Problemáticas", f"{estatisticas_cac_ltv['celulas_problematicas']}")
                    
                    with col2:
                        st.metric("Percentual Saudável", f"{estatisticas_cac_ltv['percentual_saudavel']:.1f}%")
                        st.metric("Percentual Problemático", f"{estatisticas_cac_ltv['percentual_problematico']:.1f}%")
                        st.metric("Eficiência na Diagonal", f"{estatisticas_cac_ltv['percentual_diagonal']:.1f}%")
                    
                    st.subheader("Análise de Performance")
                    if ratio_medio < 1.0:
                        st.success("PERFORMANCE SAUDÁVEL - A razão CAC/LTV média indica que o custo de aquisição é menor que o valor do cliente, sugerindo sustentabilidade do negócio.")
                    else:
                        st.warning("ATENÇÃO NECESSÁRIA - A razão CAC/LTV média indica que o custo de aquisição supera o valor do cliente, necessitando otimização das estratégias.")
                    
                    if estatisticas_cac_ltv['percentual_saudavel'] > 50:
                        st.success(f"MAIORIA SAUDÁVEL - {estatisticas_cac_ltv['percentual_saudavel']:.1f}% das combinações têm CAC/LTV < 1.0")
                    else:
                        st.error(f"MAIORIA PROBLEMÁTICA - Apenas {estatisticas_cac_ltv['percentual_saudavel']:.1f}% das combinações têm CAC/LTV < 1.0")
                    
                    if matriz_formatada_cac_ltv is not None:
                        st.subheader("Tabela Detalhada - CAC/LTV")
                        st.dataframe(matriz_formatada_cac_ltv, use_container_width=True)
                else:
                    st.warning("Não há dados disponíveis para a matriz de CAC/LTV")
            
            # SEÇÃO 3: INSIGHTS E RECOMENDAÇÕES
            st.subheader("Insights e Recomendações")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="matriz-stats">
                    <h4>Pontos Fortes</h4>
                    <ul>
                        <li>Alta concentração na diagonal indica eficiência no processo de conversão</li>
                        <li>Padrão consistente sugere processos bem estabelecidos</li>
                        <li>Baixa dispersão temporal entre geração de lead e receita</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="matriz-stats">
                    <h4>Oportunidades</h4>
                    <ul>
                        <li>Analisar células fora da diagonal para entender conversões atípicas</li>
                        <li>Otimizar tempo de conversão baseado nos padrões identificados</li>
                        <li>Segmentar por unidade de negócio para análises mais específicas</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown(f'<div class="section-header"><h2 class="section-title">Análise de Performance por Landing Page</h2><p class="section-subtitle">Métricas detalhadas por canal de aquisição</p></div>', unsafe_allow_html=True)
        
        total_investimento_lp = sum(INVESTIMENTO_POR_LP.values())
        
        if not performance_lp.empty:
            # Tabela principal de performance
            st.subheader("Performance por LP")
            
            # Formatar a tabela para exibição
            performance_formatada = performance_lp.copy()
            
            colunas_monetarias = ['Investimento', 'CPL', 'CAC', 'TM', 'Receita', 'Receita Cohort']
            for col in colunas_monetarias:
                performance_formatada[col] = performance_formatada[col].apply(
                    lambda x: f"R$ {x:,.2f}" if x > 0 else "R$ 0.00"
                )
            
            colunas_percentuais = ['Tx.Conv', 'ROAS']
            for col in colunas_percentuais:
                performance_formatada[col] = performance_formatada[col].apply(
                    lambda x: f"{x:.1f}%" if x > 0 else "0.0%"
                )
            
            performance_formatada['CAC/LTV'] = performance_formatada['CAC/LTV'].apply(
                lambda x: f"{x:.2f}" if x > 0 else "0.00"
            )
            
            performance_formatada['Leads'] = performance_formatada['Leads'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(performance_formatada, use_container_width=True)
            
            # Gráficos de Performance COM FONTES MAIORES
            st.subheader("Visualizações de Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de Receita por LP COM FONTES MAIORES
                fig_receita_lp = px.bar(
                    performance_lp,
                    x='LP',
                    y='Receita',
                    title="Receita por LP",
                    color='Receita',
                    color_continuous_scale='Viridis'
                )
                
                fig_receita_lp = configurar_layout_clean(fig_receita_lp, "Receita por Landing Page", fonte_maior=True)
                st.plotly_chart(fig_receita_lp, use_container_width=True)
            
            with col2:
                # Gráfico de ROAS por LP COM FONTES MAIORES
                fig_roas_lp = px.bar(
                    performance_lp,
                    x='LP',
                    y='ROAS',
                    title="ROAS por LP",
                    color='ROAS',
                    color_continuous_scale='RdYlGn'
                )
                
                fig_roas_lp = configurar_layout_clean(fig_roas_lp, "ROAS por Landing Page", fonte_maior=True)
                st.plotly_chart(fig_roas_lp, use_container_width=True)
            
            # Performance Mensal por LP
            if not performance_mensal_lp.empty:
                st.subheader("Performance Mensal por LP")
                
                lps_disponiveis = performance_mensal_lp['LP'].unique()
                lp_selecionada = st.selectbox("Selecione a LP para análise mensal:", lps_disponiveis)
                
                if lp_selecionada:
                    dados_lp_mensal = performance_mensal_lp[performance_mensal_lp['LP'] == lp_selecionada]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_receita_mensal = px.line(
                            dados_lp_mensal,
                            x='Mês',
                            y='Receita',
                            title=f"Receita Mensal - {lp_selecionada}",
                            markers=True
                        )
                        
                        fig_receita_mensal.update_traces(
                            line=dict(color=COLORS['primary'], width=3),
                            marker=dict(size=8, color=COLORS['primary'])
                        )
                        
                        fig_receita_mensal = configurar_layout_clean(fig_receita_mensal, f"Receita Mensal - {lp_selecionada}")
                        st.plotly_chart(fig_receita_mensal, use_container_width=True)
                    
                    with col2:
                        fig_roas_mensal = px.bar(
                            dados_lp_mensal,
                            x='Mês',
                            y='ROAS',
                            title=f"ROAS Mensal - {lp_selecionada}",
                            color='ROAS',
                            color_continuous_scale='RdYlGn'
                        )
                        
                        fig_roas_mensal = configurar_layout_clean(fig_roas_mensal, f"ROAS Mensal - {lp_selecionada}")
                        st.plotly_chart(fig_roas_mensal, use_container_width=True)
                    
                    st.subheader(f"Performance Mensal Detalhada - {lp_selecionada}")
                    
                    dados_formatados = dados_lp_mensal.copy()
                    dados_formatados['Receita'] = dados_formatados['Receita'].apply(lambda x: f"R$ {x:,.2f}")
                    dados_formatados['ROAS'] = dados_formatados['ROAS'].apply(lambda x: f"{x:.1f}%")
                    dados_formatados['CAC/LTV'] = dados_formatados['CAC/LTV'].apply(lambda x: f"{x:.2f}")
                    dados_formatados['Leads'] = dados_formatados['Leads'].apply(lambda x: f"{x:,.0f}")
                    
                    st.dataframe(dados_formatados[['Mês', 'Receita', 'Leads', 'ROAS', 'CAC/LTV']], use_container_width=True)
            
            # Análise de Eficiência
            st.subheader("Análise de Eficiência por LP")
            
            col1, col2 = st.columns(2)
            
            with col1:
                lps_saudaveis = performance_lp[performance_lp['CAC/LTV'] < 1.0]
                if not lps_saudaveis.empty:
                    st.success("LPs com Performance Saudável (CAC/LTV < 1.0):")
                    for _, lp in lps_saudaveis.iterrows():
                        st.write(f"- {lp['LP']}: CAC/LTV = {lp['CAC/LTV']:.2f}")
                else:
                    st.warning("Nenhuma LP com CAC/LTV abaixo de 1.0")
            
            with col2:
                lps_problematicas = performance_lp[performance_lp['CAC/LTV'] >= 1.0]
                if not lps_problematicas.empty:
                    st.error("LPs que Precisam de Atenção (CAC/LTV ≥ 1.0):")
                    for _, lp in lps_problematicas.iterrows():
                        st.write(f"- {lp['LP']}: CAC/LTV = {lp['CAC/LTV']:.2f}")
                else:
                    st.success("Todas as LPs têm CAC/LTV saudável")
            
            # Insights e Recomendações
            st.subheader("Insights e Recomendações")
            
            melhor_lp = performance_lp.loc[performance_lp['Receita'].idxmax()]
            pior_lp = performance_lp.loc[performance_lp['Receita'].idxmin()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="matriz-stats">
                    <h4>Melhor Performance: {melhor_lp['LP']}</h4>
                    <ul>
                        <li><strong>Receita:</strong> R$ {melhor_lp['Receita']:,.2f}</li>
                        <li><strong>ROAS:</strong> {melhor_lp['ROAS']:.1f}%</li>
                        <li><strong>CAC/LTV:</strong> {melhor_lp['CAC/LTV']:.2f}</li>
                        <li><strong>Leads:</strong> {melhor_lp['Leads']:,.0f}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="matriz-stats">
                    <h4>Oportunidade de Melhoria: {pior_lp['LP']}</h4>
                    <ul>
                        <li><strong>Receita:</strong> R$ {pior_lp['Receita']:,.2f}</li>
                        <li><strong>ROAS:</strong> {pior_lp['ROAS']:.1f}%</li>
                        <li><strong>CAC/LTV:</strong> {pior_lp['CAC/LTV']:.2f}</li>
                        <li><strong>Leads:</strong> {pior_lp['Leads']:,.0f}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.warning("Não há dados disponíveis para análise de LPs no período selecionado")

# =============================================
# APLICAÇÃO PRINCIPAL
# =============================================

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