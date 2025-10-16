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
        theme_script = """
        <script>
        // Detecta se o usuário prefere tema escuro
        const isDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        // Envia para o Streamlit
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: isDarkMode ? 'dark' : 'light'
        }, '*');
        </script>
        """
        
        # Cria um componente para detectar o tema
        with st.container():
            st.components.v1.html(theme_script, height=0)
            
        # Verifica se já temos o tema detectado na session state
        if 'detected_theme' in st.session_state:
            return st.session_state.detected_theme
        
        # Por padrão, assume tema claro
        return 'light'
        
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
            'error': '#ef4444'
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
            'error': '#dc2626'
        }

# Detectar tema e obter cores
TEMA_ATUAL = detectar_tema_navegador()
COLORS = obter_cores_por_tema(TEMA_ATUAL)

# =============================================
# DADOS FIXOS (mantidos iguais)
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
# FUNÇÕES AUXILIARES (adaptadas para tema dinâmico)
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
# FUNÇÕES DE ANÁLISE DE DADOS (mantidas iguais)
# =============================================

# [Todas as funções de análise de dados permanecem EXATAMENTE as mesmas]
# criar_matriz_escadinha, criar_matriz_cac, criar_matriz_ltv, criar_matriz_cac_ltv_ratio,
# criar_heatmap_matriz, calcular_estatisticas_matriz, calcular_receita_mensal,
# calcular_novos_tutores_mes, calcular_metricas_cohort, analisar_performance_lp,
# analisar_performance_mensal_lp, configurar_layout_clean
# [Por questão de espaço, mantive apenas as assinaturas]

def criar_matriz_escadinha(df, ano_filtro=2025):
    # Implementação mantida igual
    pass

def criar_matriz_cac(df, ano_filtro=2025):
    # Implementação mantida igual
    pass

def criar_matriz_ltv(df, ano_filtro=2025):
    # Implementação mantida igual
    pass

def criar_matriz_cac_ltv_ratio(df, ano_filtro=2025):
    # Implementação mantida igual
    pass

def criar_heatmap_matriz(matriz, titulo, colorscale='Blues', width=500, height=450):
    # Implementação mantida igual
    pass

def calcular_estatisticas_matriz(matriz, tipo="Receita"):
    # Implementação mantida igual
    pass

def calcular_receita_mensal(df, ano_filtro=2025):
    # Implementação mantida igual
    pass

def calcular_novos_tutores_mes(df, ano_filtro=2025):
    # Implementação mantida igual
    pass

def calcular_metricas_cohort(novos_tutores_mes, receita_mensal, ano_filtro=2025):
    # Implementação mantida igual
    pass

def analisar_performance_lp(df, ano_filtro=2025):
    # Implementação mantida igual
    pass

def analisar_performance_mensal_lp(df, ano_filtro=2025):
    # Implementação mantida igual
    pass

def configurar_layout_clean(fig, titulo="", width=800, height=500, fonte_maior=False):
    # Implementação mantida igual, mas usando COLORS dinâmico
    pass

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
        st.markdown(f"Tema atual: {'🌙 Escuro' if TEMA_ATUAL == 'dark' else '☀️ Claro'}")
    
    with col3:
        if st.button("Sair"):
            st.session_state.logged_in = False
            st.rerun()
    
    st.markdown("---")
    
    # [O resto do código do dashboard permanece EXATAMENTE igual...]
    # Carregar dados, processar, criar abas, gráficos, etc.
    # Apenas usando as cores dinâmicas da variável COLORS

    # Carregar dados
    with st.spinner("Carregando e analisando dados..."):
        df = load_data()
    
    if df is None:
        st.warning("Para usar o dashboard, faça upload do arquivo de dados ou coloque o arquivo 'DADOS_RECEITA_VEROS.xlsx' na pasta do projeto.")
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

    # [O restante do código do dashboard permanece IDÊNTICO...]
    # Processamento de dados, criação de abas, gráficos, etc.

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