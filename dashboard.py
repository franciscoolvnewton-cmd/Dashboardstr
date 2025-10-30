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
from plotly.offline import init_notebook_mode
import plotly.io as pio

# Análises avançadas
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.stats as stats
from scipy import signal

# =============================================
# CONFIGURAÇÕES GLOBAIS E TEMAS
# =============================================

# Configurar tema Plotly para modo claro/escuro
def setup_plotly_template():
    template_light = go.layout.Template()
    template_light.layout.plot_bgcolor = '#ffffff'
    template_light.layout.paper_bgcolor = '#ffffff'
    template_light.layout.font.color = '#111827'
    
    template_dark = go.layout.Template()
    template_dark.layout.plot_bgcolor = '#111827'
    template_dark.layout.paper_bgcolor = '#111827'
    template_dark.layout.font.color = '#f9fafb'
    
    return template_light, template_dark

TEMPLATE_LIGHT, TEMPLATE_DARK = setup_plotly_template()

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
    """Detecta se o usuário está usando tema claro ou escuro no navegador"""
    try:
        if 'detected_theme' not in st.session_state:
            st.session_state.detected_theme = 'light'
            
        return st.session_state.detected_theme
        
    except Exception as e:
        return 'light'

def obter_cores_por_tema(tema):
    """Retorna a paleta de cores adequada para o tema detectado"""
    if tema == 'dark':
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
            'paper_bg': '#111827',
            'gradient_1': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'gradient_2': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
            'gradient_3': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            'gradient_4': 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
        }
    else:
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
            'paper_bg': '#ffffff',
            'gradient_1': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'gradient_2': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
            'gradient_3': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            'gradient_4': 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
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

@st.cache_data
def load_leads_data():
    """Carrega os dados de leads do RD"""
    caminhos_dados = [
        "DADOS_RDLEADS.xlsx",
        "data/DADOS_RDLEADS.xlsx", 
        "assets/DADOS_RDLEADS.xlsx",
        "dados/DADOS_RDLEADS.xlsx"
    ]
    
    for file_path in caminhos_dados:
        try:
            if os.path.exists(file_path):
                df = pd.read_excel(file_path, engine='openpyxl')
                
                if 'email' not in df.columns or 'mes_ano_formatado_pt' not in df.columns:
                    st.warning("Colunas 'email' ou 'mes_ano_formatado_pt' não encontradas nos dados de leads")
                    return None
                
                return df
        except Exception as e:
            continue
    
    st.info("Arquivo DADOS_RDLEADS.xlsx não encontrado. A aba de análise de leads não estará disponível.")
    return None

# =============================================
# NOVAS FUNÇÕES DE ANÁLISE AVANÇADA
# =============================================

def criar_analise_sazonalidade(df):
    """Análise avançada de sazonalidade e padrões cíclicos"""
    try:
        receita_mensal_2024, _ = calcular_receita_mensal(df, 2024)
        receita_mensal_2025, _ = calcular_receita_mensal(df, 2025)
        receita_mensal = pd.concat([receita_mensal_2024, receita_mensal_2025], ignore_index=True)
        
        if len(receita_mensal) < 6:
            return None
        
        # Decomposição de sazonalidade
        receita_series = receita_mensal['Receita Bruta'].values
        x = np.arange(len(receita_series))
        
        # Ajustar curva polinomial para tendência
        z = np.polyfit(x, receita_series, 2)
        p = np.poly1d(z)
        tendencia = p(x)
        
        # Remover tendência
        detrended = receita_series - tendencia
        
        # Análise de autocorrelação para sazonalidade
        autocorr = [np.corrcoef(detrended[i:], detrended[:-i])[0,1] for i in range(1, min(13, len(detrended)//2))]
        
        # Identificar período sazonal
        periodo_sazonal = np.argmax(autocorr) + 1 if autocorr else 0
        
        return {
            'tendencia': tendencia,
            'detrended': detrended,
            'autocorr': autocorr,
            'periodo_sazonal': periodo_sazonal,
            'dados': receita_mensal
        }
    except Exception as e:
        st.error(f"Erro na análise de sazonalidade: {e}")
        return None

def analisar_impacto_investimento(df, ano_filtro=2025):
    """Analisa correlação entre investimento e resultados"""
    try:
        receita_mensal, _ = calcular_receita_mensal(df, ano_filtro)
        novos_tutores_mes, _ = calcular_novos_tutores_mes(df, ano_filtro)
        
        if receita_mensal.empty or novos_tutores_mes.empty:
            return None
        
        # Combinar dados
        analise_df = receita_mensal.merge(novos_tutores_mes, on='Mês', how='left')
        
        # Adicionar investimento
        investimento_mensal = INVESTIMENTO_MENSAL_2024 if ano_filtro == 2024 else INVESTIMENTO_MENSAL_2025
        analise_df['Investimento'] = analise_df['Mês'].map(investimento_mensal).fillna(0)
        
        # Calcular correlações
        correlacao_receita = analise_df['Investimento'].corr(analise_df['Receita Bruta'])
        correlacao_tutores = analise_df['Investimento'].corr(analise_df['Novos Tutores'])
        
        # ROI por mês
        analise_df['ROI'] = (analise_df['Receita Líquida'] - analise_df['Investimento']) / analise_df['Investimento'] * 100
        
        return {
            'dados': analise_df,
            'correlacao_receita': correlacao_receita,
            'correlacao_tutores': correlacao_tutores,
            'roi_medio': analise_df['ROI'].mean()
        }
    except Exception as e:
        st.error(f"Erro na análise de impacto: {e}")
        return None

def criar_analise_clusters(df, ano_filtro=2025):
    """Análise de clusters para segmentação de performance"""
    try:
        receita_mensal, _ = calcular_receita_mensal(df, ano_filtro)
        novos_tutores_mes, _ = calcular_novos_tutores_mes(df, ano_filtro)
        
        if receita_mensal.empty or novos_tutores_mes.empty:
            return None
        
        # Combinar métricas para clustering
        cluster_df = receita_mensal.merge(novos_tutores_mes, on='Mês', how='left')
        
        # Adicionar investimento
        investimento_mensal = INVESTIMENTO_MENSAL_2024 if ano_filtro == 2024 else INVESTIMENTO_MENSAL_2025
        cluster_df['Investimento'] = cluster_df['Mês'].map(investimento_mensal).fillna(0)
        
        # Calcular métricas adicionais
        cluster_df['CAC'] = cluster_df['Investimento'] / cluster_df['Novos Tutores']
        cluster_df['Ticket_Medio'] = cluster_df['Receita Bruta'] / cluster_df['Novos Tutores']
        cluster_df['Eficiencia'] = cluster_df['Receita Líquida'] / cluster_df['Investimento']
        
        # Remover infinitos e NaNs
        cluster_df = cluster_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(cluster_df) < 3:
            return None
        
        # Normalizar dados para clustering
        features = ['Receita Bruta', 'Novos Tutores', 'Investimento', 'CAC', 'Ticket_Medio', 'Eficiencia']
        X = cluster_df[features].copy()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=min(3, len(X_scaled)), random_state=42)
        cluster_df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analisar clusters
        analise_clusters = cluster_df.groupby('Cluster').agg({
            'Receita Bruta': 'mean',
            'Novos Tutores': 'mean',
            'Investimento': 'mean',
            'CAC': 'mean',
            'Ticket_Medio': 'mean',
            'Eficiencia': 'mean'
        }).round(2)
        
        return {
            'dados': cluster_df,
            'analise_clusters': analise_clusters,
            'centroides': kmeans.cluster_centers_,
            'features': features
        }
    except Exception as e:
        st.error(f"Erro na análise de clusters: {e}")
        return None

def calcular_metricas_avancadas_lp(df, ano_filtro=2025):
    """Métricas avançadas por LP com análise de performance"""
    try:
        df_filtrado = df[df['Considerar?'] == 'Sim'].copy()
        
        if ano_filtro == 2024:
            df_filtrado = df_filtrado[df_filtrado['Mês geração lead'].str.contains('2024|24', na=False)]
        else:
            df_filtrado = df_filtrado[df_filtrado['Mês geração lead'].str.contains('2025|25', na=False)]
        
        if df_filtrado.empty:
            return pd.DataFrame()
        
        # Agrupar por LP
        metricas_lp = df_filtrado.groupby('LP').agg({
            'E-MAIL': 'nunique',
            'VL UNI': ['sum', 'mean', 'std'],
            'Mês geração lead': 'nunique'
        }).round(2)
        
        # Simplificar colunas
        metricas_lp.columns = ['Leads', 'Receita_Total', 'Ticket_Medio', 'Desvio_Ticket', 'Meses_Ativos']
        
        # Calcular métricas avançadas
        metricas_lp['Investimento'] = metricas_lp.index.map(INVESTIMENTO_POR_LP).fillna(0)
        metricas_lp['CAC'] = metricas_lp['Investimento'] / metricas_lp['Leads']
        metricas_lp['ROI'] = (metricas_lp['Receita_Total'] * 0.56 - metricas_lp['Investimento']) / metricas_lp['Investimento'] * 100
        metricas_lp['Eficiencia'] = metricas_lp['Receita_Total'] * 0.56 / metricas_lp['Investimento']
        metricas_lp['Consistencia'] = 1 / (1 + metricas_lp['Desvio_Ticket'] / metricas_lp['Ticket_Medio'])
        
        # Score composto
        metricas_lp['Score_Performance'] = (
            metricas_lp['ROI'] * 0.3 +
            metricas_lp['Eficiencia'] * 0.3 +
            metricas_lp['Consistencia'] * 0.2 +
            (metricas_lp['Leads'] / metricas_lp['Leads'].max()) * 0.2
        )
        
        return metricas_lp.sort_values('Score_Performance', ascending=False)
        
    except Exception as e:
        st.error(f"Erro no cálculo de métricas avançadas por LP: {e}")
        return pd.DataFrame()

# =============================================
# FUNÇÕES DE ANÁLISE PREDITIVA AVANÇADA
# =============================================

def criar_modelo_preditivo_receita(df, meses_futuros=6):
    """Cria modelo preditivo para receita usando múltiplos algoritmos"""
    if df is None or df.empty:
        return None, None, None, None
    
    try:
        # Preparar dados históricos
        receita_mensal_2024, _ = calcular_receita_mensal(df, 2024)
        receita_mensal_2025, _ = calcular_receita_mensal(df, 2025)
        
        # Combinar dados de ambos os anos
        receita_mensal = pd.concat([receita_mensal_2024, receita_mensal_2025], ignore_index=True)
        
        if receita_mensal.empty or len(receita_mensal) < 3:
            return None, None, None, None
        
        # Criar features temporais
        receita_mensal = receita_mensal.reset_index(drop=True)
        receita_mensal['mes_num'] = range(1, len(receita_mensal) + 1)
        receita_mensal['trimestre'] = (receita_mensal['mes_num'] - 1) // 3 + 1
        receita_mensal['semestre'] = (receita_mensal['mes_num'] - 1) // 6 + 1
        
        # Separar variáveis
        X = receita_mensal[['mes_num', 'trimestre', 'semestre']]
        y = receita_mensal['Receita Bruta']
        
        # Treinar múltiplos modelos
        modelos = {
            'Regressão Linear': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        resultados = {}
        previsoes_combinadas = []
        
        for nome, modelo in modelos.items():
            modelo.fit(X, y)
            previsoes_treino = modelo.predict(X)
            
            # Prever próximos meses
            ultimo_mes = receita_mensal['mes_num'].max()
            meses_futuros_range = range(ultimo_mes + 1, ultimo_mes + meses_futuros + 1)
            
            X_futuro = pd.DataFrame({
                'mes_num': meses_futuros_range,
                'trimestre': [(mes - 1) // 3 + 1 for mes in meses_futuros_range],
                'semestre': [(mes - 1) // 6 + 1 for mes in meses_futuros_range]
            })
            
            previsoes = modelo.predict(X_futuro)
            previsoes_combinadas.append(previsoes)
            
            # Calcular métricas
            r2 = r2_score(y, previsoes_treino)
            mae = mean_absolute_error(y, previsoes_treino)
            
            resultados[nome] = {
                'modelo': modelo,
                'previsoes': previsoes,
                'r2': r2,
                'mae': mae
            }
        
        # Previsão combinada (média dos modelos)
        previsao_final = np.mean(previsoes_combinadas, axis=0)
        
        # Criar DataFrame com previsões
        meses_nomes = [f"Predição {i}" for i in range(1, len(previsao_final) + 1)]
        df_previsoes = pd.DataFrame({
            'Mês': meses_nomes,
            'Receita Bruta Prevista': previsao_final,
            'Receita Líquida Prevista': previsao_final * 0.56
        })
        
        # Calcular intervalo de confiança
        desvio_padrao = np.std(previsoes_combinadas, axis=0)
        df_previsoes['IC Inferior'] = df_previsoes['Receita Bruta Prevista'] - 1.96 * desvio_padrao
        df_previsoes['IC Superior'] = df_previsoes['Receita Bruta Prevista'] + 1.96 * desvio_padrao
        
        return df_previsoes, resultados, receita_mensal, desvio_padrao.mean()
        
    except Exception as e:
        st.error(f"Erro no modelo preditivo: {e}")
        return None, None, None, None

def analisar_tendencia_avancada(df):
    """Análise avançada de tendências e sazonalidade"""
    if df is None or df.empty:
        return None
    
    try:
        # Preparar dados
        receita_mensal_2024, _ = calcular_receita_mensal(df, 2024)
        receita_mensal_2025, _ = calcular_receita_mensal(df, 2025)
        
        receita_mensal = pd.concat([receita_mensal_2024, receita_mensal_2025], ignore_index=True)
        
        if receita_mensal.empty:
            return None
        
        # Análise de crescimento
        receita_mensal['Crescimento'] = receita_mensal['Receita Bruta'].pct_change() * 100
        receita_mensal['Media_Movel_3M'] = receita_mensal['Receita Bruta'].rolling(window=3).mean()
        receita_mensal['Media_Movel_6M'] = receita_mensal['Receita Bruta'].rolling(window=6).mean()
        
        # Identificar padrões sazonais
        if len(receita_mensal) >= 12:
            receita_mensal['Mes_Ano'] = range(1, len(receita_mensal) + 1)
            correlacao_sazonal = receita_mensal['Receita Bruta'].corr(receita_mensal['Mes_Ano'])
        else:
            correlacao_sazonal = 0
        
        # Calcular métricas de performance
        crescimento_total = ((receita_mensal['Receita Bruta'].iloc[-1] - receita_mensal['Receita Bruta'].iloc[0]) / 
                           receita_mensal['Receita Bruta'].iloc[0] * 100) if receita_mensal['Receita Bruta'].iloc[0] > 0 else 0
        
        volatilidade = receita_mensal['Receita Bruta'].std() / receita_mensal['Receita Bruta'].mean() * 100
        
        return {
            'dados': receita_mensal,
            'crescimento_total': crescimento_total,
            'volatilidade': volatilidade,
            'correlacao_sazonal': correlacao_sazonal,
            'media_3m': receita_mensal['Media_Movel_3M'].iloc[-1] if not receita_mensal['Media_Movel_3M'].isna().all() else 0,
            'media_6m': receita_mensal['Media_Movel_6M'].iloc[-1] if not receita_mensal['Media_Movel_6M'].isna().all() else 0
        }
        
    except Exception as e:
        st.error(f"Erro na análise de tendência: {e}")
        return None

def calcular_kpis_avancados(df, ano_filtro=2025):
    """Calcula KPIs avançados com insights estratégicos"""
    if df is None or df.empty:
        return {}
    
    try:
        # Dados básicos
        receita_mensal, _ = calcular_receita_mensal(df, ano_filtro)
        novos_tutores_mes, _ = calcular_novos_tutores_mes(df, ano_filtro)
        cohort_data = calcular_metricas_cohort(novos_tutores_mes, receita_mensal, ano_filtro)
        
        if receita_mensal.empty:
            return {}
        
        # KPIs Básicos
        receita_total = receita_mensal['Receita Bruta'].sum()
        receita_liquida_total = receita_mensal['Receita Líquida'].sum()
        total_tutores = novos_tutores_mes['Novos Tutores'].sum()
        
        # KPIs Avançados
        cac_medio = cohort_data[cohort_data['CAC'] > 0]['CAC'].mean() if not cohort_data.empty else 0
        ltv_medio = cohort_data[cohort_data['LTV'] > 0]['LTV'].mean() if not cohort_data.empty else 0
        roi_medio = cohort_data[cohort_data['ROI (%)'] != 0]['ROI (%)'].mean() if not cohort_data.empty else 0
        
        # Análise de eficiência
        investimento_total = sum(INVESTIMENTO_MENSAL_2025.values()) if ano_filtro == 2025 else sum(INVESTIMENTO_MENSAL_2024.values())
        eficiencia_marketing = (receita_liquida_total / investimento_total) if investimento_total > 0 else 0
        
        # Análise de crescimento
        if len(receita_mensal) > 1:
            crescimento_receita = ((receita_mensal['Receita Bruta'].iloc[-1] - receita_mensal['Receita Bruta'].iloc[0]) / 
                                 receita_mensal['Receita Bruta'].iloc[0] * 100)
        else:
            crescimento_receita = 0
        
        # Análise de consistência
        volatilidade_receita = (receita_mensal['Receita Bruta'].std() / receita_mensal['Receita Bruta'].mean() * 100) if receita_mensal['Receita Bruta'].mean() > 0 else 0
        
        return {
            'receita_total': receita_total,
            'receita_liquida_total': receita_liquida_total,
            'total_tutores': total_tutores,
            'cac_medio': cac_medio,
            'ltv_medio': ltv_medio,
            'roi_medio': roi_medio,
            'eficiencia_marketing': eficiencia_marketing,
            'crescimento_receita': crescimento_receita,
            'volatilidade_receita': volatilidade_receita,
            'investimento_total': investimento_total,
            'ltv_cac_ratio': ltv_medio / cac_medio if cac_medio > 0 else 0
        }
        
    except Exception as e:
        st.error(f"Erro no cálculo de KPIs avançados: {e}")
        return {}

# =============================================
# FUNÇÕES DE VISUALIZAÇÃO MELHORADAS
# =============================================

def criar_grafico_animado_evolucao(dados, x_col, y_col, title, categoria_col=None):
    """Cria gráfico animado com transição suave entre estados"""
    if dados.empty:
        return None
    
    try:
        if categoria_col:
            fig = px.line(
                dados, 
                x=x_col, 
                y=y_col, 
                color=categoria_col,
                title=title,
                template='plotly_white' if TEMA_ATUAL == 'light' else 'plotly_dark'
            )
            
            # Adicionar animação
            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="category"
                ),
                updatemenus=[dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[None, {"frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 300, "easing": "quadratic-in-out"}}],
                            label="Play",
                            method="animate"
                        ),
                        dict(
                            args=[[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate", "transition": {"duration": 0}}],
                            label="Pause",
                            method="animate"
                        )
                    ]),
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                )]
            )
        else:
            fig = px.line(
                dados, 
                x=x_col, 
                y=y_col,
                title=title,
                template='plotly_white' if TEMA_ATUAL == 'light' else 'plotly_dark'
            )
        
        # Melhorar design
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey'))
        )
        
        fig.update_layout(
            hovermode='x unified',
            showlegend=True,
            height=500,
            transition={'duration': 500}
        )
        
        return fig
    except Exception as e:
        st.error(f"Erro ao criar gráfico animado: {e}")
        return None

def criar_heatmap_interativo(matriz, titulo):
    """Cria heatmap interativo com animações"""
    if matriz is None or matriz.empty:
        return None
    
    try:
        # Transpor para exibir Receita no eixo X e Lead no eixo Y
        z_values = matriz.values.T
        x_labels = list(matriz.index)
        y_labels = list(matriz.columns)

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='<b>Mês Receita: %{x}</b><br><b>Mês Lead: %{y}</b><br>Valor: %{z:,.2f}<extra></extra>',
            showscale=True,
            text=[[f'R$ {val:,.0f}' if val > 0 else '' for val in row] for row in z_values],
            texttemplate="%{text}",
            textfont={"size": 10, "color": "white"}
        ))

        fig = configurar_heatmap(fig, titulo, "Valor")
        fig.update_xaxes(title_text="Mês de Geração da Receita")
        fig.update_yaxes(title_text="Mês de Geração do Lead")

        # Adicionar animação de entrada
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 300}}],
                        "label": "Play",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 60},
                "showactive": False,
                "type": "buttons",
                "x": 0.05,
                "xanchor": "left",
                "y": -0.1,
                "yanchor": "top"
            }]
        )

        return fig
    except Exception as e:
        st.error(f"Erro ao criar heatmap interativo: {e}")
        return None

def criar_grafico_radar_performance(metricas_lp):
    """Cria gráfico radar para comparar performance das LPs"""
    if metricas_lp.empty:
        return None
    
    try:
        # Selecionar top LPs
        top_lps = metricas_lp.head(6)
        
        fig = go.Figure()
        
        # Métricas para o radar (normalizadas)
        categorias = ['ROI', 'Eficiencia', 'Consistencia', 'Ticket_Medio', 'Leads']
        
        for lp in top_lps.index:
            valores = [
                top_lps.loc[lp, 'ROI'] / 100,  # Normalizar ROI
                top_lps.loc[lp, 'Eficiencia'],
                top_lps.loc[lp, 'Consistencia'],
                top_lps.loc[lp, 'Ticket_Medio'] / top_lps['Ticket_Medio'].max(),
                top_lps.loc[lp, 'Leads'] / top_lps['Leads'].max()
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=valores + [valores[0]],  # Fechar o polígono
                theta=categorias + [categorias[0]],
                fill='toself',
                name=lp,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Comparação de Performance por LP (Radar)",
            height=500,
            template='plotly_white' if TEMA_ATUAL == 'light' else 'plotly_dark'
        )
        
        return fig
    except Exception as e:
        st.error(f"Erro ao criar gráfico radar: {e}")
        return None

def criar_dashboard_interativo_kpis(kpis_avancados):
    """Cria dashboard interativo com KPIs animados"""
    if not kpis_avancados:
        return None
    
    try:
        # Criar subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=("ROI Médio", "LTV/CAC Ratio", "Eficiência Marketing", "Crescimento")
        )
        
        # ROI Médio - Gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = kpis_avancados.get('roi_medio', 0),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "ROI Médio (%)"},
            gauge = {
                'axis': {'range': [None, max(100, kpis_avancados.get('roi_medio', 0) * 1.2)]},
                'bar': {'color': COLORS['primary']},
                'steps': [
                    {'range': [0, 50], 'color': COLORS['error']},
                    {'range': [50, 100], 'color': COLORS['warning']},
                    {'range': [100, 200], 'color': COLORS['success']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ), row=1, col=1)
        
        # LTV/CAC Ratio
        fig.add_trace(go.Indicator(
            mode = "number+delta",
            value = kpis_avancados.get('ltv_cac_ratio', 0),
            number = {'prefix': "x"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "LTV/CAC Ratio"},
            delta = {'reference': 1, 'position': "bottom"}
        ), row=1, col=2)
        
        # Eficiência Marketing
        fig.add_trace(go.Indicator(
            mode = "number+delta",
            value = kpis_avancados.get('eficiencia_marketing', 0),
            number = {'valueformat': ".2f"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Eficiência Marketing"},
            delta = {'reference': 1, 'position': "bottom"}
        ), row=2, col=1)
        
        # Crescimento
        fig.add_trace(go.Indicator(
            mode = "number+delta",
            value = kpis_avancados.get('crescimento_receita', 0),
            number = {'suffix': "%"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Crescimento Receita"},
            delta = {'reference': 0, 'position': "bottom"}
        ), row=2, col=2)
        
        fig.update_layout(
            height=400,
            template='plotly_white' if TEMA_ATUAL == 'light' else 'plotly_dark',
            paper_bgcolor=COLORS['paper_bg'],
            font={'color': COLORS['text_primary']}
        )
        
        return fig
    except Exception as e:
        st.error(f"Erro ao criar dashboard interativo: {e}")
        return None

def criar_analise_sazonalidade_grafico(analise_sazonal):
    """Cria gráfico animado de análise de sazonalidade"""
    if not analise_sazonal:
        return None
    
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Série Temporal Original", 
                "Tendência", 
                "Componente Sazonal", 
                "Autocorrelação"
            )
        )
        
        dados = analise_sazonal['dados']
        
        # Série original
        fig.add_trace(
            go.Scatter(
                x=dados['Mês'], 
                y=dados['Receita Bruta'],
                mode='lines+markers',
                name='Receita Bruta',
                line=dict(color=COLORS['primary'], width=3)
            ), row=1, col=1
        )
        
        # Tendência
        fig.add_trace(
            go.Scatter(
                x=dados['Mês'], 
                y=analise_sazonal['tendencia'],
                mode='lines',
                name='Tendência',
                line=dict(color=COLORS['secondary'], width=3, dash='dash')
            ), row=1, col=2
        )
        
        # Componente sazonal
        fig.add_trace(
            go.Scatter(
                x=dados['Mês'], 
                y=analise_sazonal['detrended'],
                mode='lines+markers',
                name='Sazonalidade',
                line=dict(color=COLORS['warning'], width=2)
            ), row=2, col=1
        )
        
        # Autocorrelação
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(analise_sazonal['autocorr'])+1)),
                y=analise_sazonal['autocorr'],
                name='Autocorrelação',
                marker_color=COLORS['info']
            ), row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            template='plotly_white' if TEMA_ATUAL == 'light' else 'plotly_dark',
            title_text="Análise Avançada de Sazonalidade"
        )
        
        return fig
    except Exception as e:
        st.error(f"Erro ao criar gráfico de sazonalidade: {e}")
        return None

# =============================================
# FUNÇÕES EXISTENTES (MANTIDAS)
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
            from {{
                opacity: 0;
                transform: translateY(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
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
    Cria um heatmap visual da matriz escadinha COM RÓTULOS DE DADOS
    """
    if matriz is None or matriz.empty:
        return None
    
    matriz_plot = matriz.copy()
    
    x_labels = list(matriz_plot.index)
    y_labels = list(matriz_plot.columns)
    z_values = matriz_plot.values.T

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate='<b>Mês Receita: %{x}</b><br><b>Mês Lead: %{y}</b><br>Valor: %{z:,.2f}<extra></extra>',
        showscale=True,
        text=[[f'R$ {val:,.0f}' if val > 0 else '' for val in row] for row in z_values],
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        colorbar=dict(title=dict(text="Valor", side="right"))
    ))

    # Destacar diagonal principal (meses iguais)
    col_lookup = {mes: idx for idx, mes in enumerate(y_labels)}
    for idx, mes in enumerate(x_labels):
        if mes in col_lookup:
            j = col_lookup[mes]
            fig.add_shape(
                type="rect",
                x0=idx - 0.5, y0=j - 0.5, x1=idx + 0.5, y1=j + 0.5,
                line=dict(color=COLORS['accent'], width=2),
                fillcolor="rgba(0,0,0,0)"
            )

    fig = configurar_heatmap(fig, titulo, "Valor")
    fig.update_layout(width=width, height=height)
    fig.update_xaxes(title_text="Mês de Geração da Receita", tickangle=45, tickfont=dict(size=10, color=COLORS['text_primary']))
    fig.update_yaxes(title_text="Mês de Geração do Lead", tickfont=dict(size=10, color=COLORS['text_primary']))

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

def configurar_layout_clean(fig, titulo="", width=800, height=500, fonte_maior=False, show_labels=True):
    """
    Configura layout dos gráficos COM RÓTULOS DE DADOS
    """
    eixo_base = dict(
        title=dict(font=dict(size=16, color=COLORS['text_primary']), standoff=20),
        tickfont=dict(size=14, color=COLORS['text_primary']),
        gridcolor=COLORS['light_gray'],
        griddash="dot",
        zeroline=False,
        showline=True,
        linecolor=COLORS['border'],
        linewidth=1,
        mirror=True,
        ticks="outside",
        ticklen=6,
        tickcolor=COLORS['border']
    )

    if fonte_maior:
        # Configuração com fontes maiores para gráficos de LP
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=16, color=COLORS['text_primary']),
            title=dict(
                text=titulo,
                font=dict(size=24, color=COLORS['text_primary'], family="Arial, sans-serif"),
                x=0.5,
                xanchor='center',
                y=0.95,
                pad=dict(b=20)
            ),
            xaxis={**eixo_base, "tickfont": dict(size=16, color=COLORS['text_primary'])},
            yaxis={**eixo_base, "tickfont": dict(size=16, color=COLORS['text_primary']), "tickformat": ",."},
            legend=dict(
                font=dict(size=16, color=COLORS['text_primary']),
                orientation="h",
                yanchor="top",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                itemwidth=60
            ),
            plot_bgcolor=COLORS['plot_bg'],
            paper_bgcolor=COLORS['paper_bg'],
            margin=dict(l=50, r=30, t=90, b=60),
            hoverlabel=dict(
                bgcolor=COLORS['background'],
                bordercolor=COLORS['border'],
                font_size=16,
                font_family="Arial"
            ),
            hovermode='x unified',
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
                y=0.95,
                pad=dict(b=20)
            ),
            xaxis=eixo_base,
            yaxis={**eixo_base, "tickformat": ",."},
            legend=dict(
                font=dict(size=14, color=COLORS['text_primary']),
                orientation="h",
                yanchor="top",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                itemwidth=60
            ),
            plot_bgcolor=COLORS['plot_bg'],
            paper_bgcolor=COLORS['paper_bg'],
            margin=dict(l=45, r=30, t=80, b=55),
            hoverlabel=dict(
                bgcolor=COLORS['background'],
                bordercolor=COLORS['border'],
                font_size=14,
                font_family="Arial"
            ),
            hovermode='x unified',
            width=width,
            height=height
        )

    # Ajustes finos dos eixos
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['light_gray'], griddash="dot")
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['light_gray'], griddash="dot")

    # Estilizar traços conforme o tipo
    try:
        for trace in fig.data:
            if isinstance(trace, go.Bar):
                marker_color = trace.marker.color if hasattr(trace, "marker") and getattr(trace.marker, "color", None) is not None else COLORS['primary']
                marker_conf = dict(
                    color=marker_color,
                    opacity=0.9,
                    line=dict(width=0)
                )
                if hasattr(trace.marker, "pattern") and getattr(trace.marker.pattern, "shape", None):
                    marker_conf["pattern"] = dict(shape=trace.marker.pattern.shape)
                trace.update(
                    marker=marker_conf,
                    hovertemplate="<b>%{x}</b><br>Valor: %{y:,.0f}<extra></extra>"
                )
            elif isinstance(trace, go.Scatter):
                mode = trace.mode or ""
                if "lines" in mode and "markers" not in mode:
                    trace.mode = mode + "+markers"
                trace.update(
                    line=dict(width=max(trace.line.width if trace.line and trace.line.width else 2, 3)),
                    marker=dict(
                        size=8,
                        line=dict(width=1.5, color=COLORS['paper_bg']),
                        symbol="circle",
                        opacity=0.95
                    ),
                    hovertemplate="<b>%{x}</b><br>Valor: %{y:,.0f}<extra></extra>"
                )
            elif isinstance(trace, go.Funnel):
                trace.update(hovertemplate="<b>%{label}</b><br>Valor: %{value:,.0f}<extra></extra>")
    except Exception:
        pass

    # Adicionar rótulos de dados se solicitado
    if show_labels:
        try:
            fig.update_traces(
                texttemplate='%{y:,.0f}',
                textposition='top center',
                textfont=dict(size=12, color=COLORS['text_primary'])
            )
        except:
            pass

    return fig


def configurar_heatmap(fig, titulo="", colorbar_titulo="Valor"):
    """Aplica tema consistente aos heatmaps."""
    if fig is None:
        return None

    for trace in fig.data:
        if isinstance(trace, go.Heatmap):
            trace.update(
                colorscale='Tealgrn',
                hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Valor: %{z:,.0f}<extra></extra>",
                colorbar=dict(
                    title=dict(text=colorbar_titulo, font=dict(size=12, color=COLORS['text_primary'])),
                    tickformat=",.",
                    thickness=14,
                    len=0.65,
                    bgcolor=COLORS['paper_bg'],
                    outlinewidth=0
                )
            )

    fig.update_layout(
        title=dict(
            text=titulo,
            font=dict(size=20, color=COLORS['text_primary'], family="Arial, sans-serif"),
            x=0.5,
            xanchor='center'
        ),
        font=dict(family="Arial, sans-serif", size=12, color=COLORS['text_primary']),
        plot_bgcolor=COLORS['plot_bg'],
        paper_bgcolor=COLORS['paper_bg'],
        margin=dict(l=60, r=40, t=80, b=60),
        hoverlabel=dict(
            bgcolor=COLORS['background'],
            bordercolor=COLORS['border'],
            font_size=13,
            font_family="Arial"
        )
    )

    fig.update_xaxes(
        showgrid=False,
        tickfont=dict(size=11, color=COLORS['text_primary']),
        tickangle=0,
        title=dict(font=dict(size=12, color=COLORS['text_primary']))
    )
    fig.update_yaxes(
        showgrid=False,
        tickfont=dict(size=11, color=COLORS['text_primary']),
        title=dict(font=dict(size=12, color=COLORS['text_primary']))
    )

    return fig


def render_plotly_chart(fig, key=None, use_container_width=True):
    """Envolve gráficos Plotly em um card animado padronizado."""
    if fig is None:
        return

    with st.container():
        st.markdown('<div class="chart-card animate-fade-up">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=use_container_width, key=key)
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================
# FUNÇÕES DE ANÁLISE DE LEADS ATUALIZADAS
# =============================================

def analisar_leads_consolidado_mes(df_leads, df_receita, ano_filtro=2025):
    """Análise consolidada mensal de leads COM DADOS DA BASE DE RECEITA"""
    try:
        # AGORA USANDO APENAS A BASE DE RECEITA PARA CALCULAR LEADS
        if df_receita is None or df_receita.empty:
            return pd.DataFrame()
        
        # Filtrar dados da base de receita
        df_filtrado = df_receita[df_receita['Considerar?'] == 'Sim'].copy()
        
        if ano_filtro == 2024:
            df_filtrado = df_filtrado[df_filtrado['Mês geração lead'].str.contains('2024|24', na=False)]
        else:
            df_filtrado = df_filtrado[df_filtrado['Mês geração lead'].str.contains('2025|25', na=False)]
        
        if df_filtrado.empty:
            return pd.DataFrame()
        
        # Definir ordem dos meses
        if ano_filtro == 2024:
            ordem_meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                          'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
        else:
            ordem_meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                          'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
        
        # Calcular LEADS ÚNICOS por mês (baseado na base de receita)
        leads_por_mes = df_filtrado.groupby('Mês geração lead').agg({
            'E-MAIL': 'nunique',  # Conta emails únicos como leads
            'VL UNI': 'sum'       # Soma da receita
        }).reset_index()
        
        leads_por_mes.columns = ['Mês', 'Leads', 'Realizado']
        
        # Ordenar por ordem dos meses
        leads_por_mes['Mês_Ordenado'] = pd.Categorical(leads_por_mes['Mês'], categories=ordem_meses, ordered=True)
        leads_por_mes = leads_por_mes.sort_values('Mês_Ordenado').drop('Mês_Ordenado', axis=1)
        
        # Adicionar meses faltantes
        meses_df = pd.DataFrame({'Mês': ordem_meses})
        leads_por_mes = meses_df.merge(leads_por_mes, on='Mês', how='left').fillna(0)
        
        # Adicionar investimento
        investimento_mensal = INVESTIMENTO_MENSAL_2024 if ano_filtro == 2024 else INVESTIMENTO_MENSAL_2025
        leads_por_mes['Investimento'] = leads_por_mes['Mês'].map(investimento_mensal).fillna(0)
        
        # Calcular métricas
        leads_por_mes['CPL'] = leads_por_mes.apply(
            lambda x: x['Investimento'] / x['Leads'] if x['Leads'] > 0 else 0, axis=1
        )
        
        # Calcular métricas derivadas
        leads_por_mes['Tx.Conv'] = leads_por_mes.apply(
            lambda x: (x['Realizado'] / x['Investimento'] * 100) if x['Investimento'] > 0 else 0, axis=1
        )
        
        # Calcular tutores únicos por mês (já temos nos 'Leads')
        leads_por_mes['Tutores'] = leads_por_mes['Leads']
        
        leads_por_mes['CAC'] = leads_por_mes.apply(
            lambda x: x['Investimento'] / x['Tutores'] if x['Tutores'] > 0 else 0, axis=1
        )
        
        leads_por_mes['Receita'] = leads_por_mes['Realizado'] * 0.56
        leads_por_mes['ROAS'] = leads_por_mes.apply(
            lambda x: (x['Receita'] / x['Investimento'] * 100) if x['Investimento'] > 0 else 0, axis=1
        )
        
        leads_por_mes['TM'] = leads_por_mes.apply(
            lambda x: x['Realizado'] / x['Tutores'] if x['Tutores'] > 0 else 0, axis=1
        )
        
        leads_por_mes['LTV'] = leads_por_mes.apply(
            lambda x: x['Receita'] / x['Tutores'] if x['Tutores'] > 0 else 0, axis=1
        )
        
        leads_por_mes['CAC/LTV'] = leads_por_mes.apply(
            lambda x: x['CAC'] / x['LTV'] if x['LTV'] > 0 else 0, axis=1
        )
        
        return leads_por_mes
        
    except Exception as e:
        st.error(f"Erro na análise consolidada de leads: {e}")
        return pd.DataFrame()

def analisar_leads_por_lp(df_leads, df_receita, ano_filtro=2025):
    """Análise de leads por LP (categoria) COM DADOS DA BASE DE RECEITA"""
    try:
        # AGORA USANDO APENAS A BASE DE RECEITA
        if df_receita is None or df_receita.empty:
            return pd.DataFrame()
        
        # Filtrar dados da base de receita
        df_filtrado = df_receita[df_receita['Considerar?'] == 'Sim'].copy()
        
        if ano_filtro == 2024:
            df_filtrado = df_filtrado[df_filtrado['Mês geração lead'].str.contains('2024|24', na=False)]
        else:
            df_filtrado = df_filtrado[df_filtrado['Mês geração lead'].str.contains('2025|25', na=False)]
        
        if df_filtrado.empty:
            return pd.DataFrame()
        
        # Agrupar por LP
        leads_por_lp = df_filtrado.groupby('LP').agg({
            'E-MAIL': 'nunique',  # Leads únicos
            'VL UNI': 'sum'       # Receita realizada
        }).reset_index()
        
        leads_por_lp.columns = ['LP', 'Leads', 'Realizado']
        
        # Adicionar investimento
        leads_por_lp['Investimento'] = leads_por_lp['LP'].map(INVESTIMENTO_POR_LP).fillna(0)
        
        # Calcular métricas
        leads_por_lp['CPL'] = leads_por_lp.apply(
            lambda x: x['Investimento'] / x['Leads'] if x['Leads'] > 0 else 0, axis=1
        )
        
        leads_por_lp['Tx.Conv'] = leads_por_lp.apply(
            lambda x: (x['Realizado'] / x['Investimento'] * 100) if x['Investimento'] > 0 else 0, axis=1
        )
        
        leads_por_lp['Tutores'] = leads_por_lp['Leads']
        
        leads_por_lp['CAC'] = leads_por_lp.apply(
            lambda x: x['Investimento'] / x['Tutores'] if x['Tutores'] > 0 else 0, axis=1
        )
        
        leads_por_lp['Receita'] = leads_por_lp['Realizado'] * 0.56
        leads_por_lp['ROAS'] = leads_por_lp.apply(
            lambda x: (x['Receita'] / x['Investimento'] * 100) if x['Investimento'] > 0 else 0, axis=1
        )
        
        leads_por_lp['TM'] = leads_por_lp.apply(
            lambda x: x['Realizado'] / x['Tutores'] if x['Tutores'] > 0 else 0, axis=1
        )
        
        leads_por_lp['LTV'] = leads_por_lp.apply(
            lambda x: x['Receita'] / x['Tutores'] if x['Tutores'] > 0 else 0, axis=1
        )
        
        leads_por_lp['CAC/LTV'] = leads_por_lp.apply(
            lambda x: x['CAC'] / x['LTV'] if x['LTV'] > 0 else 0, axis=1
        )
        
        return leads_por_lp.sort_values('Receita', ascending=False)
        
    except Exception as e:
        st.error(f"Erro na análise de leads por LP: {e}")
        return pd.DataFrame()

def analisar_leads_por_lp_mensal(df_leads, df_receita, ano_filtro=2025):
    """Análise mensal de leads por LP COM DADOS DA BASE DE RECEITA"""
    try:
        # AGORA USANDO APENAS A BASE DE RECEITA
        if df_receita is None or df_receita.empty:
            return pd.DataFrame()
        
        # Filtrar dados da base de receita
        df_filtrado = df_receita[df_receita['Considerar?'] == 'Sim'].copy()
        
        if ano_filtro == 2024:
            df_filtrado = df_filtrado[df_filtrado['Mês geração lead'].str.contains('2024|24', na=False)]
        else:
            df_filtrado = df_filtrado[df_filtrado['Mês geração lead'].str.contains('2025|25', na=False)]
        
        if df_filtrado.empty:
            return pd.DataFrame()
        
        # Definir ordem dos meses
        if ano_filtro == 2024:
            ordem_meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                          'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
        else:
            ordem_meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                          'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
        
        # Agrupar por LP e mês
        leads_por_lp_mes = df_filtrado.groupby(['LP', 'Mês geração lead']).agg({
            'E-MAIL': 'nunique',
            'VL UNI': 'sum'
        }).reset_index()
        
        leads_por_lp_mes.columns = ['LP', 'Mês', 'Leads', 'Realizado']
        
        # Adicionar investimento mensal proporcional
        investimento_total_por_lp = leads_por_lp_mes.groupby('LP')['Leads'].sum().reset_index()
        investimento_total_por_lp['Investimento_Total'] = investimento_total_por_lp['LP'].map(INVESTIMENTO_POR_LP).fillna(0)
        
        leads_por_lp_mes = leads_por_lp_mes.merge(investimento_total_por_lp[['LP', 'Investimento_Total']], on='LP', how='left')
        
        # Calcular investimento mensal proporcional aos leads
        total_leads_por_lp = leads_por_lp_mes.groupby('LP')['Leads'].sum().reset_index()
        total_leads_por_lp.columns = ['LP', 'Total_Leads']
        
        leads_por_lp_mes = leads_por_lp_mes.merge(total_leads_por_lp, on='LP', how='left')
        leads_por_lp_mes['Investimento'] = leads_por_lp_mes.apply(
            lambda x: (x['Leads'] / x['Total_Leads']) * x['Investimento_Total'] if x['Total_Leads'] > 0 else 0, axis=1
        )
        
        # Calcular métricas
        leads_por_lp_mes['CPL'] = leads_por_lp_mes.apply(
            lambda x: x['Investimento'] / x['Leads'] if x['Leads'] > 0 else 0, axis=1
        )
        
        leads_por_lp_mes['Tx.Conv'] = leads_por_lp_mes.apply(
            lambda x: (x['Realizado'] / x['Investimento'] * 100) if x['Investimento'] > 0 else 0, axis=1
        )
        
        leads_por_lp_mes['Tutores'] = leads_por_lp_mes['Leads']
        
        leads_por_lp_mes['CAC'] = leads_por_lp_mes.apply(
            lambda x: x['Investimento'] / x['Tutores'] if x['Tutores'] > 0 else 0, axis=1
        )
        
        leads_por_lp_mes['Receita'] = leads_por_lp_mes['Realizado'] * 0.56
        leads_por_lp_mes['ROAS'] = leads_por_lp_mes.apply(
            lambda x: (x['Receita'] / x['Investimento'] * 100) if x['Investimento'] > 0 else 0, axis=1
        )
        
        leads_por_lp_mes['TM'] = leads_por_lp_mes.apply(
            lambda x: x['Realizado'] / x['Tutores'] if x['Tutores'] > 0 else 0, axis=1
        )
        
        leads_por_lp_mes['LTV'] = leads_por_lp_mes.apply(
            lambda x: x['Receita'] / x['Tutores'] if x['Tutores'] > 0 else 0, axis=1
        )
        
        leads_por_lp_mes['CAC/LTV'] = leads_por_lp_mes.apply(
            lambda x: x['CAC'] / x['LTV'] if x['LTV'] > 0 else 0, axis=1
        )
        
        # Ordenar por mês
        leads_por_lp_mes['Mês_Ordenado'] = pd.Categorical(leads_por_lp_mes['Mês'], categories=ordem_meses, ordered=True)
        leads_por_lp_mes = leads_por_lp_mes.sort_values(['LP', 'Mês_Ordenado']).drop('Mês_Ordenado', axis=1)
        
        return leads_por_lp_mes
        
    except Exception as e:
        st.error(f"Erro na análise mensal de leads por LP: {e}")
        return pd.DataFrame()

# =============================================
# DASHBOARD PRINCIPAL COMPLETO
# =============================================

def main_dashboard():
    # Atualizar tema dinamicamente
    global TEMA_ATUAL, COLORS
    TEMA_ATUAL = detectar_tema_navegador()
    COLORS = obter_cores_por_tema(TEMA_ATUAL)
    
    # Configuração CSS completa com animações
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
    
    /* Animações CSS */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    
    @keyframes slideInLeft {{
        from {{
            opacity: 0;
            transform: translateX(-30px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}

    @keyframes fadeInUpSoft {{
        from {{
            opacity: 0;
            transform: translateY(24px) scale(0.98);
        }}
        to {{
            opacity: 1;
            transform: translateY(0) scale(1);
        }}
    }}

    @keyframes glowPulse {{
        0% {{ box-shadow: 0 0 0 rgba(37, 99, 235, 0.15); }}
        50% {{ box-shadow: 0 0 22px rgba(37, 99, 235, 0.25); }}
        100% {{ box-shadow: 0 0 0 rgba(37, 99, 235, 0.15); }}
    }}
    
    /* Aplicar animações */
    .animated-card {{
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .pulse-hover:hover {{
        animation: pulse 0.6s ease-in-out;
    }}
    
    .slide-in {{
        animation: slideInLeft 0.5s ease-out;
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
        animation: fadeInUpSoft 0.6s ease both;
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
    
    /* Melhorias visuais adicionais */
    .metric-card {{
        background: {COLORS['gradient_1']};
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }}
    
    .glow-effect {{
        box-shadow: 0 0 20px {COLORS['primary']}30;
    }}

    .chart-card {{
        background: linear-gradient(145deg, {COLORS['background']} 0%, {COLORS['paper_bg']} 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 18px 35px rgba(15, 23, 42, 0.12);
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}

    .chart-card::before {{
        content: "";
        position: absolute;
        inset: -60% 20% auto -40%;
        height: 120%;
        background: radial-gradient(circle at center, {COLORS['primary']}25, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }}

    .chart-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 24px 45px rgba(15, 23, 42, 0.18);
    }}

    .chart-card:hover::before {{
        opacity: 1;
    }}

    .animate-fade-up {{
        animation: fadeInUpSoft 0.75s ease both;
    }}
    
    /* Gradientes modernos */
    .gradient-primary {{
        background: {COLORS['gradient_1']};
    }}
    
    .gradient-success {{
        background: {COLORS['gradient_4']};
    }}
    
    .gradient-warning {{
        background: {COLORS['gradient_2']};
    }}
    
    .gradient-info {{
        background: {COLORS['gradient_3']};
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
        df_leads = load_leads_data()
    
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
        if df_leads is not None:
            st.info(f"Leads carregados: {len(df_leads)} registros")
        
        # Filtro de Ano
        st.subheader("Filtro de Período")
        ano_selecionado = st.selectbox(
            "Selecione o ano:",
            options=[2024, 2025],
            index=1
        )
        
        st.info(f"Filtro Ativo: Ano {ano_selecionado}")
        
        # Novos filtros avançados
        st.subheader("Análises Avançadas")
        analise_sazonal = st.checkbox("Análise de Sazonalidade", value=True)
        analise_clusters = st.checkbox("Segmentação por Clusters", value=True)
        analise_impacto = st.checkbox("Análise de Impacto do Investimento", value=True)
    
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
        
        # Criar heatmaps COM RÓTULOS
        heatmap_receita = criar_heatmap_matriz(matriz_receita, "Receita Bruta", 'Blues', 500, 450)
        heatmap_cac = criar_heatmap_matriz(matriz_cac, "CAC (Custo Aquisição)", 'Reds', 500, 450)
        heatmap_ltv = criar_heatmap_matriz(matriz_ltv, "LTV (Valor Cliente)", 'Greens', 500, 450)
        heatmap_cac_ltv = criar_heatmap_matriz(matriz_cac_ltv, "Razão CAC/LTV", 'RdYlGn_r', 500, 450)
        
        # Calcular estatísticas
        estatisticas_receita = calcular_estatisticas_matriz(matriz_receita, "Receita") if matriz_receita is not None else None
        estatisticas_cac = calcular_estatisticas_matriz(matriz_cac, "CAC") if matriz_cac is not None else None
        estatisticas_ltv = calcular_estatisticas_matriz(matriz_ltv, "LTV") if matriz_ltv is not None else None
        estatisticas_cac_ltv = calcular_estatisticas_matriz(matriz_cac_ltv, "CAC/LTV") if matriz_cac_ltv is not None else None
        
        # Processar dados de leads (AGORA USANDO BASE DE RECEITA)
        leads_consolidado = analisar_leads_consolidado_mes(df_leads, df, ano_selecionado)
        leads_por_lp = analisar_leads_por_lp(df_leads, df, ano_selecionado)
        leads_por_lp_mensal = analisar_leads_por_lp_mensal(df_leads, df, ano_selecionado)
        
        # Análises preditivas e avançadas
        df_previsoes, resultados_modelos, dados_historicos, desvio_padrao = criar_modelo_preditivo_receita(df)
        analise_tendencia = analisar_tendencia_avancada(df)
        kpis_avancados = calcular_kpis_avancados(df, ano_selecionado)
        
        # NOVAS ANÁLISES
        if analise_sazonal:
            analise_sazonal_data = criar_analise_sazonalidade(df)
        
        if analise_impacto:
            analise_impacto_data = analisar_impacto_investimento(df, ano_selecionado)
        
        if analise_clusters:
            analise_clusters_data = criar_analise_clusters(df, ano_selecionado)
        
        metricas_avancadas_lp = calcular_metricas_avancadas_lp(df, ano_selecionado)
    
    # SISTEMA DE ABAS COMPLETO
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Visão Geral", 
        "Matrizes Escadinha",
        "LP Leads", 
        "Análise Preditiva",
        "Análises Avançadas"
    ])
    
    with tab1:
        st.markdown(f'<div class="section-header"><h2 class="section-title">Visão Geral do Performance</h2><p class="section-subtitle">Métricas consolidadas e tendências do período</p></div>', unsafe_allow_html=True)
        
        if not receita_mensal.empty:
            # Resumo das outras abas
            st.subheader("Resumo das Análises")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**Matrizes Escadinha**")
                if estatisticas_receita:
                    st.metric("Eficiência Temporal", f"{estatisticas_receita['eficiencia']:.1%}")
            
            with col2:
                st.markdown("**LP e Leads**")
                if not leads_por_lp.empty:
                    top_lp = leads_por_lp.iloc[0]
                    st.metric("Top LP", top_lp['LP'])
            
            with col3:
                st.markdown("**Análise Preditiva**")
                if df_previsoes is not None:
                    previsao_media = df_previsoes['Receita Bruta Prevista'].mean()
                    st.metric("Previsão Média", f"R$ {previsao_media:,.0f}")
            
            with col4:
                st.markdown("**Análises Avançadas**")
                if kpis_avancados:
                    st.metric("ROI Médio", f"{kpis_avancados.get('roi_medio', 0):.1f}%")

            # KPIs Avançados - APENAS MÉTRICAS PRINCIPAIS
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
            
            # Mais KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if not cohort_data.empty:
                    ltv_medio = cohort_data[cohort_data['LTV'] > 0]['LTV'].mean()
                    st.metric(
                        "LTV Médio", 
                        f"R$ {ltv_medio:,.0f}"
                    )
            
            with col2:
                if not cohort_data.empty:
                    roi_medio = cohort_data[cohort_data['ROI (%)'] != 0]['ROI (%)'].mean()
                    st.metric(
                        "ROI Médio", 
                        f"{roi_medio:.1f}%"
                    )
            
            with col3:
                if kpis_avancados and kpis_avancados.get('crescimento_receita', 0) != 0:
                    st.metric(
                        "Crescimento Receita", 
                        f"{kpis_avancados.get('crescimento_receita', 0):.1f}%"
                    )
            
            with col4:
                if kpis_avancados:
                    st.metric(
                        "Volatilidade", 
                        f"{kpis_avancados.get('volatilidade_receita', 0):.1f}%"
                    )
            
            # Gráficos Principais COM RÓTULOS
            st.subheader("Evolução Mensal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de Receita Mensal COM RÓTULOS
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
                    mode='lines+markers+text',
                    texttemplate='%{y:,.0f}',
                    textposition='top center',
                    textfont=dict(size=10, color=COLORS['text_primary'])
                )
                
                fig_receita = configurar_layout_clean(fig_receita, "Receita Mensal", show_labels=True)
                render_plotly_chart(fig_receita)
            
            with col2:
                # Gráfico de Novos Tutores COM RÓTULOS
                fig_tutores = px.bar(
                    novos_tutores_mes,
                    x='Mês',
                    y='Novos Tutores',
                    title="Novos Tutores por Mês",
                    color_discrete_sequence=[COLORS['info']]
                )
                
                fig_tutores.update_traces(
                    texttemplate='%{y}',
                    textposition='outside',
                    textfont=dict(size=10, color=COLORS['text_primary'])
                )
                
                fig_tutores = configurar_layout_clean(fig_tutores, "Novos Tutores por Mês", show_labels=True)
                render_plotly_chart(fig_tutores)
            
            # Análise de Cohort
            if not cohort_data.empty:
                st.subheader("Análise de Cohort - Métricas por Mês")
                
                # Formatar cohort_data para exibição
                cohort_formatado = cohort_data.copy()
                colunas_monetarias = ['Investimento', 'CAC', 'LTV']
                for col in colunas_monetarias:
                    if col in cohort_formatado.columns:
                        cohort_formatado[col] = cohort_formatado[col].apply(lambda x: f"R$ {x:,.0f}" if x > 0 else "R$ 0")
                
                cohort_formatado['ROI (%)'] = cohort_formatado['ROI (%)'].apply(lambda x: f"{x:.1f}%")
                cohort_formatado['CAC/LTV'] = cohort_formatado['CAC/LTV'].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(cohort_formatado, use_container_width=True)
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
            # SEÇÃO 1: VISUALIZAÇÕES DAS MATRIZES (COM RÓTULOS)
            st.subheader("Visualizações das Matrizes")
            
            st.markdown('<div class="heatmap-grid">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if heatmap_receita:
                    render_plotly_chart(heatmap_receita)
                else:
                    st.warning("Matriz de Receita não disponível")
                
                if heatmap_cac:
                    render_plotly_chart(heatmap_cac)
                else:
                    st.warning("Matriz de CAC não disponível")
            
            with col2:
                if heatmap_ltv:
                    render_plotly_chart(heatmap_ltv)
                else:
                    st.warning("Matriz de LTV não disponível")
                
                if heatmap_cac_ltv:
                    render_plotly_chart(heatmap_cac_ltv)
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
        st.markdown(f'<div class="section-header"><h2 class="section-title">Análise de Leads - Base Receita</h2><p class="section-subtitle">Métricas de performance baseadas nos dados de RECEITA (E-MAILS ÚNICOS)</p></div>', unsafe_allow_html=True)
        
        st.info("🔍 **Dados calculados a partir da base de RECEITA** - Utilizando E-MAILS únicos como indicador de leads convertidos")
        
        # CONSOLIDADO POR MES
        st.subheader("CONSOLIDADO POR MÊS")
        if not leads_consolidado.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1: 
                st.metric("Total Leads (E-MAILS)", f"{leads_consolidado['Leads'].sum():,}")
            with col2: 
                st.metric("CPL Médio", f"R$ {leads_consolidado['CPL'].mean():.2f}")
            with col3: 
                st.metric("Receita Total", f"R$ {leads_consolidado['Receita'].sum():,.0f}")
            with col4: 
                st.metric("ROAS Médio", f"{leads_consolidado['ROAS'].mean():.1f}%")
            
            # Formatar tabela
            consolidado_formatado = leads_consolidado.copy()
            colunas_monetarias = ['Investimento', 'CPL', 'Realizado', 'CAC', 'Receita', 'TM', 'LTV']
            for col in colunas_monetarias:
                if col in consolidado_formatado.columns:
                    consolidado_formatado[col] = consolidado_formatado[col].apply(lambda x: f"R$ {x:,.2f}" if x > 0 else "R$ 0.00")
            
            if 'Tx.Conv' in consolidado_formatado.columns:
                consolidado_formatado['Tx.Conv'] = consolidado_formatado['Tx.Conv'].apply(lambda x: f"{x:.1f}%" if x > 0 else "0.0%")
            if 'ROAS' in consolidado_formatado.columns:
                consolidado_formatado['ROAS'] = consolidado_formatado['ROAS'].apply(lambda x: f"{x:.1f}%" if x > 0 else "0.0%")
            if 'CAC/LTV' in consolidado_formatado.columns:
                consolidado_formatado['CAC/LTV'] = consolidado_formatado['CAC/LTV'].apply(lambda x: f"{x:.2f}" if x > 0 else "0.00")
            
            st.dataframe(consolidado_formatado, use_container_width=True)
            
            # Gráficos consolidados COM RÓTULOS
            col1, col2 = st.columns(2)
            with col1:
                fig_consolidado_leads = px.bar(leads_consolidado, x='Mês', y='Leads', title="Leads por Mês (E-MAILS Únicos)")
                fig_consolidado_leads.update_traces(
                    texttemplate='%{y}',
                    textposition='outside',
                    textfont=dict(size=10, color=COLORS['text_primary'])
                )
                fig_consolidado_leads = configurar_layout_clean(fig_consolidado_leads, "Leads por Mês", show_labels=True)
                render_plotly_chart(fig_consolidado_leads)
            
            with col2:
                fig_consolidado_roas = px.line(leads_consolidado, x='Mês', y='ROAS', title="ROAS por Mês", markers=True)
                fig_consolidado_roas.update_traces(
                    mode='lines+markers+text',
                    texttemplate='%{y:.1f}%',
                    textposition='top center',
                    textfont=dict(size=10, color=COLORS['text_primary'])
                )
                fig_consolidado_roas = configurar_layout_clean(fig_consolidado_roas, "ROAS por Mês", show_labels=True)
                render_plotly_chart(fig_consolidado_roas)
        else:
            st.warning("Não há dados consolidados disponíveis para o período selecionado")
        
        st.markdown("---")
        
        # POR LP
        st.subheader("POR LP")
        if not leads_por_lp.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1: 
                st.metric("Melhor LP", leads_por_lp.iloc[0]['LP'])
            with col2: 
                st.metric("Receita Top LP", f"R$ {leads_por_lp.iloc[0]['Receita']:,.0f}")
            with col3: 
                st.metric("ROAS Top LP", f"{leads_por_lp.iloc[0]['ROAS']:.1f}%")
            with col4: 
                st.metric("CAC/LTV Top LP", f"{leads_por_lp.iloc[0]['CAC/LTV']:.2f}")
            
            # Formatar tabela
            por_lp_formatado = leads_por_lp.copy()
            colunas_monetarias = ['Investimento', 'CPL', 'Realizado', 'CAC', 'Receita', 'TM', 'LTV']
            for col in colunas_monetarias:
                if col in por_lp_formatado.columns:
                    por_lp_formatado[col] = por_lp_formatado[col].apply(lambda x: f"R$ {x:,.2f}" if x > 0 else "R$ 0.00")
            
            if 'Tx.Conv' in por_lp_formatado.columns:
                por_lp_formatado['Tx.Conv'] = por_lp_formatado['Tx.Conv'].apply(lambda x: f"{x:.1f}%" if x > 0 else "0.0%")
            if 'ROAS' in por_lp_formatado.columns:
                por_lp_formatado['ROAS'] = por_lp_formatado['ROAS'].apply(lambda x: f"{x:.1f}%" if x > 0 else "0.0%")
            if 'CAC/LTV' in por_lp_formatado.columns:
                por_lp_formatado['CAC/LTV'] = por_lp_formatado['CAC/LTV'].apply(lambda x: f"{x:.2f}" if x > 0 else "0.00")
            
            st.dataframe(por_lp_formatado, use_container_width=True)
            
            # Gráficos COM RÓTULOS
            col1, col2 = st.columns(2)
            with col1:
                fig_leads_lp = px.bar(leads_por_lp.head(10), x='LP', y='Leads', title="Top 10 LPs por Volume de Leads")
                fig_leads_lp.update_traces(
                    texttemplate='%{y}',
                    textposition='outside',
                    textfont=dict(size=10, color=COLORS['text_primary'])
                )
                fig_leads_lp = configurar_layout_clean(fig_leads_lp, "Top 10 LPs por Volume de Leads", show_labels=True)
                render_plotly_chart(fig_leads_lp)
            with col2:
                fig_receita_lp = px.bar(leads_por_lp.head(10), x='LP', y='Receita', title="Top 10 LPs por Receita")
                fig_receita_lp.update_traces(
                    texttemplate='%{y:,.0f}',
                    textposition='outside',
                    textfont=dict(size=10, color=COLORS['text_primary'])
                )
                fig_receita_lp = configurar_layout_clean(fig_receita_lp, "Top 10 LPs por Receita", show_labels=True)
                render_plotly_chart(fig_receita_lp)
        else:
            st.warning("Não há dados por LP disponíveis para o período selecionado")
        
        st.markdown("---")
        
        # POR LP MENSAL
        st.subheader("POR LP MENSAL")
        if not leads_por_lp_mensal.empty:
            lps_disponiveis = leads_por_lp_mensal['LP'].unique()
            lp_selecionada = st.selectbox("Selecione a LP para análise detalhada:", lps_disponiveis)
            
            if lp_selecionada:
                dados_lp = leads_por_lp_mensal[leads_por_lp_mensal['LP'] == lp_selecionada]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1: 
                    st.metric("Total Leads", f"{dados_lp['Leads'].sum():,}")
                with col2: 
                    st.metric("Receita Total", f"R$ {dados_lp['Receita'].sum():,.0f}")
                with col3: 
                    st.metric("ROAS Médio", f"{dados_lp['ROAS'].mean():.1f}%")
                with col4: 
                    st.metric("CAC/LTV Médio", f"{dados_lp['CAC/LTV'].mean():.2f}")
                
                # Formatar tabela
                mensal_formatado = dados_lp.copy()
                colunas_monetarias = ['Investimento', 'CPL', 'Realizado', 'CAC', 'Receita', 'TM', 'LTV']
                for col in colunas_monetarias:
                    if col in mensal_formatado.columns:
                        mensal_formatado[col] = mensal_formatado[col].apply(lambda x: f"R$ {x:,.2f}" if x > 0 else "R$ 0.00")
                
                if 'Tx.Conv' in mensal_formatado.columns:
                    mensal_formatado['Tx.Conv'] = mensal_formatado['Tx.Conv'].apply(lambda x: f"{x:.1f}%" if x > 0 else "0.0%")
                if 'ROAS' in mensal_formatado.columns:
                    mensal_formatado['ROAS'] = mensal_formatado['ROAS'].apply(lambda x: f"{x:.1f}%" if x > 0 else "0.0%")
                if 'CAC/LTV' in mensal_formatado.columns:
                    mensal_formatado['CAC/LTV'] = mensal_formatado['CAC/LTV'].apply(lambda x: f"{x:.2f}" if x > 0 else "0.00")
                
                st.dataframe(mensal_formatado, use_container_width=True)
                
                # Gráficos mensais COM RÓTULOS
                col1, col2 = st.columns(2)
                with col1:
                    fig_evolucao_leads = px.line(dados_lp, x='Mês', y='Leads', title=f"Evolução de Leads - {lp_selecionada}", markers=True)
                    fig_evolucao_leads.update_traces(
                        mode='lines+markers+text',
                        texttemplate='%{y}',
                        textposition='top center',
                        textfont=dict(size=10, color=COLORS['text_primary'])
                    )
                    fig_evolucao_leads = configurar_layout_clean(fig_evolucao_leads, f"Evolução de Leads - {lp_selecionada}", show_labels=True)
                    render_plotly_chart(fig_evolucao_leads)
                with col2:
                    fig_evolucao_receita = px.line(dados_lp, x='Mês', y='Receita', title=f"Evolução de Receita - {lp_selecionada}", markers=True)
                    fig_evolucao_receita.update_traces(
                        mode='lines+markers+text',
                        texttemplate='%{y:,.0f}',
                        textposition='top center',
                        textfont=dict(size=10, color=COLORS['text_primary'])
                    )
                    fig_evolucao_receita = configurar_layout_clean(fig_evolucao_receita, f"Evolução de Receita - {lp_selecionada}", show_labels=True)
                    render_plotly_chart(fig_evolucao_receita)
        else:
            st.warning("Não há dados mensais por LP disponíveis para o período selecionado")
    
    with tab4:
        st.markdown(f'<div class="section-header"><h2 class="section-title">Análise Preditiva e Insights Avançados</h2><p class="section-subtitle">Previsões e análises estratégicas baseadas em machine learning</p></div>', unsafe_allow_html=True)
        
        # ANÁLISE PREDITIVA
        st.subheader("Previsão de Receita")
        
        if df_previsoes is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Gráfico de previsão COM RÓTULOS
                fig_previsao = go.Figure()
                
                # Adicionar histórico
                if dados_historicos is not None:
                    fig_previsao.add_trace(go.Scatter(
                        x=dados_historicos['Mês'],
                        y=dados_historicos['Receita Bruta'],
                        mode='lines+markers',
                        name='Histórico',
                        line=dict(color=COLORS['primary'], width=3),
                        marker=dict(size=8)
                    ))
                
                # Adicionar previsões
                fig_previsao.add_trace(go.Scatter(
                    x=df_previsoes['Mês'],
                    y=df_previsoes['Receita Bruta Prevista'],
                    mode='lines+markers+text',
                    name='Previsão',
                    line=dict(color=COLORS['warning'], width=3, dash='dash'),
                    marker=dict(size=8),
                    text=[f'R$ {x:,.0f}' for x in df_previsoes['Receita Bruta Prevista']],
                    textposition='top center'
                ))
                
                # Adicionar intervalo de confiança
                fig_previsao.add_trace(go.Scatter(
                    x=df_previsoes['Mês'].tolist() + df_previsoes['Mês'].tolist()[::-1],
                    y=df_previsoes['IC Superior'].tolist() + df_previsoes['IC Inferior'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 152, 0, 0.2)',
                    line=dict(color='rgba(255, 152, 0, 0)'),
                    name='Intervalo Confiança 95%'
                ))
                
                fig_previsao = configurar_layout_clean(fig_previsao, "Previsão de Receita - Próximos 6 Meses", show_labels=True)
                render_plotly_chart(fig_previsao)
            
            with col2:
                st.metric("Precisão Média dos Modelos", 
                         f"{(sum([resultados_modelos[modelo]['r2'] for modelo in resultados_modelos]) / len(resultados_modelos)) * 100:.1f}%")
                st.metric("Erro Médio de Previsão", f"R$ {desvio_padrao:,.0f}")
                
                st.markdown("""
                **Modelos Utilizados:**
                - Regressão Linear
                - Random Forest
                - Média Combinada
                """)
                
                # Métricas dos modelos individuais
                for nome, resultado in resultados_modelos.items():
                    with st.expander(f"Métricas {nome}"):
                        st.metric("R²", f"{resultado['r2']:.3f}")
                        st.metric("MAE", f"R$ {resultado['mae']:,.0f}")
        else:
            st.warning("Não há dados suficientes para gerar previsões. São necessários pelo menos 3 meses de dados históricos.")
        
        # ANÁLISE DE TENDÊNCIA
        st.subheader("Análise de Tendência")
        
        if analise_tendencia:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Crescimento Total", f"{analise_tendencia['crescimento_total']:.1f}%")
            
            with col2:
                st.metric("Volatilidade", f"{analise_tendencia['volatilidade']:.1f}%")
            
            with col3:
                st.metric("Correlação Sazonal", f"{analise_tendencia['correlacao_sazonal']:.2f}")
            
            with col4:
                st.metric("Média Móvel 3M", f"R$ {analise_tendencia['media_3m']:,.0f}")
            
            # Gráfico de tendência COM RÓTULOS
            fig_tendencia = go.Figure()
            
            fig_tendencia.add_trace(go.Scatter(
                x=analise_tendencia['dados']['Mês'],
                y=analise_tendencia['dados']['Receita Bruta'],
                mode='lines+markers',
                name='Receita Bruta',
                line=dict(color=COLORS['primary'], width=3)
            ))
            
            if 'Media_Movel_3M' in analise_tendencia['dados'].columns:
                fig_tendencia.add_trace(go.Scatter(
                    x=analise_tendencia['dados']['Mês'],
                    y=analise_tendencia['dados']['Media_Movel_3M'],
                    mode='lines',
                    name='Média Móvel 3M',
                    line=dict(color=COLORS['warning'], width=2, dash='dash')
                ))
            
            fig_tendencia = configurar_layout_clean(fig_tendencia, "Análise de Tendência e Sazonalidade", show_labels=True)
            render_plotly_chart(fig_tendencia)
        
        # INSIGHTS ESTRATÉGICOS
        st.subheader("Insights Estratégicos")
        
        if kpis_avancados:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="matriz-stats">
                    <h4>Performance Financeira</h4>
                    <ul>
                        <li><strong>Eficiência Marketing:</strong> R$ {kpis_avancados.get('receita_liquida_total', 0):,.0f} / R$ {kpis_avancados.get('investimento_total', 0):,.0f} = {kpis_avancados.get('eficiencia_marketing', 0):.2f}</li>
                        <li><strong>ROI Médio:</strong> {kpis_avancados.get('roi_medio', 0):.1f}%</li>
                        <li><strong>LTV/CAC Ratio:</strong> {kpis_avancados.get('ltv_cac_ratio', 0):.2f}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="matriz-stats">
                    <h4>Recomendações</h4>
                    <ul>
                        <li>{'✅ Manter estratégia atual' if kpis_avancados.get('ltv_cac_ratio', 0) > 1 else '⚠️ Otimizar aquisição'}</li>
                        <li>{'📈 Expandir investimento' if kpis_avancados.get('roi_medio', 0) > 100 else '🎯 Focar em eficiência'}</li>
                        <li>{'🔍 Analisar sazonalidade' if abs(analise_tendencia['correlacao_sazonal']) > 0.5 else '📊 Manter monitoramento'}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # ALERTAS E OPORTUNIDADES
        st.subheader("Alertas e Oportunidades")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if kpis_avancados and kpis_avancados.get('volatilidade_receita', 0) > 30:
                st.error("**ALERTA:** Alta volatilidade na receita. Recomenda-se investigar causas.")
            elif kpis_avancados:
                st.success("**ESTÁVEL:** Baixa volatilidade na receita.")
            
            if kpis_avancados and kpis_avancados.get('crescimento_receita', 0) < 0:
                st.warning("**ATENÇÃO:** Receita em declínio. Avaliar estratégias.")
            elif kpis_avancados and kpis_avancados.get('crescimento_receita', 0) > 20:
                st.success("**CRESCIMENTO:** Receita em forte expansão.")
        
        with col2:
            if kpis_avancados and kpis_avancados.get('ltv_cac_ratio', 0) < 1:
                st.error("**CRÍTICO:** LTV menor que CAC. Revisar urgentemente estratégia de aquisição.")
            elif kpis_avancados and kpis_avancados.get('ltv_cac_ratio', 0) > 3:
                st.success("**EXCELENTE:** LTV significativamente maior que CAC. Pode-se aumentar investimento.")
    
    with tab5:  # NOVA ABA PARA ANÁLISES AVANÇADAS
        st.markdown(f'<div class="section-header"><h2 class="section-title">Análises Avançadas e Segmentação</h2><p class="section-subtitle">Insights profundos com machine learning e estatística</p></div>', unsafe_allow_html=True)
        
        # ANÁLISE DE SAZONALIDADE
        if analise_sazonal and analise_sazonal_data:
            st.subheader("Análise de Sazonalidade e Tendência")
            sazonalidade_chart = criar_analise_sazonalidade_grafico(analise_sazonal_data)
            if sazonalidade_chart:
                render_plotly_chart(sazonalidade_chart)
                
                # Insights da sazonalidade
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Período Sazonal Identificado", 
                             f"{analise_sazonal_data['periodo_sazonal']} meses")
                with col2:
                    autocorr_max = max(analise_sazonal_data['autocorr']) if analise_sazonal_data['autocorr'] else 0
                    st.metric("Força da Sazonalidade", f"{autocorr_max:.2f}")
                with col3:
                    st.metric("Recomendação", 
                             "Ajustar Campanhas" if analise_sazonal_data['periodo_sazonal'] > 0 else "Padrão Estável")
        
        # ANÁLISE DE IMPACTO DO INVESTIMENTO
        if analise_impacto and analise_impacto_data:
            st.subheader("Análise de Correlação Investimento-Resultados")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Correlação Investimento-Receita", 
                         f"{analise_impacto_data['correlacao_receita']:.3f}")
            with col2:
                st.metric("Correlação Investimento-Tutores", 
                         f"{analise_impacto_data['correlacao_tutores']:.3f}")
            with col3:
                st.metric("ROI Médio do Período", 
                         f"{analise_impacto_data['roi_medio']:.1f}%")
            
            # Gráfico de correlação
            fig_correlacao = px.scatter(
                analise_impacto_data['dados'],
                x='Investimento',
                y='Receita Bruta',
                size='Novos Tutores',
                color='Receita Líquida',
                hover_data=['Mês'],
                title="Relação Investimento vs Receita",
                template='plotly_white' if TEMA_ATUAL == 'light' else 'plotly_dark'
            )
            render_plotly_chart(fig_correlacao)
        
        # ANÁLISE DE CLUSTERS
        if analise_clusters and analise_clusters_data:
            st.subheader("Segmentação por Clusters de Performance")
            
            # Mostrar análise dos clusters
            st.dataframe(analise_clusters_data['analise_clusters'], use_container_width=True)
            
            # Gráfico de clusters
            fig_clusters = px.scatter_3d(
                analise_clusters_data['dados'],
                x='Receita Bruta',
                y='Novos Tutores',
                z='Investimento',
                color='Cluster',
                size='Eficiencia',
                hover_data=['Mês'],
                title="Segmentação por Clusters - Análise 3D",
                template='plotly_white' if TEMA_ATUAL == 'light' else 'plotly_dark'
            )
            render_plotly_chart(fig_clusters)
            
            # Interpretação dos clusters
            st.subheader("Interpretação dos Clusters")
            cluster_analysis = analise_clusters_data['analise_clusters']
            
            for cluster_id in cluster_analysis.index:
                with st.expander(f"Cluster {cluster_id} - Características"):
                    cluster_data = cluster_analysis.loc[cluster_id]
                    st.write(f"**Receita Média:** R$ {cluster_data['Receita Bruta']:,.0f}")
                    st.write(f"**Novos Tutores:** {cluster_data['Novos Tutores']:.0f}")
                    st.write(f"**Eficiência:** {cluster_data['Eficiencia']:.2f}")
                    st.write(f"**CAC:** R$ {cluster_data['CAC']:,.0f}")
                    
                    # Recomendações baseadas no cluster
                    if cluster_data['Eficiencia'] > 1.5:
                        st.success("**Alta Eficiência** - Manter e expandir estratégia")
                    elif cluster_data['Eficiencia'] < 0.8:
                        st.warning("**Baixa Eficiência** - Revisar estratégia de aquisição")
        
        # ANÁLISE DE SENSIBILIDADE
        st.subheader("Análise de Sensibilidade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Simulador de cenários
            st.markdown("#### Simulador de Cenários")
            investimento_base = st.number_input("Investimento Base (R$)", 
                                              value=100000, step=5000)
            variacao_ticket = st.slider("Variação no Ticket Médio (%)", -20, 20, 0)
            variacao_conversao = st.slider("Variação na Taxa de Conversão (%)", -20, 20, 0)
            
            # Cálculo simplificado do impacto
            receita_estimada = investimento_base * (1 + variacao_ticket/100) * (1 + variacao_conversao/100)
            st.metric("Receita Estimada", f"R$ {receita_estimada:,.0f}")
        
        with col2:
            # Análise de break-even
            st.markdown("#### Análise de Break-Even")
            custo_fixo = st.number_input("Custo Fixo Mensal (R$)", value=50000, step=5000)
            ticket_medio = st.number_input("Ticket Médio (R$)", value=1500, step=100)
            custo_variavel = st.number_input("Custo Variável por Tutor (R$)", value=500, step=50)
            
            if ticket_medio > custo_variavel:
                break_even = custo_fixo / (ticket_medio - custo_variavel)
                st.metric("Tutores para Break-Even", f"{break_even:.0f}")
            else:
                st.error("Ticket médio insuficiente para cobrir custos variáveis")
        
        # NOVO: Gráfico Radar de Performance
        if not metricas_avancadas_lp.empty:
            st.subheader("Análise Comparativa de LPs - Radar")
            radar_chart = criar_grafico_radar_performance(metricas_avancadas_lp)
            if radar_chart:
                render_plotly_chart(radar_chart)

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
