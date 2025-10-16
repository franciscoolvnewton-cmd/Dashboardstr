import streamlit as st
import pandas as pd
import numpy as np
import sys
import subprocess
import os
from PIL import Image
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Verificar e instalar dependências missing
def install_missing_dependencies():
    missing_deps = []
    try:
        import openpyxl
    except ImportError:
        missing_deps.append('openpyxl')
    
    try:
        import plotly
    except ImportError:
        missing_deps.append('plotly')
    
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        missing_deps.append('scikit-learn')
    
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        missing_deps.append('scikit-learn')
    
    if missing_deps:
        st.error(f"📦 Dependências missing: {', '.join(missing_deps)}")
        st.info("🔧 Instalando dependências automaticamente...")
        
        for dep in missing_deps:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                st.success(f"✅ {dep} instalado com sucesso!")
            except subprocess.CalledProcessError:
                st.error(f"❌ Falha ao instalar {dep}")
                st.stop()
        
        st.rerun()

# Instalar dependências se necessário
install_missing_dependencies()

# Agora importe as bibliotecas após a instalação
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Dashboard Intelligence - Veros",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de cores harmoniosa
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
    'white': '#ffffff'
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

# Função para carregar o logo
@st.cache_data
def load_logo():
    logo_path = r"C:\Users\USER\Downloads\images.jpg"
    try:
        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            return logo
        else:
            st.warning("⚠️ Logo não encontrado no caminho especificado")
            return None
    except Exception as e:
        st.warning(f"⚠️ Erro ao carregar logo: {e}")
        return None

# Função para carregar e processar os dados
@st.cache_data
def load_data():
    file_path = r"C:\Users\USER\Documents\Veros_DataHub\DADOS_RECEITA_VEROS.xlsx"
    
    try:
        # Carregar os dados
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Verificar se as colunas necessárias existem
        colunas_necessarias = ['Mês geração receita', 'Mês geração lead', 'Considerar?', 'LP', 'VL UNI', 'E-MAIL']
        colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
        
        if colunas_faltantes:
            st.warning(f"⚠️ Colunas faltantes no dataset: {', '.join(colunas_faltantes)}")
            st.info("📋 Colunas disponíveis no seu arquivo:")
            st.write(list(df.columns))
        
        # Converter colunas de data
        date_columns = ['DT Receita', 'Data do lead', 'Data e-mail', 'Data e-mail corrigido', 'Data telefone']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Converter valor unitário para numérico
        if 'VL UNI' in df.columns:
            df['VL UNI'] = pd.to_numeric(df['VL UNI'], errors='coerce')
        
        return df
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar o arquivo: {e}")
        st.info("💡 Solução: Execute no terminal: pip install openpyxl")
        return None

# Função para criar matriz escadinha de Receita - CORRIGIDA
def criar_matriz_escadinha(df, ano_filtro=2025):
    """
    Cria uma matriz escadinha mostrando a receita bruta em função da diferença
    entre mês de geração da receita e mês de geração do lead
    """
    if df is None or df.empty:
        return None, None
    
    # Criar cópia para evitar warnings
    df_filtrado = df.copy()
    
    # Aplicar condições de filtro
    try:
        # Condição 1: Considerar = Sim
        condicao1 = df_filtrado['Considerar?'] == 'Sim'
        
        # Condição 2: LP válido
        condicao2 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        # Condição 3: Filtrar por ano - CORREÇÃO APLICADA
        if ano_filtro == 2024:
            # Para 2024, filtrar apenas dados de 2024
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
            condicao4 = df_filtrado['Mês geração lead'].str.contains('2024|24', na=False)
        else:  # 2025
            # Para 2025, filtrar apenas dados de 2025
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
            condicao4 = df_filtrado['Mês geração lead'].str.contains('2025|25', na=False)
        
        # Aplicar filtros
        mask = condicao1 & condicao2 & condicao3 & condicao4
        df_filtrado = df_filtrado[mask].copy()
        
        if df_filtrado.empty:
            return None, None
        
        # Definir ordem dos meses apenas para o ano selecionado
        if ano_filtro == 2024:
            ordem_meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                          'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
        else:
            ordem_meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                          'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
        
        # Filtrar apenas meses que estão na ordem definida
        df_filtrado = df_filtrado[
            df_filtrado['Mês geração receita'].isin(ordem_meses) & 
            df_filtrado['Mês geração lead'].isin(ordem_meses)
        ].copy()
        
        if df_filtrado.empty:
            return None, None
        
        # Criar matriz de receita apenas com meses do ano selecionado
        matriz_receita = pd.DataFrame(0, index=ordem_meses, columns=ordem_meses, dtype=float)
        
        # Preencher a matriz com os valores de receita
        for _, row in df_filtrado.iterrows():
            mes_receita = row['Mês geração receita']
            mes_lead = row['Mês geração lead']
            valor = row['VL UNI']
            
            if mes_receita in ordem_meses and mes_lead in ordem_meses:
                matriz_receita.loc[mes_receita, mes_lead] += valor
        
        # Criar matriz formatada para exibição
        matriz_formatada = matriz_receita.copy()
        for col in matriz_formatada.columns:
            matriz_formatada[col] = matriz_formatada[col].apply(lambda x: f"R$ {x:,.0f}" if x > 0 else "-")
        
        return matriz_receita, matriz_formatada
    
    except Exception as e:
        st.error(f"Erro ao criar matriz escadinha: {e}")
        return None, None

# Função para criar matriz escadinha de CAC - CORRIGIDA
def criar_matriz_cac(df, ano_filtro=2025):
    """
    Cria uma matriz escadinha mostrando o CAC (Custo de Aquisição por Cliente)
    em função da diferença entre mês de geração da receita e mês de geração do lead
    """
    if df is None or df.empty:
        return None, None
    
    # Criar cópia para evitar warnings
    df_filtrado = df.copy()
    
    # Aplicar condições de filtro
    try:
        # Condição 1: Considerar = Sim
        condicao1 = df_filtrado['Considerar?'] == 'Sim'
        
        # Condição 2: LP válido
        condicao2 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        # Condição 3: Filtrar por ano - CORREÇÃO APLICADA
        if ano_filtro == 2024:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
            condicao4 = df_filtrado['Mês geração lead'].str.contains('2024|24', na=False)
            investimento_mensal = INVESTIMENTO_MENSAL_2024
        else:  # 2025
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
            condicao4 = df_filtrado['Mês geração lead'].str.contains('2025|25', na=False)
            investimento_mensal = INVESTIMENTO_MENSAL_2025
        
        # Aplicar filtros
        mask = condicao1 & condicao2 & condicao3 & condicao4
        df_filtrado = df_filtrado[mask].copy()
        
        if df_filtrado.empty:
            return None, None
        
        # Definir ordem dos meses apenas para o ano selecionado
        if ano_filtro == 2024:
            ordem_meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                          'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
        else:
            ordem_meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                          'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
        
        # Filtrar apenas meses que estão na ordem definida
        df_filtrado = df_filtrado[
            df_filtrado['Mês geração receita'].isin(ordem_meses) & 
            df_filtrado['Mês geração lead'].isin(ordem_meses)
        ].copy()
        
        if df_filtrado.empty:
            return None, None
        
        # Criar matriz de novos tutores apenas com meses do ano selecionado
        matriz_tutores = pd.DataFrame(0, index=ordem_meses, columns=ordem_meses, dtype=float)
        
        # Preencher a matriz com contagem de novos tutores únicos
        tutores_por_combinacao = df_filtrado.groupby(['Mês geração receita', 'Mês geração lead'])['E-MAIL'].nunique()
        
        for (mes_receita, mes_lead), count in tutores_por_combinacao.items():
            if mes_receita in ordem_meses and mes_lead in ordem_meses:
                matriz_tutores.loc[mes_receita, mes_lead] = count
        
        # Criar matriz de CAC apenas com meses do ano selecionado
        matriz_cac = pd.DataFrame(0, index=ordem_meses, columns=ordem_meses, dtype=float)
        
        # Calcular CAC: Investimento do mês do lead / número de tutores
        for mes_lead in ordem_meses:
            if mes_lead in investimento_mensal:
                investimento = investimento_mensal[mes_lead]
                for mes_receita in ordem_meses:
                    num_tutores = matriz_tutores.loc[mes_receita, mes_lead]
                    if num_tutores > 0:
                        matriz_cac.loc[mes_receita, mes_lead] = investimento / num_tutores
        
        # Criar matriz formatada para exibição
        matriz_formatada = matriz_cac.copy()
        for col in matriz_formatada.columns:
            matriz_formatada[col] = matriz_formatada[col].apply(
                lambda x: f"R$ {x:,.0f}" if x > 0 else "-"
            )
        
        return matriz_cac, matriz_formatada
    
    except Exception as e:
        st.error(f"Erro ao criar matriz CAC: {e}")
        return None, None

# Função para criar matriz escadinha de LTV - CORRIGIDA
def criar_matriz_ltv(df, ano_filtro=2025):
    """
    Cria uma matriz escadinha mostrando o LTV (Lifetime Value)
    em função da diferença entre mês de geração da receita e mês de geração do lead
    """
    if df is None or df.empty:
        return None, None
    
    # Criar cópia para evitar warnings
    df_filtrado = df.copy()
    
    # Aplicar condições de filtro
    try:
        # Condição 1: Considerar = Sim
        condicao1 = df_filtrado['Considerar?'] == 'Sim'
        
        # Condição 2: LP válido
        condicao2 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        # Condição 3: Filtrar por ano - CORREÇÃO APLICADA
        if ano_filtro == 2024:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
            condicao4 = df_filtrado['Mês geração lead'].str.contains('2024|24', na=False)
        else:  # 2025
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
            condicao4 = df_filtrado['Mês geração lead'].str.contains('2025|25', na=False)
        
        # Aplicar filtros
        mask = condicao1 & condicao2 & condicao3 & condicao4
        df_filtrado = df_filtrado[mask].copy()
        
        if df_filtrado.empty:
            return None, None
        
        # Definir ordem dos meses apenas para o ano selecionado
        if ano_filtro == 2024:
            ordem_meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                          'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
        else:
            ordem_meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                          'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
        
        # Filtrar apenas meses que estão na ordem definida
        df_filtrado = df_filtrado[
            df_filtrado['Mês geração receita'].isin(ordem_meses) & 
            df_filtrado['Mês geração lead'].isin(ordem_meses)
        ].copy()
        
        if df_filtrado.empty:
            return None, None
        
        # Criar matriz de receita líquida (56% da bruta) apenas com meses do ano selecionado
        matriz_receita_bruta = pd.DataFrame(0, index=ordem_meses, columns=ordem_meses, dtype=float)
        matriz_tutores = pd.DataFrame(0, index=ordem_meses, columns=ordem_meses, dtype=float)
        
        # Preencher as matrizes
        for _, row in df_filtrado.iterrows():
            mes_receita = row['Mês geração receita']
            mes_lead = row['Mês geração lead']
            valor = row['VL UNI']
            
            if mes_receita in ordem_meses and mes_lead in ordem_meses:
                matriz_receita_bruta.loc[mes_receita, mes_lead] += valor
        
        # Calcular receita líquida (56% da bruta)
        matriz_receita_liquida = matriz_receita_bruta * 0.56
        
        # Calcular número de tutores únicos por combinação
        tutores_por_combinacao = df_filtrado.groupby(['Mês geração receita', 'Mês geração lead'])['E-MAIL'].nunique()
        
        for (mes_receita, mes_lead), count in tutores_por_combinacao.items():
            if mes_receita in ordem_meses and mes_lead in ordem_meses:
                matriz_tutores.loc[mes_receita, mes_lead] = count
        
        # Criar matriz de LTV: Receita líquida / número de tutores
        matriz_ltv = pd.DataFrame(0, index=ordem_meses, columns=ordem_meses, dtype=float)
        
        for mes_receita in ordem_meses:
            for mes_lead in ordem_meses:
                num_tutores = matriz_tutores.loc[mes_receita, mes_lead]
                receita_liquida = matriz_receita_liquida.loc[mes_receita, mes_lead]
                if num_tutores > 0:
                    matriz_ltv.loc[mes_receita, mes_lead] = receita_liquida / num_tutores
        
        # Criar matriz formatada para exibição
        matriz_formatada = matriz_ltv.copy()
        for col in matriz_formatada.columns:
            matriz_formatada[col] = matriz_formatada[col].apply(
                lambda x: f"R$ {x:,.0f}" if x > 0 else "-"
            )
        
        return matriz_ltv, matriz_formatada
    
    except Exception as e:
        st.error(f"Erro ao criar matriz LTV: {e}")
        return None, None

# Função para criar matriz escadinha de CAC/LTV - CORRIGIDA
def criar_matriz_cac_ltv_ratio(df, ano_filtro=2025):
    """
    Cria uma matriz escadinha mostrando a razão CAC/LTV
    em função da diferença entre mês de geração da receita e mês de geração do lead
    """
    if df is None or df.empty:
        return None, None
    
    try:
        # Obter matrizes de CAC e LTV - CORREÇÃO: Passar o ano_filtro
        matriz_cac, _ = criar_matriz_cac(df, ano_filtro)
        matriz_ltv, _ = criar_matriz_ltv(df, ano_filtro)
        
        if matriz_cac is None or matriz_ltv is None:
            return None, None
        
        # Calcular razão CAC/LTV
        matriz_ratio = pd.DataFrame(0, index=matriz_cac.index, columns=matriz_cac.columns, dtype=float)
        
        for i in matriz_cac.index:
            for j in matriz_cac.columns:
                cac_val = matriz_cac.loc[i, j]
                ltv_val = matriz_ltv.loc[i, j]
                if cac_val > 0 and ltv_val > 0:
                    matriz_ratio.loc[i, j] = cac_val / ltv_val
        
        # Criar matriz formatada para exibição
        matriz_formatada = matriz_ratio.copy()
        for col in matriz_formatada.columns:
            matriz_formatada[col] = matriz_formatada[col].apply(
                lambda x: f"{x:.2f}" if x > 0 else "-"
            )
        
        return matriz_ratio, matriz_formatada
    
    except Exception as e:
        st.error(f"Erro ao criar matriz CAC/LTV: {e}")
        return None, None

# FUNÇÃO ATUALIZADA: Criar heatmap da matriz escadinha COM RÓTULOS
def criar_heatmap_matriz(matriz, titulo, colorscale='Blues', width=500, height=450):
    """
    Cria um heatmap visual da matriz escadinha COM RÓTULOS DE DADOS
    """
    if matriz is None or matriz.empty:
        return None
    
    # Criar cópia para não modificar a original
    matriz_plot = matriz.copy()
    
    # Formatar valores para exibição nos rótulos
    if "CAC/LTV" in titulo:
        # Para razão CAC/LTV, mostrar 2 casas decimais
        text_values = [[f"{val:.2f}" if val > 0 else "" for val in row] for row in matriz_plot.values]
    elif "Receita" in titulo:
        # Para receita, mostrar valores em milhares
        text_values = [[f"R${val/1000:.0f}K" if val > 0 else "" for val in row] for row in matriz_plot.values]
    else:
        # Para CAC e LTV, mostrar valores formatados
        text_values = [[f"R${val:,.0f}" if val > 0 else "" for val in row] for row in matriz_plot.values]
    
    # Criar heatmap COM RÓTULOS
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
        ),
        # ADICIONAR RÓTULOS DE DADOS
        text=text_values,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "black"}
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
            font=dict(size=14)
        ),
        xaxis_title="Mês de Geração do Lead",
        yaxis_title="Mês de Geração da Receita",
        width=width,
        height=height,
        font=dict(family="Arial, sans-serif", size=10),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=8)
        ),
        yaxis=dict(
            tickfont=dict(size=8)
        )
    )
    
    return fig

# Função para calcular estatísticas da matriz
def calcular_estatisticas_matriz(matriz, tipo="Receita"):
    """
    Calcula estatísticas importantes da matriz escadinha
    """
    if matriz is None or matriz.empty:
        return None
    
    stats = {}
    
    # Valor total
    stats['valor_total'] = matriz.sum().sum()
    
    # Valor na diagonal principal (meses iguais)
    diagonal_principal = np.diag(matriz)
    stats['valor_diagonal'] = diagonal_principal.sum()
    
    # Percentual na diagonal
    stats['percentual_diagonal'] = (stats['valor_diagonal'] / stats['valor_total'] * 100) if stats['valor_total'] > 0 else 0
    
    # Valor fora da diagonal
    stats['valor_fora_diagonal'] = stats['valor_total'] - stats['valor_diagonal']
    
    # Mês com maior valor na diagonal
    if len(diagonal_principal) > 0 and diagonal_principal.sum() > 0:
        mes_max_diagonal = matriz.index[np.argmax(diagonal_principal)]
        stats['mes_maior_diagonal'] = mes_max_diagonal
        stats['valor_maior_diagonal'] = diagonal_principal.max()
    else:
        stats['mes_maior_diagonal'] = "N/A"
        stats['valor_maior_diagonal'] = 0
    
    # Eficiência (razão entre valor diagonal e total)
    stats['eficiencia'] = stats['percentual_diagonal'] / 100
    
    # Estatísticas gerais para todas as matrizes
    stats['total_celulas'] = matriz.size
    stats['celulas_preenchidas'] = (matriz.values > 0).sum()
    stats['percentual_preenchidas'] = (stats['celulas_preenchidas'] / stats['total_celulas'] * 100) if stats['total_celulas'] > 0 else 0
    
    # Estatísticas específicas por tipo
    if tipo == "CAC/LTV":
        # Para CAC/LTV, valores abaixo de 1 são bons
        stats['celulas_saudaveis'] = (matriz.values < 1.0).sum()
        stats['percentual_saudavel'] = (stats['celulas_saudaveis'] / stats['total_celulas'] * 100) if stats['total_celulas'] > 0 else 0
        stats['celulas_problematicas'] = (matriz.values >= 1.0).sum()
        stats['percentual_problematico'] = (stats['celulas_problematicas'] / stats['total_celulas'] * 100) if stats['total_celulas'] > 0 else 0
    
    elif tipo == "CAC":
        # Para CAC, identificamos células com valores muito altos
        if stats['celulas_preenchidas'] > 0:
            cac_medio = stats['valor_total'] / stats['celulas_preenchidas']
            stats['celulas_acima_media'] = (matriz.values > cac_medio).sum()
            stats['percentual_acima_media'] = (stats['celulas_acima_media'] / stats['total_celulas'] * 100) if stats['total_celulas'] > 0 else 0
    
    elif tipo == "LTV":
        # Para LTV, identificamos células com valores muito baixos
        if stats['celulas_preenchidas'] > 0:
            ltv_medio = stats['valor_total'] / stats['celulas_preenchidas']
            stats['celulas_abaixo_media'] = (matriz.values < ltv_medio).sum()
            stats['percentual_abaixo_media'] = (stats['celulas_abaixo_media'] / stats['total_celulas'] * 100) if stats['total_celulas'] > 0 else 0
    
    return stats

# Função para aplicar a lógica de receita bruta mensal com filtro de data
def calcular_receita_mensal(df, ano_filtro=2025):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Criar cópia para evitar warnings
    df_filtrado = df.copy()
    
    # Aplicar condições de filtro
    try:
        # Condição 1: Meses iguais (Mês geração receita == Mês geração lead)
        condicao1 = df_filtrado['Mês geração receita'] == df_filtrado['Mês geração lead']
        
        # Condição 2: Considerar = Sim
        condicao2 = df_filtrado['Considerar?'] == 'Sim'
        
        # Condição 3: LP válido
        condicao3 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        # Condição 4: Filtrar por ano - CORREÇÃO APLICADA
        if ano_filtro == 2024:
            condicao4 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
        else:  # 2025
            condicao4 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
        
        # Aplicar filtros
        mask = condicao1 & condicao2 & condicao3 & condicao4
        df_filtrado = df_filtrado[mask].copy()
        
        # Calcular receita bruta mensal
        if not df_filtrado.empty and 'Mês geração receita' in df_filtrado.columns and 'VL UNI' in df_filtrado.columns:
            receita_mensal = df_filtrado.groupby('Mês geração receita')['VL UNI'].sum().reset_index()
            receita_mensal.columns = ['Mês', 'Receita Bruta']
            
            # Calcular receita líquida (56% da receita bruta)
            receita_mensal['Receita Líquida'] = receita_mensal['Receita Bruta'] * 0.56
            
            # Ordenar por mês de forma cronológica
            if ano_filtro == 2024:
                ordem_meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                              'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
            else:  # 2025
                ordem_meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                              'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
            
            # Manter apenas meses que existem na ordem definida
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
    
    # Criar cópia para evitar warnings
    df_filtrado = df.copy()
    
    # Aplicar condições de filtro para Novos Tutores
    try:
        # Condição 1: E-MAIL únicos
        # Condição 2: Meses iguais
        condicao1 = df_filtrado['Mês geração receita'] == df_filtrado['Mês geração lead']
        
        # Condição 3: Considerar = Sim
        condicao2 = df_filtrado['Considerar?'] == 'Sim'
        
        # Condição 4: LP válido
        condicao3 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        # Condição 5: Filtrar por ano - CORREÇÃO APLICADA
        if ano_filtro == 2024:
            condicao4 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
        else:  # 2025
            condicao4 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
        
        # Aplicar filtros
        mask = condicao1 & condicao2 & condicao3 & condicao4
        df_filtrado_tutores = df_filtrado[mask].copy()
        
        # Remover duplicados por E-MAIL (manter primeiro registro)
        df_tutores_unicos = df_filtrado_tutores.drop_duplicates(subset=['E-MAIL'], keep='first')
        
        # Calcular Novos Tutores por Mês
        if not df_tutores_unicos.empty and 'Mês geração receita' in df_tutores_unicos.columns:
            novos_tutores_mes = df_tutores_unicos.groupby('Mês geração receita').size().reset_index()
            novos_tutores_mes.columns = ['Mês', 'Novos Tutores']
            
            # Ordenar por mês de forma cronológica
            if ano_filtro == 2024:
                ordem_meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                              'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
            else:  # 2025
                ordem_meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                              'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
            
            # Manter apenas meses que existem na ordem definida
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
    
    # Criar DataFrame base com todos os meses do ano selecionado
    if ano_filtro == 2024:
        meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
        investimento_mensal = INVESTIMENTO_MENSAL_2024
    else:  # 2025
        meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
        investimento_mensal = INVESTIMENTO_MENSAL_2025
    
    cohort_data = pd.DataFrame({
        'Mês': meses
    })
    
    # Adicionar investimento apenas para meses que existem nos dados
    cohort_data['Investimento'] = cohort_data['Mês'].map(investimento_mensal).fillna(0)
    
    # Juntar com novos tutores
    cohort_data = cohort_data.merge(novos_tutores_mes, on='Mês', how='left')
    
    # Juntar com receita para calcular ticket médio líquido
    cohort_data = cohort_data.merge(receita_mensal[['Mês', 'Receita Líquida']], on='Mês', how='left')
    
    # Calcular métricas
    cohort_data['Novos Tutores'] = cohort_data['Novos Tutores'].fillna(0)
    cohort_data['Receita Líquida'] = cohort_data['Receita Líquida'].fillna(0)
    
    # CAC = Investimento / Novos Tutores
    cohort_data['CAC'] = cohort_data.apply(
        lambda x: x['Investimento'] / x['Novos Tutores'] if x['Novos Tutores'] > 0 else 0, 
        axis=1
    )
    
    # LTV (Ticket Médio Líquido) = Receita Líquida / Novos Tutores
    cohort_data['LTV'] = cohort_data.apply(
        lambda x: x['Receita Líquida'] / x['Novos Tutores'] if x['Novos Tutores'] > 0 else 0, 
        axis=1
    )
    
    # CAC/LTV = CAC / LTV
    cohort_data['CAC/LTV'] = cohort_data.apply(
        lambda x: x['CAC'] / x['LTV'] if x['LTV'] > 0 else 0, 
        axis=1
    )
    
    # ROI = (Receita Líquida - Investimento) / Investimento * 100
    cohort_data['ROI (%)'] = cohort_data.apply(
        lambda x: ((x['Receita Líquida'] - x['Investimento']) / x['Investimento']) * 100 if x['Investimento'] > 0 else 0, 
        axis=1
    )
    
    return cohort_data

# NOVA FUNÇÃO: Análise de Performance por LP
def analisar_performance_lp(df, ano_filtro=2025):
    """
    Analisa a performance de cada Landing Page (LP) com todas as métricas solicitadas
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Criar cópia para evitar warnings
    df_filtrado = df.copy()
    
    try:
        # Aplicar filtros básicos
        condicao1 = df_filtrado['Considerar?'] == 'Sim'
        condicao2 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        # Filtrar por ano
        if ano_filtro == 2024:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
        else:  # 2025
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
        
        mask = condicao1 & condicao2 & condicao3
        df_filtrado = df_filtrado[mask].copy()
        
        if df_filtrado.empty:
            return pd.DataFrame()
        
        # Lista de LPs para análise
        lps_para_analise = list(INVESTIMENTO_POR_LP.keys())
        
        # Criar DataFrame para resultados
        resultados = []
        
        for lp in lps_para_analise:
            # Filtrar dados para o LP específico
            df_lp = df_filtrado[df_filtrado['LP'] == lp].copy()
            
            if df_lp.empty:
                # Se não há dados, criar linha com zeros
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
            
            # Calcular métricas
            investimento = INVESTIMENTO_POR_LP.get(lp, 0)
            
            # Leads únicos (considerando E-MAIL como identificador único)
            leads_unicos = df_lp['E-MAIL'].nunique()
            
            # CPL (Custo por Lead)
            cpl = investimento / leads_unicos if leads_unicos > 0 else 0
            
            # Realizado (Receita Bruta)
            realizado = df_lp['VL UNI'].sum()
            
            # Taxa de Conversão (Receita/Leads)
            tx_conv = (realizado / investimento * 100) if investimento > 0 else 0
            
            # CAC (Custo de Aquisição por Cliente)
            # Considerando apenas tutores únicos que geraram receita
            tutores_unicos_com_receita = df_lp.drop_duplicates(subset=['E-MAIL'])['E-MAIL'].nunique()
            cac = investimento / tutores_unicos_com_receita if tutores_unicos_com_receita > 0 else 0
            
            # Receita Líquida (56% da receita bruta)
            receita_liquida = realizado * 0.56
            
            # ROAS (Return on Ad Spend)
            roas = (receita_liquida / investimento * 100) if investimento > 0 else 0
            
            # Ticket Médio (TM)
            tm = realizado / tutores_unicos_com_receita if tutores_unicos_com_receita > 0 else 0
            
            # LTV (considerando receita líquida)
            ltv = receita_liquida / tutores_unicos_com_receita if tutores_unicos_com_receita > 0 else 0
            
            # CAC/LTV
            cac_ltv_ratio = cac / ltv if ltv > 0 else 0
            
            # Receita Cohort (mesmo conceito de receita líquida)
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
        
        # Criar DataFrame final
        df_resultados = pd.DataFrame(resultados)
        
        # Ordenar por Receita (maior para menor)
        df_resultados = df_resultados.sort_values('Receita', ascending=False)
        
        return df_resultados
    
    except Exception as e:
        st.error(f"Erro na análise de LPs: {e}")
        return pd.DataFrame()

# NOVA FUNÇÃO: Performance Mensal por LP
def analisar_performance_mensal_lp(df, ano_filtro=2025):
    """
    Analisa a performance mensal de cada Landing Page (LP)
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Criar cópia para evitar warnings
    df_filtrado = df.copy()
    
    try:
        # Aplicar filtros básicos
        condicao1 = df_filtrado['Considerar?'] == 'Sim'
        condicao2 = (df_filtrado['LP'].notna()) & (df_filtrado['LP'] != '00. Not Mapped') & (df_filtrado['LP'] != '')
        
        # Filtrar por ano
        if ano_filtro == 2024:
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2024|24', na=False)
            meses = ['jan./24', 'fev./24', 'mar./24', 'abr./24', 'mai./24', 'jun./24', 
                    'jul./24', 'ago./24', 'set./24', 'out./24', 'nov./24', 'dez./24']
        else:  # 2025
            condicao3 = df_filtrado['Mês geração receita'].str.contains('2025|25', na=False)
            meses = ['jan./25', 'fev./25', 'mar./25', 'abr./25', 'mai./25', 'jun./25', 
                    'jul./25', 'ago./25', 'set./25', 'out./25', 'nov./25', 'dez./25']
        
        mask = condicao1 & condicao2 & condicao3
        df_filtrado = df_filtrado[mask].copy()
        
        if df_filtrado.empty:
            return pd.DataFrame()
        
        # Lista de LPs para análise
        lps_para_analise = list(INVESTIMENTO_POR_LP.keys())
        
        # Criar DataFrame para resultados mensais
        resultados_mensais = []
        
        for lp in lps_para_analise:
            for mes in meses:
                # Filtrar dados para o LP específico e mês específico
                df_lp_mes = df_filtrado[(df_filtrado['LP'] == lp) & (df_filtrado['Mês geração receita'] == mes)].copy()
                
                if df_lp_mes.empty:
                    # Se não há dados, criar linha com zeros
                    resultados_mensais.append({
                        'LP': lp,
                        'Mês': mes,
                        'Receita': 0,
                        'Leads': 0,
                        'ROAS': 0,
                        'CAC/LTV': 0
                    })
                    continue
                
                # Calcular métricas mensais
                investimento_mensal_lp = INVESTIMENTO_POR_LP.get(lp, 0) / 12  # Distribuição mensal proporcional
                
                # Leads únicos no mês
                leads_unicos_mes = df_lp_mes['E-MAIL'].nunique()
                
                # Receita no mês
                receita_mes = df_lp_mes['VL UNI'].sum()
                receita_liquida_mes = receita_mes * 0.56
                
                # ROAS mensal
                roas_mes = (receita_liquida_mes / investimento_mensal_lp * 100) if investimento_mensal_lp > 0 else 0
                
                # CAC/LTV mensal (simplificado)
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
        
        # Criar DataFrame final
        df_resultados_mensais = pd.DataFrame(resultados_mensais)
        
        return df_resultados_mensais
    
    except Exception as e:
        st.error(f"Erro na análise mensal de LPs: {e}")
        return pd.DataFrame()

# Configuração de layout limpo para gráficos COM RÓTULOS
def configurar_layout_clean(fig, titulo="", width=800, height=500):
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
            tickfont=dict(size=14, color=COLORS['gray']),
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
            tickfont=dict(size=14, color=COLORS['gray']),
            gridcolor=COLORS['light_gray'],
            linecolor=COLORS['light_gray'],
            showgrid=True,
            zeroline=False,
            tickformat=",."
        ),
        legend=dict(
            font=dict(size=14, color=COLORS['dark_gray']),
            bgcolor='rgba(255,255,255,0.9)',
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
    
    # Adicionar rótulos de dados em todos os traços
    for trace in fig.data:
        if hasattr(trace, 'y'):
            if isinstance(trace, go.Bar):
                # Para gráficos de barras
                trace.update(
                    texttemplate='%{y:,.0f}',
                    textposition='outside'
                )
            elif isinstance(trace, go.Scatter):
                # Para gráficos de linha
                trace.update(
                    texttemplate='%{y:,.0f}',
                    textposition='top center',
                    mode='lines+markers+text'
                )
    
    return fig

# Interface principal
def main():
    # Configuração CSS com design limpo e profissional
    st.markdown(f"""
    <style>
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    
    h1, h2, h3 {{
        color: {COLORS['dark_gray']} !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        font-family: 'Arial', sans-serif;
    }}
    
    .stMetric {{
        background-color: {COLORS['white']};
        border: 1px solid {COLORS['light_gray']};
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
        color: {COLORS['gray']} !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }}
    
    .stMetric div {{
        color: {COLORS['dark_gray']} !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
    }}
    
    /* Estilos para as abas - Design limpo e profissional */
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
        color: {COLORS['gray']};
        transition: all 0.3s ease;
        border: none;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(37, 99, 235, 0.1);
        color: {COLORS['primary']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']};
        color: white;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }}
    
    /* Estilos para a matriz escadinha */
    .matriz-container {{
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid {COLORS['light_gray']};
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
        background: white;
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
        color: {COLORS['gray']};
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metrica-valor {{
        font-size: 1.4rem;
        color: {COLORS['dark_gray']};
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    
    .metrica-descricao {{
        font-size: 0.7rem;
        color: {COLORS['gray']};
        font-style: italic;
    }}
    
    .section-header {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['primary_light']});
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0;
    }}
    
    .section-title {{
        color: white !important;
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
    
    @media (max-width: 1200px) {{
        .heatmap-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    
    .info-box {{
        background: {COLORS['info_light']}15;
        border: 1px solid {COLORS['info_light']};
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    
    /* Estilos para tabelas */
    .dataframe {{
        width: 100%;
        border-collapse: collapse;
    }}
    
    .dataframe th {{
        background-color: {COLORS['primary']};
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }}
    
    .dataframe td {{
        padding: 10px;
        border-bottom: 1px solid {COLORS['light_gray']};
    }}
    
    .dataframe tr:hover {{
        background-color: {COLORS['light_gray']};
    }}
    </style>
    """, unsafe_allow_html=True)

    # Header com Logo
    logo = load_logo()
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if logo:
            st.image(logo, width=80)
        else:
            st.markdown("### 📊")
    
    with col2:
        st.title("Veros Intelligence Dashboard")
        st.markdown("Análise preditiva e insights automatizados")
    
    st.markdown("---")
    
    # Verificar se arquivo existe
    file_path = r"C:\Users\USER\Documents\Veros_DataHub\DADOS_RECEITA_VEROS.xlsx"
    if not os.path.exists(file_path):
        st.error(f"Arquivo não encontrado: {file_path}")
        st.info("Verifique se o caminho do arquivo está correto")
        return
    
    # Carregar dados
    with st.spinner("Carregando e analisando dados..."):
        df = load_data()
    
    if df is None:
        st.stop()
    
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
        
        # Criar matrizes escadinhas - CORREÇÃO: Passar ano_filtro para todas as funções
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
        
        # NOVO: Análise de Performance por LP
        performance_lp = analisar_performance_lp(df, ano_filtro=ano_selecionado)
        performance_mensal_lp = analisar_performance_mensal_lp(df, ano_filtro=ano_selecionado)
    
    # SISTEMA DE ABAS ATUALIZADO - ADICIONANDO ANÁLISE DE LP
    tab1, tab2, tab3 = st.tabs([
        "Visão Geral", 
        "Matrizes Escadinha",
        "📊 Análise de LPs"
    ])
    
    with tab1:
        # Visão Geral (mantida igual)
        st.markdown('<div class="section-header"><h2 class="section-title">Visão Geral do Performance</h2><p class="section-subtitle">Métricas consolidadas e tendências do período</p></div>', unsafe_allow_html=True)
        
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
                
                # Adicionar rótulos de dados
                fig_receita.update_traces(
                    texttemplate='R$ %{y:,.0f}',
                    textposition='top center',
                    mode='lines+markers+text'
                )
                
                fig_receita = configurar_layout_clean(fig_receita, "Receita Mensal")
                st.plotly_chart(fig_receita, use_container_width=True)
            
            with col2:
                # Gráfico de Novos Tutores COM RÓTULOS
                fig_tutores = px.bar(
                    novos_tutores_mes,
                    x='Mês',
                    y='Novos Tutores',
                    title="Novos Tutores por Mês",
                    color_discrete_sequence=[COLORS['info']],
                    text_auto=True  # Adiciona rótulos automaticamente
                )
                
                # Melhorar os rótulos
                fig_tutores.update_traces(
                    texttemplate='%{y}',
                    textposition='outside'
                )
                
                fig_tutores = configurar_layout_clean(fig_tutores, "Novos Tutores por Mês")
                st.plotly_chart(fig_tutores, use_container_width=True)
        else:
            st.warning("Não há dados disponíveis para o período selecionado")
    
    with tab2:
        # ABA CONSOLIDADA - TODAS AS MATRIZES ESCADINHA (COMPLETA)
        st.markdown('<div class="section-header"><h2 class="section-title">Matrizes Escadinha - Análise Temporal Completa</h2><p class="section-subtitle">Relação entre geração de receita e geração de leads</p></div>', unsafe_allow_html=True)
        
        # Informação sobre o filtro aplicado
        st.markdown(f"""
        <div class="info-box">
            <h4>📅 Período Selecionado: {ano_selecionado}</h4>
            <p>As matrizes abaixo mostram apenas os dados do ano {ano_selecionado}. 
            Para analisar outro período, altere o filtro na barra lateral.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Introdução e explicação
        st.markdown("""
        <div class="matriz-stats">
            <h4>📊 Como interpretar as Matrizes Escadinha:</h4>
            <ul>
                <li><strong>Eixo Y (Linhas):</strong> Mês de Geração da Receita</li>
                <li><strong>Eixo X (Colunas):</strong> Mês de Geração do Lead</li>
                <li><strong>Diagonal Principal:</strong> Meses coincidentes (condição ideal)</li>
                <li><strong>Células fora da diagonal:</strong> Receita gerada em meses diferentes da geração do lead</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Verificar se há dados para as matrizes
        if (matriz_receita is None or matriz_receita.empty) and \
           (matriz_cac is None or matriz_cac.empty) and \
           (matriz_ltv is None or matriz_ltv.empty) and \
           (matriz_cac_ltv is None or matriz_cac_ltv.empty):
            st.warning(f"⚠️ Não há dados disponíveis para criar as matrizes escadinha do ano {ano_selecionado}.")
            st.info("💡 Tente selecionar outro ano ou verifique se os dados estão corretamente formatados.")
        else:
            # SEÇÃO 1: VISUALIZAÇÕES DAS MATRIZES
            st.subheader("📈 Visualizações das Matrizes")
            
            # Grid de Heatmaps
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
            
            # SEÇÃO 3: ANÁLISE DETALHADA POR MATRIZ
            st.subheader("🔍 Análise Detalhada por Matriz")
            
            # Abas para análise detalhada de cada matriz
            analise_tab1, analise_tab2, analise_tab3, analise_tab4 = st.tabs([
                "💰 Receita", "💸 CAC", "💎 LTV", "⚖️ CAC/LTV"
            ])
            
            with analise_tab1:
                if estatisticas_receita:
                    st.markdown("""
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
                        st.dataframe(matriz_formatada_receita, use_container_width=True, height=300)
                else:
                    st.warning("Não há dados disponíveis para a matriz de Receita")
            
            with analise_tab2:
                if estatisticas_cac:
                    st.markdown("""
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
                        st.dataframe(matriz_formatada_cac, use_container_width=True, height=300)
                else:
                    st.warning("Não há dados disponíveis para a matriz de CAC")
            
            with analise_tab3:
                if estatisticas_ltv:
                    st.markdown("""
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
                        st.dataframe(matriz_formatada_ltv, use_container_width=True, height=300)
                else:
                    st.warning("Não há dados disponíveis para a matriz de LTV")
            
            with analise_tab4:
                if estatisticas_cac_ltv:
                    st.markdown("""
                    <div class="matriz-stats">
                        <h4>Análise da Matriz de CAC/LTV</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Análise de Performance
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
                    
                    # Análise de performance
                    st.subheader("📋 Análise de Performance")
                    if ratio_medio < 1.0:
                        st.success("**✅ PERFORMANCE SAUDÁVEL** - A razão CAC/LTV média indica que o custo de aquisição é menor que o valor do cliente, sugerindo sustentabilidade do negócio.")
                    else:
                        st.warning("**⚠️ ATENÇÃO NECESSÁRIA** - A razão CAC/LTV média indica que o custo de aquisição supera o valor do cliente, necessitando otimização das estratégias.")
                    
                    if estatisticas_cac_ltv['percentual_saudavel'] > 50:
                        st.success(f"✅ MAIORIA SAUDÁVEL - {estatisticas_cac_ltv['percentual_saudavel']:.1f}% das combinações têm CAC/LTV < 1.0")
                    else:
                        st.error(f"❌ MAIORIA PROBLEMÁTICA - Apenas {estatisticas_cac_ltv['percentual_saudavel']:.1f}% das combinações têm CAC/LTV < 1.0")
                    
                    if matriz_formatada_cac_ltv is not None:
                        st.subheader("Tabela Detalhada - CAC/LTV")
                        st.dataframe(matriz_formatada_cac_ltv, use_container_width=True, height=300)
                else:
                    st.warning("Não há dados disponíveis para a matriz de CAC/LTV")
            
            # SEÇÃO 4: INSIGHTS E RECOMENDAÇÕES
            st.subheader("💡 Insights e Recomendações")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="matriz-stats">
                    <h4>🎯 Pontos Fortes</h4>
                    <ul>
                        <li>Alta concentração na diagonal indica eficiência no processo de conversão</li>
                        <li>Padrão consistente sugere processos bem estabelecidos</li>
                        <li>Baixa dispersão temporal entre geração de lead e receita</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="matriz-stats">
                    <h4>🚀 Oportunidades</h4>
                    <ul>
                        <li>Analisar células fora da diagonal para entender conversões atípicas</li>
                        <li>Otimizar tempo de conversão baseado nos padrões identificados</li>
                        <li>Segmentar por unidade de negócio para análises mais específicas</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        # NOVA ABA: ANÁLISE DE LPs
        st.markdown('<div class="section-header"><h2 class="section-title">Análise de Performance por Landing Page</h2><p class="section-subtitle">Métricas detalhadas por canal de aquisição</p></div>', unsafe_allow_html=True)
        
        # Informação sobre investimento total
        total_investimento_lp = sum(INVESTIMENTO_POR_LP.values())
        
        if not performance_lp.empty:
            # Tabela principal de performance
            st.subheader("📈 Performance por LP")
            
            # Formatar a tabela para exibição
            performance_formatada = performance_lp.copy()
            
            # Formatar colunas monetárias
            colunas_monetarias = ['Investimento', 'CPL', 'CAC', 'TM', 'Receita', 'Receita Cohort']
            for col in colunas_monetarias:
                performance_formatada[col] = performance_formatada[col].apply(
                    lambda x: f"R$ {x:,.2f}" if x > 0 else "R$ 0.00"
                )
            
            # Formatar colunas percentuais
            colunas_percentuais = ['Tx.Conv', 'ROAS']
            for col in colunas_percentuais:
                performance_formatada[col] = performance_formatada[col].apply(
                    lambda x: f"{x:.1f}%" if x > 0 else "0.0%"
                )
            
            # Formatar CAC/LTV
            performance_formatada['CAC/LTV'] = performance_formatada['CAC/LTV'].apply(
                lambda x: f"{x:.2f}" if x > 0 else "0.00"
            )
            
            # Formatar Leads
            performance_formatada['Leads'] = performance_formatada['Leads'].apply(lambda x: f"{x:,.0f}")
            
            # Exibir tabela
            st.dataframe(performance_formatada, use_container_width=True)
            
            # Gráficos de Performance COM RÓTULOS
            st.subheader("📊 Visualizações de Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de Receita por LP COM RÓTULOS
                fig_receita_lp = px.bar(
                    performance_lp,
                    x='LP',
                    y='Receita',
                    title="Receita por LP",
                    color='Receita',
                    color_continuous_scale='Viridis',
                    text_auto=True
                )
                
                # Melhorar rótulos
                fig_receita_lp.update_traces(
                    texttemplate='R$ %{y:,.0f}',
                    textposition='outside'
                )
                
                fig_receita_lp = configurar_layout_clean(fig_receita_lp, "Receita por Landing Page")
                st.plotly_chart(fig_receita_lp, use_container_width=True)
            
            with col2:
                # Gráfico de ROAS por LP COM RÓTULOS
                fig_roas_lp = px.bar(
                    performance_lp,
                    x='LP',
                    y='ROAS',
                    title="ROAS por LP",
                    color='ROAS',
                    color_continuous_scale='RdYlGn',
                    text_auto=True
                )
                
                # Melhorar rótulos
                fig_roas_lp.update_traces(
                    texttemplate='%{y:.1f}%',
                    textposition='outside'
                )
                
                fig_roas_lp = configurar_layout_clean(fig_roas_lp, "ROAS por Landing Page")
                st.plotly_chart(fig_roas_lp, use_container_width=True)
            
            # NOVA SEÇÃO: Performance Mensal por LP
            if not performance_mensal_lp.empty:
                st.subheader("📅 Performance Mensal por LP")
                
                # Selecionar LP para análise mensal
                lps_disponiveis = performance_mensal_lp['LP'].unique()
                lp_selecionada = st.selectbox("Selecione a LP para análise mensal:", lps_disponiveis)
                
                if lp_selecionada:
                    # Filtrar dados para a LP selecionada
                    dados_lp_mensal = performance_mensal_lp[performance_mensal_lp['LP'] == lp_selecionada]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Gráfico de Receita Mensal
                        fig_receita_mensal = px.line(
                            dados_lp_mensal,
                            x='Mês',
                            y='Receita',
                            title=f"Receita Mensal - {lp_selecionada}",
                            markers=True
                        )
                        
                        fig_receita_mensal.update_traces(
                            line=dict(color=COLORS['primary'], width=3),
                            marker=dict(size=8, color=COLORS['primary']),
                            texttemplate='R$ %{y:,.0f}',
                            textposition='top center'
                        )
                        
                        fig_receita_mensal = configurar_layout_clean(fig_receita_mensal, f"Receita Mensal - {lp_selecionada}")
                        st.plotly_chart(fig_receita_mensal, use_container_width=True)
                    
                    with col2:
                        # Gráfico de ROAS Mensal
                        fig_roas_mensal = px.bar(
                            dados_lp_mensal,
                            x='Mês',
                            y='ROAS',
                            title=f"ROAS Mensal - {lp_selecionada}",
                            color='ROAS',
                            color_continuous_scale='RdYlGn'
                        )
                        
                        fig_roas_mensal.update_traces(
                            texttemplate='%{y:.1f}%',
                            textposition='outside'
                        )
                        
                        fig_roas_mensal = configurar_layout_clean(fig_roas_mensal, f"ROAS Mensal - {lp_selecionada}")
                        st.plotly_chart(fig_roas_mensal, use_container_width=True)
                    
                    # Tabela de performance mensal detalhada
                    st.subheader(f"📋 Performance Mensal Detalhada - {lp_selecionada}")
                    
                    # Formatar tabela mensal
                    dados_formatados = dados_lp_mensal.copy()
                    dados_formatados['Receita'] = dados_formatados['Receita'].apply(lambda x: f"R$ {x:,.2f}")
                    dados_formatados['ROAS'] = dados_formatados['ROAS'].apply(lambda x: f"{x:.1f}%")
                    dados_formatados['CAC/LTV'] = dados_formatados['CAC/LTV'].apply(lambda x: f"{x:.2f}")
                    dados_formatados['Leads'] = dados_formatados['Leads'].apply(lambda x: f"{x:,.0f}")
                    
                    st.dataframe(dados_formatados[['Mês', 'Receita', 'Leads', 'ROAS', 'CAC/LTV']], use_container_width=True)
            
            # Análise de Eficiência
            st.subheader("💡 Análise de Eficiência por LP")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # LPs com melhor CAC/LTV
                lps_saudaveis = performance_lp[performance_lp['CAC/LTV'] < 1.0]
                if not lps_saudaveis.empty:
                    st.success("**✅ LPs com Performance Saudável (CAC/LTV < 1.0):**")
                    for _, lp in lps_saudaveis.iterrows():
                        st.write(f"- **{lp['LP']}**: CAC/LTV = {lp['CAC/LTV']:.2f}")
                else:
                    st.warning("⚠️ Nenhuma LP com CAC/LTV abaixo de 1.0")
            
            with col2:
                # LPs com pior CAC/LTV
                lps_problematicas = performance_lp[performance_lp['CAC/LTV'] >= 1.0]
                if not lps_problematicas.empty:
                    st.error("**❌ LPs que Precisam de Atenção (CAC/LTV ≥ 1.0):**")
                    for _, lp in lps_problematicas.iterrows():
                        st.write(f"- **{lp['LP']}**: CAC/LTV = {lp['CAC/LTV']:.2f}")
                else:
                    st.success("✅ Todas as LPs têm CAC/LTV saudável")
            
            # Insights e Recomendações
            st.subheader("🎯 Insights e Recomendações")
            
            # Encontrar melhor e pior LP
            melhor_lp = performance_lp.loc[performance_lp['Receita'].idxmax()]
            pior_lp = performance_lp.loc[performance_lp['Receita'].idxmin()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="matriz-stats">
                    <h4>🏆 Melhor Performance: {melhor_lp['LP']}</h4>
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
                    <h4>📉 Oportunidade de Melhoria: {pior_lp['LP']}</h4>
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

if __name__ == "__main__":
    main()