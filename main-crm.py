import streamlit as st
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

st.set_page_config(page_title="CRM Semanal", layout="wide")

st.title("🔍 Revisión Semanal del Pipeline CRM")

# --- CARGA DEL ARCHIVO EXCEL ---
st.sidebar.header("📤 Carga de Export")
uploaded_file = st.sidebar.file_uploader("Sube el Excel exportado desde BHZ", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # --- FILTRAR COLUMNAS RELEVANTES ---
    columnas = [
        "Estado Oportunidad", "Enlace a la Oportunidad", "Título", "Responsable", "Cliente", "Importe",
        "Importe Servicio", "Margen Bruto", "Porcentaje de Margen Bruto", "Probabilidad", "Fecha de detección",
        "Fecha Cierre Oportunidad", "Modificado En", "Modelo de Ejecuciones", "Fecha presentación propuesta",
        "Acuerdo Marco", "Servicio/Subservicio/XtechCore.", "Fecha de Inicio Estimada",
        "2025 backlog", "2026 backlog", "2027 backlog", "2028 backlog"
    ]
    df = df[columnas]
    df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_").replace(".", "") for c in df.columns]

    # --- PARSE FECHAS ---
    for col in ["fecha_cierre_oportunidad", "fecha_de_detección", "modificado_en", "fecha_presentación_propuesta", "fecha_de_inicio_estimada"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- PANEL DE FILTROS ---
    with st.expander("📊 Filtros"):
        col1, col2, col3 = st.columns(3)
        with col1:
            estado = st.multiselect("Estado", df.estado_oportunidad.unique(), default=df.estado_oportunidad.unique())
        with col2:
            responsables = st.multiselect("Responsable", df.responsable.unique(), default=df.responsable.unique())
        with col3:
            clientes = st.multiselect("Cliente", df.cliente.unique(), default=df.cliente.unique())

        df = df[
            df.estado_oportunidad.isin(estado) &
            df.responsable.isin(responsables) &
            df.cliente.isin(clientes)
        ]

    # --- DATATABLE ---
    st.subheader("📋 Oportunidades filtradas")
    st.dataframe(df, use_container_width=True)

    # --- INDICADORES Y GRÁFICOS ---
    st.subheader("📈 Dashboard de Indicadores")
    col1, col2, col3 = st.columns(3)

    total_pipeline = df["importe"].sum()
    atrasadas = df[df.fecha_cierre_oportunidad < dt.datetime.today()]
    mes_actual = df[df.fecha_cierre_oportunidad.dt.month == dt.datetime.today().month]

    col1.metric("💰 Total Pipeline", f"${total_pipeline:,.0f}")
    col2.metric("⚠️ Ofertas Atrasadas", len(atrasadas))
    col3.metric("📆 Cierre este mes", len(mes_actual))

    # --- BACKLOG POR AÑO ---
    st.markdown("#### 📊 Backlog Proyectado")
    backlog_cols = ["backlog_2025", "backlog_2026", "backlog_2027", "backlog_2028"]
    for col in backlog_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    backlog_totales = df[backlog_cols].sum()

    fig, ax = plt.subplots()
    backlog_totales.plot(kind="bar", ax=ax)
    ax.set_ylabel("CLP")
    ax.set_title("Backlog por año")
    st.pyplot(fig)

    # --- PIPELINE POR CLIENTE ---
    st.markdown("#### 🏢 Pipeline por Cliente")
    pipeline_cliente = df.groupby("cliente")["importe"].sum().sort_values(ascending=False).head(10)

    fig2, ax2 = plt.subplots()
    pipeline_cliente.plot(kind="barh", ax=ax2)
    ax2.set_xlabel("CLP")
    ax2.invert_yaxis()
    st.pyplot(fig2)

else:
    st.info("Carga un archivo Excel para comenzar.")