import streamlit as st
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

st.set_page_config(page_title="CRM Semanal", layout="wide")

st.title("🔍 Revisión Semanal del Pipeline CRM")

# --- CARGA DEL ARCHIVO EXCEL ---
st.sidebar.header("📤 Carga de Export")
uploaded_file = st.sidebar.file_uploader("Sube el Excel exportado desde BHZ", type=["xlsx"])


# --- Definición de pestañas ---
tab1, tab2 = st.tabs(["📋 Tabla", "📊 Dashboard"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Limpiar y convertir columnas monetarias
    for col in ["importe", "importe_servicio", "2025 backlog", "2026 backlog", "2027 backlog", "2028 backlog"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

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

    # Convertir columnas de backlog a numérico desde el principio
    for col in ["2025 backlog", "2026 backlog", "2027 backlog", "2028 backlog"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- PARSE FECHAS ---
    for col in ["fecha_cierre_oportunidad", "fecha_de_detección", "modificado_en", "fecha_presentación_propuesta", "fecha_de_inicio_estimada"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    with tab1:
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

        # Crear columna de título como hipervínculo
        df["título_link"] = df.apply(
            lambda row: f'<a href="{row["enlace_a_la_oportunidad"]}" target="_blank">{row["título"]}</a>',
            axis=1
        )

        # Reordenar y renombrar columnas para la vista
        columnas_tabla = [
            "cliente", "título_link", "importe", "importe_servicio", "probabilidad",
            "fecha_cierre_oportunidad", "2025_backlog", "2026_backlog", "2027_backlog", "2028_backlog"
        ]
        # Asegurar que los nombres internos de backlog coincidan con columnas renombradas
        df.rename(columns={
            "2025_backlog": "backlog_2025",
            "2026_backlog": "backlog_2026",
            "2027_backlog": "backlog_2027",
            "2028_backlog": "backlog_2028"
        }, inplace=True)

        # Ajustar columnas_tabla para usar los nombres correctos
        columnas_tabla = [
            "cliente", "título_link", "importe", "importe_servicio", "probabilidad",
            "fecha_cierre_oportunidad", "backlog_2025", "backlog_2026", "backlog_2027", "backlog_2028"
        ]

        df_mostrar = df[columnas_tabla].copy()
        df_mostrar.columns = [
            "Cliente", "Título", "Importe", "Importe Servicio", "Probabilidad",
            "Fecha de Cierre", "Backlog 2025", "Backlog 2026", "Backlog 2027", "Backlog 2028"
        ]

        # Mostrar como tabla HTML con enlaces
        st.write(df_mostrar.to_html(escape=False, index=False), unsafe_allow_html=True)

    with tab2:
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

        # Convertir a numérico antes de usar en cualquier parte
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