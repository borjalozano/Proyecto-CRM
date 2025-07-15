import streamlit as st
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="CRM Semanal", layout="wide")

st.image("Logo Babel Horizontal (1).jpg", width=250)
st.title("🔍 Revisión Semanal del Pipeline CRM")

# --- CARGA DEL ARCHIVO EXCEL ---
st.sidebar.header("📤 Carga de Export")

uploaded_file = st.sidebar.file_uploader("Sube el Excel exportado desde BHZ", type=["xlsx"])
st.sidebar.markdown("**🎨 Leyenda de colores:**  \n"
                    "- 🔴 Rojo: Oferta atrasada  \n"
                    "- 🟡 Amarillo: Cierre este mes  \n"
                    "- 🟢 Verde: Cierre futuro")


# --- Definición de pestañas ---
tab1, tab2, tab3 = st.tabs(["📋 Tabla", "📊 Dashboard", "🤖 Análisis IA"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Limpiar y convertir columnas monetarias
    if "importe" in df.columns:
        df["importe"] = df["importe"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        df["importe"] = pd.to_numeric(df["importe"], errors="coerce").fillna(0).round(0).astype(int)

    for col in ["2025 backlog", "2026 backlog", "2027 backlog", "2028 backlog"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(0).astype(int)

    # --- FILTRAR COLUMNAS RELEVANTES ---
    columnas = [
        "Estado Oportunidad", "Enlace a la Oportunidad", "Título", "Responsable", "Cliente", "Importe",
        "Margen Bruto", "Porcentaje de Margen Bruto", "Probabilidad", "Fecha de detección",
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

    df = df[df.acuerdo_marco.str.lower() != "sí"]

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
            "cliente", "título_link", "importe", "probabilidad", "responsable",
            "fecha_cierre_oportunidad", "backlog_2025", "backlog_2026", "backlog_2027", "backlog_2028"
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
            "cliente", "título_link", "importe", "probabilidad", "responsable",
            "fecha_cierre_oportunidad", "backlog_2025", "backlog_2026", "backlog_2027", "backlog_2028"
        ]

        df_mostrar = df[columnas_tabla].copy()
        df_mostrar.columns = [
            "Cliente", "Título", "Importe", "Probabilidad", "Responsable",
            "Fecha de Cierre", "Backlog 2025", "Backlog 2026", "Backlog 2027", "Backlog 2028"
        ]

        # Formatear columnas monetarias como CLP sin  decimales
        for col in ["Importe", "Backlog 2025", "Backlog 2026", "Backlog 2027", "Backlog 2028"]:
            df_mostrar[col] = pd.to_numeric(df_mostrar[col], errors="coerce").fillna(0).astype(int)
            df_mostrar[col] = df_mostrar[col].apply(lambda x: f"${x:,.0f}".replace(",", "."))
        df_mostrar["Probabilidad"] = pd.to_numeric(df_mostrar["Probabilidad"], errors="coerce").fillna(0).astype(int)
        df_mostrar["Probabilidad"] = df_mostrar["Probabilidad"].apply(lambda x: f"{x}%")

        df_mostrar = df_mostrar.sort_values(by="Fecha de Cierre", ascending=True)

        # Mostrar como tabla HTML con enlaces y colorear filas
        def row_style(row):
            if pd.isnull(row["Fecha de Cierre"]):
                return ''
            elif row["Fecha de Cierre"] < dt.datetime.today():
                return 'background-color: #f8d7da'  # Rojo claro
            elif row["Fecha de Cierre"].month == dt.datetime.today().month and row["Fecha de Cierre"].year == dt.datetime.today().year:
                return 'background-color: #fff3cd'  # Amarillo claro
            else:
                return 'background-color: #d4edda'  # Verde claro
        styled_table = df_mostrar.style.apply(lambda row: [row_style(row)] * len(row), axis=1)
        st.write(styled_table.to_html(escape=False, index=False), unsafe_allow_html=True)

    with tab2:
        with st.expander("📊 Filtros Dashboard"):
            col1, col2, col3 = st.columns(3)
            with col1:
                estado_d = st.multiselect("Estado", df.estado_oportunidad.unique(), default=df.estado_oportunidad.unique(), key="estado_dashboard")
            with col2:
                responsables_d = st.multiselect("Responsable", df.responsable.unique(), default=df.responsable.unique(), key="responsable_dashboard")
            with col3:
                clientes_d = st.multiselect("Cliente", df.cliente.unique(), default=df.cliente.unique(), key="cliente_dashboard")

            df = df[
                df.estado_oportunidad.isin(estado_d) &
                df.responsable.isin(responsables_d) &
                df.cliente.isin(clientes_d)
            ]

        # --- INDICADORES Y GRÁFICOS ---
        st.subheader("📈 Dashboard de Indicadores")
        col1, col2, col3 = st.columns(3)

        total_pipeline = df["importe"].sum()
        atrasadas = df[df.fecha_cierre_oportunidad < dt.datetime.today()]
        mes_actual = df[df.fecha_cierre_oportunidad.dt.month == dt.datetime.today().month]

        col1.metric("💰 Total Pipeline", f"${total_pipeline:,.0f}")
        col2.metric("⚠️ Ofertas Atrasadas", len(atrasadas))
        col3.metric("📆 Ofertas con Cierre este mes", len(mes_actual))

        col4 = st.columns(1)[0]
        col4.metric("📦 Total Oportunidades en Pipeline", f"{len(df):,}".replace(",", "."))

        # --- BACKLOG POR AÑO ---
        st.markdown("#### 📊 Pipeline Proyectado")
        backlog_cols = ["backlog_2025", "backlog_2026", "backlog_2027", "backlog_2028"]

        # Convertir a numérico antes de usar en cualquier parte
        for col in backlog_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        backlog_totales = [df["backlog_2025"].sum(), df["backlog_2026"].sum(), df["backlog_2027"].sum(), df["backlog_2028"].sum()]
        backlog_totales = [round(val / 1_000_000, 1) for val in backlog_totales]

        import pandas as pd
        pipeline_df = pd.DataFrame({
            "Año": ["2025", "2026", "2027", "2028"],
            "Millones CLP": backlog_totales
        })
        fig = px.bar(
            pipeline_df.astype({"Año": "category"}),
            x="Año",
            y="Millones CLP",
            labels={'x': 'Año', 'y': 'Millones CLP'},
            text="Millones CLP",
            title="Pipeline por año"
        )
        fig.update_traces(texttemplate='%{text:.1f}', hovertemplate='CLP %{y:.1f} millones')
        st.plotly_chart(fig, use_container_width=True)

        # --- PIPELINE POR CLIENTE ---
        st.markdown("#### 🏢 Pipeline por Cliente")
        pipeline_cliente = (df.groupby("cliente")["importe"].sum().sort_values(ascending=False).head(10) / 1_000_000).round(1)

        fig2 = px.bar(
            x=pipeline_cliente.values,
            y=pipeline_cliente.index,
            orientation='h',
            labels={'x': 'Millones CLP', 'y': 'Cliente'},
            text=pipeline_cliente.values,
            title="Pipeline por Cliente"
        )
        fig2.update_traces(texttemplate='%{text:.1f}', hovertemplate='CLP %{x:.1f} millones')
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("🤖 Predicción de Cierre de Oportunidades")

        hist_file = st.file_uploader("Cargar histórico completo (.xlsx)", type=["xlsx"], key="hist")
        if hist_file:
            df_hist_full = pd.read_excel(hist_file)

            model_option = st.selectbox("Selecciona el modelo a aplicar", ["Random Forest (v1)"])
            if model_option == "Random Forest (v1)":
                st.info(
                    "🌳 **Random Forest (v1)**: Este modelo está basado en múltiples árboles de decisión entrenados "
                    "sobre distintas combinaciones de datos. Evalúa factores como importe, probabilidad, tipo de servicio y responsable, "
                    "para estimar si una oportunidad se ganará o no. Es robusto ante ruido y útil cuando se combinan variables categóricas y numéricas."
                )
                # Preparar etiquetas
                df_hist_full["estado_objetivo"] = df_hist_full["Estado Oportunidad"].str.lower().map({
                    "ganada": 1,
                    "descartada": 0,
                    "perdida": 0
                })
                df_hist_modelo = df_hist_full.dropna(subset=["estado_objetivo"])

                # Columnas a usar
                columnas_modelo = [
                    "Importe", "Probabilidad", "Responsable", "Cliente",
                    "Tipo de Trarificación", "Modelo de Ejecuciones",
                    "Servicio/Subservicio/XtechCore.",
                    "2025 backlog", "2026 backlog", "2027 backlog", "2028 backlog"
                ]
                df_model = df_hist_modelo[columnas_modelo + ["estado_objetivo"]].copy()

                # Limpiar numéricos
                for col in ["Importe", "2025 backlog", "2026 backlog", "2027 backlog", "2028 backlog"]:
                    df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0)

                # Codificar texto
                from sklearn.preprocessing import LabelEncoder
                label_cols = ["Responsable", "Cliente", "Tipo de Trarificación", "Modelo de Ejecuciones", "Servicio/Subservicio/XtechCore."]
                encoders = {}
                for col in label_cols:
                    enc = LabelEncoder()
                    df_model[col] = enc.fit_transform(df_model[col].astype(str))
                    encoders[col] = enc

                # Entrenar modelo
                from sklearn.ensemble import RandomForestClassifier
                X = df_model.drop("estado_objetivo", axis=1)
                y = df_model["estado_objetivo"]
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)

                # Aplicar a oportunidades vivas
                estados_excluir = ["ganada", "descartada", "perdida"]
                df_vivas = df_hist_full[~df_hist_full["Estado Oportunidad"].str.lower().isin(estados_excluir)].copy()
                df_vivas_model = df_vivas[columnas_modelo].copy()

                for col in ["Importe", "2025 backlog", "2026 backlog", "2027 backlog", "2028 backlog"]:
                    df_vivas_model[col] = pd.to_numeric(df_vivas_model[col], errors="coerce").fillna(0)

                for col in label_cols:
                    df_vivas_model[col] = df_vivas_model[col].astype(str)
                    df_vivas_model[col] = df_vivas_model[col].apply(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)

                df_vivas["Predicción"] = model.predict(df_vivas_model)
                df_vivas["Probabilidad de Ganar"] = model.predict_proba(df_vivas_model)[:, 1]

                # Ajustar probabilidad según etapa del pipeline
                ponderador_estado = {
                    "preparación de la propuesta": 1.10,
                    "propuesta enviada": 1.15,
                    "validada": 1.20,
                    "en validación": 1.20,
                    "identificada": 0.85,
                }
                df_vivas["Estado Oportunidad"] = df_vivas["Estado Oportunidad"].str.lower()
                df_vivas["Probabilidad de Ganar Ajustada"] = df_vivas.apply(
                    lambda row: min(row["Probabilidad de Ganar"] * ponderador_estado.get(row["Estado Oportunidad"], 1), 1.0),
                    axis=1
                )

                st.markdown("### 📋 Predicciones sobre oportunidades vivas")
                def color_prob(val):
                    if val >= 0.7:
                        return "background-color: #d4edda"  # verde
                    elif val >= 0.5:
                        return "background-color: #fff3cd"  # amarillo
                    else:
                        return "background-color: #f8d7da"  # rojo

                styled_df = df_vivas[[
                    "Estado Oportunidad", "Título", "Cliente", "Responsable", "Importe",
                    "Probabilidad", "Fecha Cierre Oportunidad",
                    "Predicción", "Probabilidad de Ganar Ajustada"
                ]].style.format({"Importe": "${:,.0f}", "Probabilidad de Ganar Ajustada": "{:.2f}"}).applymap(color_prob, subset=["Probabilidad de Ganar Ajustada"])

                st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

else:
    st.info("Carga un archivo Excel para comenzar.")