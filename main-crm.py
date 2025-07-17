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
tab1, tab2, tab3 = st.tabs(["📋 Tabla", "📊 Dashboard", "📈 Análisis Predictivo"])

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

        # --- RESUMEN GENERADO CON IA ---
        with st.expander("🧠 Análisis generado por IA"):
            if st.button("Generar resumen con IA", key="resumen_ia_tab1"):
                from openai import OpenAI

                resumen = f"""
                Se han cargado {len(df_mostrar)} oportunidades.
                El importe total visible es de {df_mostrar['Importe'].str.replace('$','').str.replace('.','').astype(float).sum():,.0f} CLP.
                Hay {sum(df_mostrar['Fecha de Cierre'].dt.month == dt.datetime.today().month)} oportunidades con cierre este mes.
                El promedio de probabilidad declarada es de {pd.to_numeric(df_mostrar['Probabilidad'].str.replace('%','')).mean():.1f}%.
                """

                if "OPENAI_API_KEY" in st.secrets:
                    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Eres un asistente experto en análisis de oportunidades comerciales."},
                            {"role": "user", "content": f"Con base en este resumen del pipeline: {resumen}, ¿qué observaciones clave destacarías?"}
                        ]
                    )
                    st.markdown(response.choices[0].message.content)
                else:
                    st.warning("No se ha configurado la API key de OpenAI. Agrega OPENAI_API_KEY a tus secretos.")

        # --- DATATABLE ---
        st.subheader("📋 Oportunidades")

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
        st.write(
            styled_table.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )


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

            model_option = st.selectbox(
                "Selecciona el modelo a aplicar",
                ["Random Forest (v1)", "Logistic Regression", "XGBoost", "LightGBM", "MLPClassifier"]
            )

            # Mostrar info del modelo seleccionado
            if model_option == "Random Forest (v1)":
                st.info("🌳 **Random Forest (v1):** Combina múltiples árboles de decisión entrenados con distintas partes del dataset. Ideal para datos mixtos (numéricos y categóricos). Robusto ante ruido.")
            elif model_option == "Logistic Regression":
                st.info("📈 **Logistic Regression:** Modelo estadístico clásico, útil como línea base. Rápido, simple y fácil de interpretar.")
            elif model_option == "XGBoost":
                st.info("⚡ **XGBoost:** Variante avanzada de boosting. Alta precisión, pero puede requerir más ajustes. Excelente en datos tabulares.")
            elif model_option == "LightGBM":
                st.info("🚀 **LightGBM:** Modelo basado en boosting optimizado para velocidad. Muy bueno con muchas filas y columnas.")
            elif model_option == "MLPClassifier":
                st.info("🧠 **Red Neuronal (MLP):** Modelo con capas ocultas que puede capturar relaciones no lineales. Requiere algo más de procesamiento y entrenamiento.")

            if model_option == "Random Forest (v1)":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_option == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000)
            elif model_option == "XGBoost":
                from xgboost import XGBClassifier
                model = XGBClassifier(eval_metric='logloss')
            elif model_option == "LightGBM":
                from lightgbm import LGBMClassifier
                model = LGBMClassifier()
            elif model_option == "MLPClassifier":
                from sklearn.neural_network import MLPClassifier
                model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)

            # Preparar etiquetas
            df_hist_full["estado_objetivo"] = df_hist_full["Estado Oportunidad"].str.lower().map({
                "ganada": 1,
                "descartada": 0,
                "perdida": 0
            })
            df_hist_modelo = df_hist_full.dropna(subset=["estado_objetivo"])

            # Historial por responsable
            win_por_responsable = df_hist_modelo[df_hist_modelo["estado_objetivo"] == 1]["Responsable"].value_counts(normalize=True)
            df_hist_modelo["ganadas_por_responsable"] = df_hist_modelo["Responsable"].map(win_por_responsable).fillna(0)

            # Historial por cliente
            win_por_cliente = df_hist_modelo[df_hist_modelo["estado_objetivo"] == 1]["Cliente"].value_counts(normalize=True)
            df_hist_modelo["ganadas_por_cliente"] = df_hist_modelo["Cliente"].map(win_por_cliente).fillna(0)

            # Columnas a usar
            columnas_modelo = [
                "Importe", "Probabilidad", "Responsable", "Cliente",
                "Tipo de Trarificación", "Modelo de Ejecuciones",
                "Servicio/Subservicio/XtechCore.",
                "ganadas_por_responsable", "ganadas_por_cliente",
                "2025 backlog", "2026 backlog", "2027 backlog", "2028 backlog"
            ]
            df_model = df_hist_modelo[columnas_modelo + ["estado_objetivo"]].copy()

            # Limpiar numéricos
            for col in ["Importe", "2025 backlog", "2026 backlog", "2027 backlog", "2028 backlog"]:
                df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0)

            # Codificar texto
            from sklearn.preprocessing import LabelEncoder
            label_cols = [
                "Responsable", "Cliente", "Tipo de Trarificación",
                "Modelo de Ejecuciones", "Servicio/Subservicio/XtechCore."
            ]
            encoders = {}
            for col in label_cols:
                enc = LabelEncoder()
                df_model[col] = enc.fit_transform(df_model[col].astype(str))
                encoders[col] = enc

            # Entrenar modelo
            X = df_model.drop("estado_objetivo", axis=1)
            y = df_model["estado_objetivo"]
            model.fit(X.values, y.values)

            # Aplicar a oportunidades vivas
            estados_excluir = ["ganada", "descartada", "perdida"]
            df_vivas = df_hist_full[~df_hist_full["Estado Oportunidad"].str.lower().isin(estados_excluir)].copy()

            # Calcular historial para oportunidades vivas
            df_vivas["ganadas_por_responsable"] = df_vivas["Responsable"].map(win_por_responsable).fillna(0)
            df_vivas["ganadas_por_cliente"] = df_vivas["Cliente"].map(win_por_cliente).fillna(0)
            df_vivas_model = df_vivas[columnas_modelo].copy()

            for col in ["Importe", "2025 backlog", "2026 backlog", "2027 backlog", "2028 backlog"]:
                df_vivas_model[col] = pd.to_numeric(df_vivas_model[col], errors="coerce").fillna(0)

            for col in label_cols:
                df_vivas_model[col] = df_vivas_model[col].astype(str)
                df_vivas_model[col] = df_vivas_model[col].apply(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)

            df_vivas["Predicción"] = model.predict(df_vivas_model.values)
            df_vivas["Probabilidad de Ganar"] = model.predict_proba(df_vivas_model.values)[:, 1]

            # Ajustar probabilidad según estado de la oportunidad
            ajuste_estado = {
                "identificada": 0.85,
                "preparación de la propuesta": 1.10,
                "propuesta enviada": 1.15,
                "validada": 1.20,
                "en validación": 1.20,
            }
            df_vivas["estado_normalizado"] = df_vivas["Estado Oportunidad"].str.lower()
            df_vivas["Probabilidad Ajustada"] = df_vivas.apply(
                lambda row: min(row["Probabilidad de Ganar"] * ajuste_estado.get(row["estado_normalizado"], 1), 1.0),
                axis=1
            )
            df_vivas = df_vivas.drop(columns=["estado_normalizado"])
            df_vivas = df_vivas.sort_values(by="Probabilidad Ajustada", ascending=False)
            df_vivas["Probabilidad Ajustada"] = df_vivas["Probabilidad Ajustada"].apply(lambda x: f"{x:.0%}")

            df_vivas["Importe"] = df_vivas["Importe"].apply(lambda x: f"${x:,.0f}".replace(",", "."))

            st.markdown("### 📋 Predicciones sobre oportunidades vivas")
            st.dataframe(df_vivas[[
                "Estado Oportunidad", "Título", "Cliente", "Responsable", "Importe",
                "Probabilidad", "Fecha Cierre Oportunidad",
                "Predicción", "Probabilidad Ajustada"
            ]], use_container_width=True)

            # --- ANÁLISIS DEL MODELO SELECCIONADO ---
            st.markdown("### 📌 Análisis del modelo seleccionado")
            with st.expander("🧠 Análisis del modelo con IA"):
                if st.button("Generar explicación del modelo", key="analisis_ia_tab3"):
                    from openai import OpenAI

                    top_5 = df_vivas[df_vivas["Predicción"] == 1].head(5)
                    resumen_top = "\n".join(
                        f"- {row['Título']} (Cliente: {row['Cliente']}, Responsable: {row['Responsable']}, Cierre: {row['Fecha Cierre Oportunidad'].date()}, Prob. ajustada: {row['Probabilidad Ajustada']})"
                        for _, row in top_5.iterrows()
                    )

                    top_clientes_prob = df_vivas[df_vivas["Predicción"] == 1]["Cliente"].value_counts().head(5)
                    resumen_clientes = "\n".join(
                        f"- {cliente}: {count} oportunidades ganadas predichas"
                        for cliente, count in top_clientes_prob.items()
                    )

                    top_clientes = top_5["Cliente"].value_counts().idxmax() if not top_5.empty else "N/A"
                    top_responsables = top_5["Responsable"].value_counts().idxmax() if not top_5.empty else "N/A"

                    explicacion = f"""
El modelo seleccionado es {model_option}.
Se ha aplicado sobre {len(df_vivas)} oportunidades vivas con datos como importe, probabilidad, historial del cliente y responsable.
{len(df_vivas[df_vivas['Predicción'] == 1])} de ellas fueron predichas como ganadas.

Las 5 oportunidades más prometedoras según el modelo son:
{resumen_top}

📌 Observación: El cliente más frecuente entre estas oportunidades es **{top_clientes}**, y el responsable con más presencia es **{top_responsables}**.

📊 Clientes destacados con más oportunidades ganadas según el modelo:
{resumen_clientes}
"""

                    if "OPENAI_API_KEY" in st.secrets:
                        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "Eres un experto en ciencia de datos y analítica predictiva. Explica cómo funciona el modelo seleccionado y qué insights clave puede extraer un gerente comercial."},
                                {"role": "user", "content": explicacion}
                            ]
                        )
                        st.markdown(response.choices[0].message.content)
                    else:
                        st.warning("No se ha configurado la API key de OpenAI. Agrega OPENAI_API_KEY a tus secretos.")

else:
    st.info("Carga un archivo Excel para comenzar.")