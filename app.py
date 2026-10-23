"""Local Streamlit interface for the Vendor Invoice Intelligence demo."""

from __future__ import annotations

from datetime import date
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import (
    DB_PATH,
    MODEL_DIR,
    MODES,
    REGIONS,
    ROOT,
    analyze_batch,
    analyze_invoice,
    ensure_models,
    load_history,
    load_invoices,
    read_batch_csv,
    safe_csv,
    save_review,
)


st.set_page_config(
    page_title="VendorIQ · Invoice Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #0b1019; color: #edf2fa; }
    html, body, [class*="css"], .stMarkdown, button, input, textarea {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.035em;
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
    [data-testid="stSidebar"] { background: #101723; border-right: 1px solid #202c3d; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 0.25rem 0; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #14202f, #111a28);
        border: 1px solid #243247; border-radius: 14px; padding: 1.2rem;
    }
    [data-testid="stMetricLabel"] { color: #9aaec5; }
    [data-testid="stMetricValue"] { font-size: 1.85rem; color: #f2f6ff; }
    [data-testid="stForm"] { border: 1px solid #253247; border-radius: 14px; padding: 1.4rem; }
    .brand { font-family: 'Space Grotesk', sans-serif; font-size: 1.65rem; font-weight: 700; letter-spacing: -0.06em; }
    .brand span { color: #61dfbe; }
    .eyebrow { color: #61dfbe; font-size: 0.72rem; letter-spacing: 0.16em; font-weight: 700; margin-bottom: 0.7rem; }
    .hero { padding: 0.2rem 0 1.7rem; }
    .hero h1 { font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.13; margin: 0 0 0.7rem; }
    .hero p { color: #9eafc5; max-width: 720px; line-height: 1.7; margin: 0; }
    .topline { display: flex; justify-content: space-between; align-items: center; color: #8ea2bb; font-size: 0.8rem; margin-bottom: 1.8rem; }
    .pill { border: 1px solid #2e554c; border-radius: 30px; padding: 0.35rem 0.8rem; background: #142a27; color: #79e2c4; }
    .muted { color: #90a3bc; font-size: 0.84rem; line-height: 1.7; }
    .section-title { margin-top: 1.9rem; margin-bottom: 0.6rem; }
    .stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 600; }
    [data-testid="stDataFrame"] { border: 1px solid #243247; border-radius: 10px; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing demo models…")
def cached_bundle():
    return ensure_models()


@st.cache_data(show_spinner=False)
def cached_invoices():
    return load_invoices()


def money(value: float) -> str:
    return f"${float(value):,.2f}"


def metric_number(value, digits=3) -> str:
    try:
        value = float(value)
        return f"{value:.{digits}f}" if math.isfinite(value) else "—"
    except (TypeError, ValueError):
        return "—"


def chart_style(fig, height=320):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#111a28",
        plot_bgcolor="#111a28",
        font=dict(family="DM Sans, sans-serif", color="#a8bbd1", size=12),
        margin=dict(l=18, r=18, t=22, b=20),
        legend=dict(orientation="h", y=-0.2),
        colorway=["#61dfbe", "#82aaff", "#eabf78", "#bf9eff", "#ff8c9c"],
    )
    fig.update_xaxes(gridcolor="#243044", zeroline=False)
    fig.update_yaxes(gridcolor="#243044", zeroline=False)
    return fig


def page_heading(eyebrow, title, description):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f"<h1>{title}</h1><p>{description}</p></div>",
        unsafe_allow_html=True,
    )


def default_record():
    return dict(
        invoice_id="MANUAL-001",
        vendor_name="Sample vendor",
        region=REGIONS[0],
        invoice_date=date.today().isoformat(),
        quantity=50,
        invoice_amount=2500.0,
        weight_kg=125.0,
        distance_km=420.0,
        shipping_mode=MODES[0],
        freight_cost=145.0,
        line_total=2500.0,
        line_count=8,
        payment_terms_days=30,
    )


def show_review_result(record, result):
    st.subheader("Review result")
    st.caption(f"Invoice {record['invoice_id']} · {record['vendor_name']}")
    if result["decision"] == "Review required":
        st.warning("Review required — inspect the findings below before approving this invoice.")
    else:
        st.success("No flags detected by the configured model and rules.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected freight", money(result["expected_freight"]))
    c2.metric("Model risk score", f"{result['risk_score']:.1%}")
    c3.metric("Amount mismatch", f"{result.get('amount_mismatch_pct', 0):.1f}%")
    st.caption("The model score is uncalibrated and trained on synthetic data. Policy rules can require review independently of the score.")
    reasons = result.get("reasons", [])
    if reasons:
        st.markdown("**Findings**")
        for reason in reasons:
            st.write(f"• {reason}")
    else:
        st.write("The submitted values passed the configured checks.")
    for warning in result.get("warnings", []):
        st.warning(warning)


def overview(data, metrics):
    page_heading(
        "YOUR OPERATIONS, AT A GLANCE",
        "A clearer view of every invoice.",
        "Explore vendor activity, understand freight costs, and find invoices that deserve a closer look.",
    )
    if data.empty:
        st.info("No demo invoices are available. Restart the app to initialize the demo dataset.")
        return
    controls = st.columns([1, 1, 2])
    region = controls[0].selectbox("Region", ["All regions", *REGIONS], key="overview_region")
    mode = controls[1].selectbox("Shipping mode", ["All modes", *MODES], key="overview_mode")
    filtered = data.copy()
    if region != "All regions":
        filtered = filtered.loc[filtered["region"] == region]
    if mode != "All modes":
        filtered = filtered.loc[filtered["shipping_mode"] == mode]
    if filtered.empty:
        st.info("No invoices match these filters.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Invoices", f"{len(filtered):,}")
    c2.metric("Invoice value", money(filtered["invoice_amount"].sum()))
    c3.metric("Freight spend", money(filtered["freight_cost"].sum()))
    c4.metric("Active vendors", f"{filtered['vendor_name'].nunique():,}")
    left, right = st.columns([1.65, 1])
    with left:
        st.subheader("Invoice activity")
        monthly = filtered.assign(month=pd.to_datetime(filtered["invoice_date"]).dt.to_period("M").astype(str))
        monthly = monthly.groupby("month", as_index=False)["invoice_amount"].sum()
        fig = px.area(monthly, x="month", y="invoice_amount", labels={"month": "Month", "invoice_amount": "Invoice value (USD)"})
        fig.update_traces(line_color="#61dfbe", fillcolor="rgba(97,223,190,0.12)")
        st.plotly_chart(chart_style(fig), width="stretch")
    with right:
        st.subheader("Shipping mix")
        grouped = filtered.groupby("shipping_mode", as_index=False).size()
        fig = px.pie(grouped, names="shipping_mode", values="size", hole=0.68)
        fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color="#111a28", width=3)))
        st.plotly_chart(chart_style(fig), width="stretch")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Freight by region")
        grouped = filtered.groupby("region", as_index=False)["freight_cost"].mean()
        fig = px.bar(grouped, x="region", y="freight_cost", labels={"region": "Region", "freight_cost": "Average freight (USD)"})
        fig.update_traces(marker_color="#82aaff", marker_line_width=0)
        st.plotly_chart(chart_style(fig, 280), width="stretch")
    with right:
        st.subheader("Largest vendors")
        vendors = filtered.groupby("vendor_name", as_index=False).agg(invoice_value=("invoice_amount", "sum"), invoices=("invoice_id", "count"))
        vendors = vendors.sort_values("invoice_value", ascending=False).head(8)
        vendors["invoice_value"] = vendors["invoice_value"].round(2)
        st.dataframe(vendors.rename(columns={"vendor_name": "Vendor", "invoice_value": "Invoice value (USD)", "invoices": "Invoices"}), hide_index=True, width="stretch")
    st.subheader("Invoice explorer")
    columns = ["invoice_id", "vendor_name", "region", "invoice_date", "invoice_amount", "freight_cost", "shipping_mode"]
    st.dataframe(filtered[columns].head(250), hide_index=True, width="stretch")
    st.caption(f"Showing up to 250 of {len(filtered):,} matching records. All records and model outcomes are synthetic demo data; currency is USD.")
    st.download_button("Download filtered invoices", safe_csv(filtered), "demo_invoices.csv", "text/csv")


def freight_estimator(bundle):
    page_heading("PLAN WITH CONTEXT", "Estimate the cost of a shipment.", "Enter shipment details to estimate freight using the demo regression model.")
    with st.form("freight_form"):
        c1, c2, c3 = st.columns(3)
        region = c1.selectbox("Region", REGIONS, key="f_region")
        mode = c2.selectbox("Shipping mode", MODES, key="f_mode")
        quantity = c3.number_input("Quantity", min_value=1, max_value=1000000, value=50, step=1, key="f_quantity")
        weight = c1.number_input("Weight (kg)", min_value=0.01, max_value=1000000.0, value=125.0, step=5.0, key="f_weight")
        distance = c2.number_input("Distance (km)", min_value=0.1, max_value=50000.0, value=420.0, step=10.0, key="f_distance")
        amount = c3.number_input("Invoice amount (USD)", min_value=0.01, max_value=100000000.0, value=2500.0, step=100.0, key="f_amount")
        line_count = c1.number_input("Line item count", min_value=1, max_value=10000, value=3, key="f_lines")
        submitted = st.form_submit_button("Estimate freight", type="primary")
    if submitted:
        record = default_record()
        record.update(region=region, shipping_mode=mode, quantity=int(quantity), weight_kg=float(weight), distance_km=float(distance), invoice_amount=float(amount), line_total=float(amount), line_count=int(line_count))
        try:
            result = analyze_invoice(record, bundle)
            st.session_state["freight_result"] = result
        except (ValueError, TypeError) as exc:
            st.session_state.pop("freight_result", None)
            st.error(str(exc))
    result = st.session_state.get("freight_result")
    if result:
        st.metric("Estimated freight cost", money(result["expected_freight"]))
        st.caption("A point estimate in USD, not a carrier quote. Accuracy is measured on held-out synthetic invoices.")
        for warning in result.get("warnings", []):
            st.warning(warning)
    with st.expander("What influences this estimate?"):
        st.write("The model learns relationships between shipment and invoice details and historical freight cost. Region and shipping mode are encoded automatically. Review the model report for measured errors and feature importance.")


def invoice_review(bundle):
    page_heading("CONFIDENCE BEFORE APPROVAL", "Give every invoice a closer look.", "Compare declared freight with an estimate, check line totals, and screen for review signals.")
    with st.form("invoice_form"):
        st.markdown("**Invoice details**")
        c1, c2, c3 = st.columns(3)
        invoice_id = c1.text_input("Invoice reference", value="MANUAL-001", key="i_reference")
        vendor_name = c2.text_input("Vendor name", value="Sample vendor", key="i_vendor")
        invoice_date = c3.date_input("Invoice date", value=date.today(), key="i_date")
        region = c1.selectbox("Region", REGIONS, key="i_region")
        mode = c2.selectbox("Shipping mode", MODES, key="i_mode")
        terms = c3.number_input("Payment terms (days)", min_value=0, max_value=365, value=30, step=1, key="i_terms")
        st.markdown("**Amounts and shipment**")
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input("Invoice amount (USD)", min_value=0.01, max_value=100000000.0, value=2500.0, step=100.0, key="i_amount")
        line_total = c2.number_input("Line item total (USD)", min_value=0.01, max_value=100000000.0, value=2500.0, step=100.0, key="i_total")
        freight = c3.number_input("Declared freight (USD)", min_value=0.0, max_value=10000000.0, value=145.0, step=10.0, key="i_freight")
        quantity = c1.number_input("Quantity", min_value=1, max_value=1000000, value=50, step=1, key="i_quantity")
        weight = c2.number_input("Weight (kg)", min_value=0.01, max_value=1000000.0, value=125.0, step=5.0, key="i_weight")
        distance = c3.number_input("Distance (km)", min_value=0.1, max_value=50000.0, value=420.0, step=10.0, key="i_distance")
        line_count = c1.number_input("Line item count", min_value=1, max_value=10000, value=8, step=1, key="i_lines")
        submitted = st.form_submit_button("Analyze invoice", type="primary")
    if submitted:
        record = dict(invoice_id=invoice_id.strip(), vendor_name=vendor_name.strip(), invoice_date=invoice_date.isoformat(), region=region, shipping_mode=mode, payment_terms_days=int(terms), invoice_amount=float(amount), line_total=float(line_total), freight_cost=float(freight), quantity=int(quantity), weight_kg=float(weight), distance_km=float(distance), line_count=int(line_count))
        try:
            if not record["invoice_id"] or not record["vendor_name"]:
                raise ValueError("Enter an invoice reference and vendor name.")
            result = analyze_invoice(record, bundle)
            st.session_state["invoice_result"] = result
            st.session_state["invoice_record"] = record
            st.session_state.pop("invoice_saved_id", None)
        except (ValueError, TypeError) as exc:
            st.session_state.pop("invoice_result", None)
            st.error(str(exc))
    result = st.session_state.get("invoice_result")
    record = st.session_state.get("invoice_record")
    if result and record:
        show_review_result(record, result)
        st.caption("These findings refer to the last analyzed values. Submit again after editing the form.")
        saved_id = st.session_state.get("invoice_saved_id")
        if st.button("Save to review history", disabled=saved_id is not None):
            try:
                saved_id = save_review(record, result)
                st.session_state["invoice_saved_id"] = saved_id
            except (ValueError, TypeError) as exc:
                st.error(str(exc))
        if saved_id is not None:
            st.success(f"Saved to review history (record {saved_id}).")


def batch_analysis(data, bundle):
    page_heading("ONE FILE. A CLEARER PICTURE.", "Review invoices in a batch.", "Upload a CSV to get a result for each row, or explore an example with deliberately unusual invoices.")
    fields = ["invoice_id", "vendor_name", "region", "invoice_date", "quantity", "invoice_amount", "weight_kg", "distance_km", "shipping_mode", "freight_cost", "line_total", "line_count", "payment_terms_days"]
    template = pd.DataFrame([default_record()])[fields]
    st.download_button("Download CSV template", safe_csv(template), "invoice_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload invoice CSV", type=["csv"], help="Use the downloaded template for column names. Amounts are in USD.")
    left, right = st.columns([1, 2])
    run_upload = left.button("Analyze uploaded CSV", type="primary", disabled=uploaded is None)
    run_example = right.button("Run example batch")
    if run_upload or run_example:
        try:
            if run_example:
                frame = data[fields].head(8).copy() if not data.empty else template.copy()
                frame["invoice_id"] = [f"EXAMPLE-{i + 1:03d}" for i in range(len(frame))]
                if len(frame):
                    frame.loc[frame.index[0], "freight_cost"] = max(float(frame.iloc[0]["freight_cost"]) * 4, 2500.0)
                if len(frame) > 1:
                    frame.loc[frame.index[1], "invoice_amount"] = float(frame.iloc[1]["line_total"]) * 1.35
            else:
                frame = read_batch_csv(uploaded.getvalue())
            with st.spinner("Analyzing invoices…"):
                st.session_state["batch_result"] = analyze_batch(frame, bundle)
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
            st.session_state.pop("batch_result", None)
            st.error(str(exc))
    results = st.session_state.get("batch_result")
    if results is not None:
        st.subheader("Batch results")
        errors = results["error"].fillna("").astype(str).str.strip().ne("") if "error" in results else pd.Series(False, index=results.index)
        flags = results["decision"].eq("Review required") if "decision" in results else pd.Series(False, index=results.index)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows submitted", f"{len(results):,}")
        c2.metric("Review required", f"{int(flags.sum()):,}")
        c3.metric("Invalid rows", f"{int(errors.sum()):,}")
        st.dataframe(results, hide_index=True, width="stretch")
        st.download_button("Download analysis results", safe_csv(results), "invoice_analysis.csv", "text/csv")
        st.caption("Invalid rows are retained with an error. Batch analysis does not add entries to review history. Risk scores are uncalibrated demo outputs.")
    with st.expander("CSV format and checks"):
        st.write("Required columns: quantity, invoice_amount, weight_kg, distance_km, freight_cost, line_total. Optional defaults: line_count=3, shipping_mode=Standard, region=North, payment_terms_days=30. Invoice references and vendor names are optional batch metadata. Maximum 5,000 rows; invalid rows remain in the output.")
        st.write(f"Regions: {', '.join(REGIONS)}. Shipping modes: {', '.join(MODES)}.")


def review_history():
    page_heading("YOUR REVIEW TRAIL", "Keep the context behind each decision.", "Saved manual reviews stay in your local SQLite database and can be exported as CSV.")
    history = load_history()
    if history.empty:
        st.info("No reviews saved yet. Analyze an invoice and select ‘Save to review history’ to create your first entry.")
        return
    st.caption("Showing up to 1,000 most recent saved reviews.")
    c1, c2 = st.columns(2)
    c1.metric("Saved reviews", f"{len(history):,}")
    if "decision" in history:
        c2.metric("Review required", f"{int(history['decision'].eq('Review required').sum()):,}")
    selected = st.selectbox("Show decisions", ["All decisions", "Review required", "No flags detected"])
    visible = history if selected == "All decisions" or "decision" not in history else history.loc[history["decision"] == selected]
    st.dataframe(visible, hide_index=True, width="stretch")
    st.download_button("Download review history", safe_csv(visible), "review_history.csv", "text/csv")


def importance_chart(rows, key):
    if not rows:
        st.info("Feature importance is unavailable.")
        return
    frame = pd.DataFrame(rows)
    if "feature" not in frame:
        st.dataframe(frame, hide_index=True, width="stretch")
        return
    value_column = "importance" if "importance" in frame else next((c for c in frame if c != "feature"), None)
    if value_column is None:
        return
    frame = frame.sort_values(value_column).tail(10)
    fig = px.bar(frame, x=value_column, y="feature", orientation="h", labels={value_column: "Importance", "feature": ""})
    fig.update_traces(marker_color="#61dfbe" if key == "freight" else "#82aaff")
    st.plotly_chart(chart_style(fig, 350), width="stretch", key=f"importance_{key}")


def model_report(metrics):
    page_heading("KNOW THE LIMITS", "Transparent models. Measured performance.", "Inspect the holdout results, training scope, and signals behind each prediction.")
    st.info("This project uses reproducible synthetic invoices. These metrics describe demo data and do not establish performance on real vendor invoices. Risk scores are uncalibrated.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset rows", f"{int(metrics.get('rows', 0)):,}")
    c2.metric("Training rows", f"{int(metrics.get('train_rows', 0)):,}")
    c3.metric("Holdout rows", f"{int(metrics.get('holdout_rows', 0)):,}")
    st.caption(f"Split: {metrics.get('split', 'See model metadata')} · Trained at: {metrics.get('trained_at_utc', 'Unknown')} · scikit-learn: {metrics.get('sklearn_version', 'Unknown')}")
    freight = metrics.get("freight", {})
    risk = metrics.get("risk", {})
    regression_tab, risk_tab, scope_tab = st.tabs(["Freight regression", "Risk classification", "Data & scope"])
    with regression_tab:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", metric_number(freight.get("r2")))
        c2.metric("Mean absolute error", money(freight["mae"]) if "mae" in freight else "—")
        c3.metric("Root mean square error", money(freight["rmse"]) if "rmse" in freight else "—")
        c4.metric("Baseline MAE", money(freight["baseline_mae"]) if "baseline_mae" in freight else "—")
        st.caption("Error amounts are USD. MAE is the average absolute error on the holdout set; the baseline provides a simple comparison.")
        examples = pd.DataFrame(metrics.get("regression_examples", []))
        if not examples.empty and {"actual", "predicted"}.issubset(examples.columns):
            st.subheader("Actual vs. estimated freight")
            fig = px.scatter(examples, x="actual", y="predicted", labels={"actual": "Actual freight (USD)", "predicted": "Estimated freight (USD)"}, opacity=0.65)
            low = float(examples[["actual", "predicted"]].min().min())
            high = float(examples[["actual", "predicted"]].max().max())
            fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect estimate", line=dict(color="#eabf78", dash="dash")))
            st.plotly_chart(chart_style(fig), width="stretch")
            st.caption("A sample of held-out normal invoices. The dashed line marks an exact estimate.")
        st.subheader("Feature importance")
        importance_chart(freight.get("feature_importance", []), "freight")
    with risk_tab:
        c1, c2, c3, c4, c5 = st.columns(5)
        for column, label, field in zip([c1, c2, c3, c4, c5], ["Accuracy", "Precision", "Recall", "F1 score", "ROC AUC"], ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            column.metric(label, metric_number(risk.get(field)))
        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            matrix = risk.get("confusion_matrix")
            if matrix is not None:
                fig = go.Figure(go.Heatmap(z=matrix, x=["Predicted clear", "Predicted review"], y=["Actual clear", "Actual review"], colorscale=[[0, "#152337"], [1, "#61dfbe"]], showscale=False, text=matrix, texttemplate="%{text}"))
                st.plotly_chart(chart_style(fig, 350), width="stretch")
            else:
                st.info("Confusion matrix unavailable.")
        with right:
            st.subheader("Feature importance")
            importance_chart(risk.get("feature_importance", []), "risk")
        st.caption("Classification metrics evaluate the learned synthetic review label. Deterministic invoice rules can also flag invoices independently.")
    with scope_tab:
        st.write("**Data source:**", metrics.get("data_source", "Reproducible synthetic demonstration data"))
        st.write("**Currency:** USD")
        st.write("Use the training ranges below to understand where the model has examples. Inputs outside that scope may produce less reliable estimates.")
        ranges = metrics.get("training_ranges", {})
        if isinstance(ranges, dict) and ranges:
            try:
                frame = pd.DataFrame(ranges).T.rename_axis("Feature").rename(columns={0: "Minimum", 1: "Maximum"})
                st.dataframe(frame, width="stretch")
            except ValueError:
                st.json(ranges)
        else:
            st.json(ranges)
        with st.expander("Local project files"):
            st.code(f"Project: {ROOT}\nDatabase: {DB_PATH}\nModels: {MODEL_DIR}", language="text")
        st.write("Connect validated real data and evaluate on a representative holdout set before using this demo to support operational decisions.")


with st.sidebar:
    st.markdown('<div class="brand">◈ Vendor<span>IQ</span></div>', unsafe_allow_html=True)
    st.caption("INVOICE INTELLIGENCE")
    st.markdown("---")
    page = st.radio("Workspace", ["Overview", "Freight estimator", "Invoice review", "Batch analysis", "Review history", "Model report"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="eyebrow">LOCAL WORKSPACE</div><div class="muted">SQLite data · Saved reviews<br>Freight estimation · Risk screening</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("DEMO DATA · USD\n\nSynthetic invoices and model outcomes. No live vendor or carrier connections.")

st.markdown('<div class="topline"><span>WORKSPACE / INVOICE OPERATIONS</span><span class="pill">● Synthetic demo</span></div>', unsafe_allow_html=True)

try:
    bundle = cached_bundle()
    data = cached_invoices()
except Exception as exc:
    st.error(f"The demo could not initialize: {exc}")
    st.info("Check that dependencies are installed and the project data and model directories are writable, then restart the app.")
    st.stop()

if page == "Overview":
    overview(data, bundle.get("metrics", {}))
elif page == "Freight estimator":
    freight_estimator(bundle)
elif page == "Invoice review":
    invoice_review(bundle)
elif page == "Batch analysis":
    batch_analysis(data, bundle)
elif page == "Review history":
    review_history()
elif page == "Model report":
    model_report(bundle.get("metrics", {}))
