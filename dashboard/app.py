import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import io, tempfile
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage

st.set_page_config(page_title="📈 Stock Forecasting Dashboard", layout="wide")
st.title("📊 Stock Price Forecasting App")
st.markdown("Predict with **SARIMA**, **Prophet**, **LSTM**; compare models, download results, and export PDF reports.")

# Sidebar
st.sidebar.header("⚙️ Settings")
end_date = date.today()
start_date = end_date - timedelta(days=5*365)
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
mode = st.sidebar.radio("Mode", ("Single Model", "Compare Models"))
forecast_period = st.sidebar.slider("Forecast Period (Days)", 30, 365, 90)
if mode == "Single Model":
    model_choice = st.sidebar.selectbox("Select Forecasting Model", ("SARIMA","Prophet","LSTM"))
else:
    model_choice = st.sidebar.multiselect("Select Models to Compare", ["SARIMA","Prophet","LSTM"], default=["SARIMA","Prophet"])

# Data
@st.cache_data
def load_data(ticker, start, end, fallback_csv):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df is None or df.empty:
            raise RuntimeError("Empty data")
        return df.reset_index(), "live"
    except Exception:
        df = pd.read_csv(fallback_csv, parse_dates=["Date"])
        return df, "sample"

df, source = load_data(ticker, start_date, end_date, "data/raw/sample_AAPL.csv")
st.caption(f"Data source: {'Yahoo Finance' if source=='live' else 'Local Sample CSV'}")

# Plot
def plot_stock(df, name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close"))
    fig.update_layout(title=f"{name} Stock Price (Historical)", xaxis_title="Date", yaxis_title="Price (USD)")
    return fig
st.plotly_chart(plot_stock(df, ticker), use_container_width=True)

# Models
def sarima_forecast(df, steps):
    d = df.set_index("Date")["Close"]
    model = SARIMAX(d, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=steps)
    out = fc.conf_int()
    out["Forecast"] = fc.predicted_mean
    out.index.name = "Date"
    return out

def prophet_forecast(df, steps):
    d = df[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"})
    m = Prophet(daily_seasonality=True)
    m.fit(d)
    future = m.make_future_dataframe(periods=steps)
    fc = m.predict(future)
    return fc, d

def lstm_forecast(df, steps, look_back=60):
    series = df[["Date","Close"]].set_index("Date").values
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(series)
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i,0]); y.append(scaled[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential([LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)), LSTM(50), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    seq = scaled[-look_back:].reshape(1, look_back, 1)
    preds = []
    for _ in range(steps):
        p = model.predict(seq, verbose=0)[0][0]
        preds.append(p)
        seq = np.append(seq[:,1:,:], [[[p]]], axis=1)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.DataFrame({"Date": future_dates, "Forecast": preds})

# Evaluation
def evaluate_model(df, model_name):
    split_idx = int(len(df)*0.8)
    train, test = df[:split_idx], df[split_idx:]
    if model_name == "SARIMA":
        fc = sarima_forecast(train, len(test)).reset_index()
        y_pred = fc["Forecast"].values; y_true = test["Close"].values
    elif model_name == "Prophet":
        fc, d = prophet_forecast(train, len(test))
        fc = fc.set_index("ds").loc[test["Date"]]
        y_pred = fc["yhat"].values; y_true = test["Close"].values
    elif model_name == "LSTM":
        fc = lstm_forecast(train, len(test)).set_index("Date").loc[test["Date"]]
        y_pred = fc["Forecast"].values; y_true = test["Close"].values
    else:
        return None
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return dict(Model=model_name, RMSE=rmse, MAPE=mape, R2=r2)

# UI
if mode == "Single Model":
    if st.sidebar.button("Run Forecast"):
        if model_choice == "SARIMA":
            st.subheader(f"🔮 {ticker} Forecast ({forecast_period} days) using SARIMA")
            fc = sarima_forecast(df, forecast_period)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Historical"))
            fig.add_trace(go.Scatter(x=fc.index, y=fc["Forecast"], name="Forecast"))
            st.plotly_chart(fig, use_container_width=True)
        elif model_choice == "Prophet":
            st.subheader(f"🔮 {ticker} Forecast ({forecast_period} days) using Prophet")
            fc, d = prophet_forecast(df, forecast_period)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d["ds"], y=d["y"], name="Historical"))
            fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast"))
            st.plotly_chart(fig, use_container_width=True)
        elif model_choice == "LSTM":
            st.subheader(f"🔮 {ticker} Forecast ({forecast_period} days) using LSTM")
            fc = lstm_forecast(df, forecast_period)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Historical"))
            fig.add_trace(go.Scatter(x=fc["Date"], y=fc["Forecast"], name="Forecast"))
            st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader(f"📊 Comparing Forecasts for {ticker} ({forecast_period} days)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Historical"))
    metrics_results = []
    forecast_dfs = []

    if "SARIMA" in model_choice:
        sar = sarima_forecast(df, forecast_period).reset_index()[["Date","Forecast"]]
        sar["Model"] = "SARIMA"; forecast_dfs.append(sar)
        fig.add_trace(go.Scatter(x=sar["Date"], y=sar["Forecast"], name="SARIMA", line=dict(dash="dash")))
        metrics_results.append(evaluate_model(df, "SARIMA"))
    if "Prophet" in model_choice:
        pro, d = prophet_forecast(df, forecast_period)
        pro_df = pro.rename(columns={"ds":"Date","yhat":"Forecast"})[["Date","Forecast"]]
        pro_df["Model"] = "Prophet"; forecast_dfs.append(pro_df)
        fig.add_trace(go.Scatter(x=pro_df["Date"], y=pro_df["Forecast"], name="Prophet", line=dict(dash="dot")))
        metrics_results.append(evaluate_model(df, "Prophet"))
    if "LSTM" in model_choice:
        lstm_df = lstm_forecast(df, forecast_period)
        lstm_df["Model"] = "LSTM"; forecast_dfs.append(lstm_df)
        fig.add_trace(go.Scatter(x=lstm_df["Date"], y=lstm_df["Forecast"], name="LSTM", line=dict(dash="longdash")))
        metrics_results.append(evaluate_model(df, "LSTM"))

    fig.update_layout(title=f"{ticker} Forecast Comparison", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table + downloads
    if metrics_results:
        st.subheader("📈 Model Performance (80/20 split)")
        met_df = pd.DataFrame(metrics_results)
        st.dataframe(met_df.style.format({"RMSE":"{:.2f}","MAPE":"{:.2%}","R2":"{:.2f}"}))

        def to_csv(df): return df.to_csv(index=False).encode("utf-8")
        def to_excel(df):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Metrics")
            return buf.getvalue()

        st.download_button("⬇️ Download Metrics (CSV)", data=to_csv(met_df), file_name=f"{ticker}_metrics.csv", mime="text/csv")
        st.download_button("⬇️ Download Metrics (Excel)", data=to_excel(met_df), file_name=f"{ticker}_metrics.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Forecast table + downloads + PDF
    if forecast_dfs:
        st.subheader("🔮 Forecasted Values")
        all_fc = pd.concat(forecast_dfs, ignore_index=True)
        st.dataframe(all_fc)

        def to_csv(df): return df.to_csv(index=False).encode("utf-8")
        def to_excel(df):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Forecasts")
            return buf.getvalue()

        st.download_button("⬇️ Download Forecasts (CSV)", data=to_csv(all_fc), file_name=f"{ticker}_forecasts.csv", mime="text/csv")
        st.download_button("⬇️ Download Forecasts (Excel)", data=to_excel(all_fc), file_name=f"{ticker}_forecasts.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.subheader("📑 Generate PDF Report")
        def generate_pdf_report(ticker, metrics_df, forecasts_df, chart_fig):
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            doc = SimpleDocTemplate(tmpfile.name, pagesize=A4)
            elements = []
            styles = getSampleStyleSheet()
            title = Paragraph(f"<b>Stock Forecast Report: {ticker}</b>", styles["Title"])
            elements.append(title); elements.append(Spacer(1, 12))
            if metrics_df is not None and not metrics_df.empty:
                elements.append(Paragraph("<b>Model Performance Metrics</b>", styles["Heading2"]))
                mdat = [metrics_df.columns.tolist()] + metrics_df.round(3).values.tolist()
                table = Table(mdat, hAlign="LEFT")
                table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.grey),
                                           ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
                                           ("ALIGN",(0,0),(-1,-1),"CENTER"),
                                           ("GRID",(0,0),(-1,-1),0.5,colors.black)]))
                elements.append(table); elements.append(Spacer(1,12))
            if forecasts_df is not None and not forecasts_df.empty:
                elements.append(Paragraph("<b>Forecasted Values (Preview)</b>", styles["Heading2"]))
                prev = forecasts_df.head(20)
                fdat = [prev.columns.tolist()] + prev.values.tolist()
                ftable = Table(fdat, hAlign="LEFT")
                ftable.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightblue),
                                            ("ALIGN",(0,0),(-1,-1),"CENTER"),
                                            ("GRID",(0,0),(-1,-1),0.5,colors.black)]))
                elements.append(ftable); elements.append(Spacer(1,12))
            # Save fig as PNG using kaleido if available
            try:
                png_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                chart_fig.write_image(png_tmp, format="png")
                elements.append(Paragraph("<b>Forecast Comparison Chart</b>", styles["Heading2"]))
                elements.append(RLImage(png_tmp, width=400, height=250))
            except Exception:
                pass
            doc.build(elements)
            return tmpfile.name

        if st.button("📥 Generate PDF Report"):
            pdf_path = generate_pdf_report(ticker, met_df if metrics_results else pd.DataFrame(), all_fc, fig)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download PDF Report", data=f, file_name=f"{ticker}_forecast_report.pdf", mime="application/pdf")
