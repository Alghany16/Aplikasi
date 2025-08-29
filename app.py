import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
import requests

# ---------------------------- CONFIG ----------------------------
st.set_page_config(page_title="Prediksi Harga Saham - GRU", layout="centered")
st.title("📈 Prediksi Harga Saham 7 Hari ke Depan (GRU Model)")

# ---------------------------- FUNGSI API ----------------------------
api_key = "9eb96b3ae91449639fc1eb15a25230b4"  # API key tetap

def download_stock_data(symbol, start_date, end_date):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&start_date={start_date}&end_date={end_date}&apikey={api_key}&format=JSON"
    try:
        response = requests.get(url)
        data = response.json()
        if 'values' not in data:
            return None
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df['close'] = df['close'].astype(float)
        df.sort_index(inplace=True)
        return df[['close']]
    except Exception as e:
        return None

# ---------------------------- FUNGSI PREDIKSI ----------------------------
def predict_next_7_days(df_close, model_path="gru_model.keras"):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_close.values.reshape(-1, 1))
    x_input = scaled_data[-60:].reshape(1, 60, 1)
    model = load_model(model_path)

    predictions_scaled = []
    current_input = x_input.copy()
    for _ in range(7):
        pred_scaled = model.predict(current_input, verbose=0)
        predictions_scaled.append(pred_scaled[0, 0])
        next_input = pred_scaled[0, :1].reshape(1, 1, 1)
        current_input = np.concatenate((current_input[:, 1:, :], next_input), axis=1)

    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    return predictions

# ---------------------------- FUNGSI EVALUASI ----------------------------
def evaluate_prediction(df_actual, predicted_prices):
    df_actual.index = pd.to_datetime(df_actual.index)
    n = min(len(df_actual), len(predicted_prices))
    df_terakhir = df_actual.tail(n)
    aktual = df_terakhir['close'].values[:n]
    prediksi = predicted_prices[:n]

    mae = mean_absolute_error(aktual, prediksi)
    rmse = np.sqrt(mean_squared_error(aktual, prediksi))

    df_eval = df_terakhir.copy()
    df_eval = df_eval.assign(Prediksi=prediksi)
    df_eval['Error'] = abs(df_eval['close'] - df_eval['Prediksi'])
    df_eval['% Error'] = 100 * df_eval['Error'] / df_eval['close']

    return df_eval, mae, rmse

# ---------------------------- INPUT UTAMA ----------------------------
symbol = st.text_input("Masukkan simbol saham (Contoh: AAPL, BBCA.JK)", value="AAPL").upper()
tanggal_akhir = st.date_input("Tanggal akhir data (untuk training)", value=datetime.date(2025, 4, 30))
start_train = "2024-01-01"

if st.button("🔍 Jalankan Prediksi"):
    df = download_stock_data(symbol, start_train, tanggal_akhir.strftime("%Y-%m-%d"))

    if df is not None and len(df) >= 60:
        st.subheader(f"📉 Data Harga Saham: {symbol}")
        st.line_chart(df['close'])

        predicted = predict_next_7_days(df['close'])
        start_date = tanggal_akhir
        future_dates = [start_date + datetime.timedelta(days=i+1) for i in range(7)]
        df_pred = pd.DataFrame({'Tanggal': future_dates, 'Prediksi': predicted})

        st.subheader("📅 Prediksi Harga 7 Hari ke Depan")
        st.table(df_pred.set_index('Tanggal').style.format("{:.3f}"))

        # Ambil data aktual jika tersedia (untuk evaluasi)
        end_eval = (tanggal_akhir + datetime.timedelta(days=10)).strftime("%Y-%m-%d")
        df_actual = download_stock_data(symbol, (tanggal_akhir + datetime.timedelta(days=1)).strftime("%Y-%m-%d"), end_eval)

        if df_actual is not None and not df_actual.empty:
            df_eval, mae, rmse = evaluate_prediction(df_actual, predicted)

            st.subheader("📊 Evaluasi Prediksi vs Aktual")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_eval.index, y=df_eval['close'], name='Aktual', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=df_eval.index, y=df_eval['Prediksi'], name='Prediksi', mode='lines+markers'))
            fig.update_layout(title='Perbandingan Prediksi vs Aktual', xaxis_title='Tanggal', yaxis_title='Harga')
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"MAE: {mae:.2f} | RMSE: {rmse:.2f}")
        else:
            st.warning("⚠️ Data aktual tidak tersedia untuk evaluasi.")
    else:
        st.error("❌ Gagal mengambil data dari API atau data kurang dari 60 hari.")
