import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --- 1. SETUP MODEL (SAMA DENGAN FILE 03) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- 2. FUNGSI UNTUK LOAD MODEL ---
@st.cache_resource # Biar ga loading ulang terus tiap klik tombol
def load_my_model():
    model = LSTMModel()
    model.load_state_dict(torch.load('models/lstm_stock_model.pth'))
    model.eval()
    return model

# --- 3. TAMPILAN WEBSITE ---
st.set_page_config(page_title="AI Saham BBRI", page_icon="💰")
st.title("💰 BBRI Stock Predictor")
st.markdown("Aplikasi ini memprediksi harga saham **BBRI** menggunakan model **LSTM Deep Learning**.")

# Input ticker di Sidebar
ticker = st.sidebar.text_input("Ticker Saham", value="BBRI.JK")
predict_button = st.button("Mulai Prediksi")

if predict_button:
    with st.spinner('Sedang mengambil data terbaru...'):
        # 1. Ambil Data Real-time
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Bersihkan data
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.filter(['Close']).dropna()

        # 2. Preprocessing
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df.values)
        last_60_days = scaled_data[-60:]
        X_input = torch.from_numpy(last_60_days).type(torch.Tensor).unsqueeze(0)

        # 3. Prediksi
        model = load_my_model()
        with torch.no_grad():
            pred_scaled = model(X_input).numpy()
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]

        # 4. Tampilkan Hasil
        last_price = df['Close'].values[-1]
        diff = pred_price - last_price
        
        col1, col2 = st.columns(2)
        col1.metric("Harga Terakhir", f"Rp {last_price:,.0f}")
        col2.metric("Prediksi Besok", f"Rp {pred_price:,.0f}", f"{diff:,.2f}")

        if diff > 0:
            st.success("Kesimpulan: **BULLISH** 🚀")
        else:
            st.error("Kesimpulan: **BEARISH** 🔻")

        # 5. Grafik Mini
        st.subheader("Grafik Harga Penutupan Terakhir")
        st.line_chart(df.tail(30))

st.info("Catatan: Ini adalah proyek riset mahasiswa Informatika. Jangan dijadikan patokan mutlak investasi.")