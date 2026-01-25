import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# --- 1. SETUP MODEL ---
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

@st.cache_resource
def load_my_model():
    model = LSTMModel()
    model.load_state_dict(torch.load('models/lstm_stock_model.pth'))
    model.eval()
    return model

# --- 2. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Stock Predictor", page_icon="📈")
st.title("📈 AI Multi-Stock Predictor")

# --- 3. SIDEBAR: INPUT & PILIHAN SAHAM ---
st.sidebar.header("⚙️ Konfigurasi")

# Pilihan saham populer (IHSG & Global)
popular_tickers = ["BBRI.JK", "BBCA.JK", "TLKM.JK", "ASII.JK", "GOTO.JK", "AAPL", "TSLA", "BTC-USD"]
selected_ticker = st.sidebar.selectbox("Pilih Saham Populer:", popular_tickers)

# Input manual untuk fleksibilitas
user_ticker = st.sidebar.text_input("Atau ketik Ticker lain (Yahoo Finance):", "")

# Logika penentuan ticker: Prioritaskan input manual
ticker = user_ticker.upper() if user_ticker else selected_ticker

st.sidebar.info(f"Ticker Aktif: **{ticker}**")
predict_button = st.button(f"Prediksi {ticker}")

# --- 4. LOGIKA PREDIKSI ---
if predict_button:
    with st.spinner(f'Mengambil data {ticker}...'):
        # 1. Ambil Data (1 Tahun terakhir untuk konteks)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            st.error("Ticker tidak ditemukan. Pastikan kode benar (contoh: BBRI.JK atau AAPL).")
        else:
            # Bersihkan MultiIndex jika ada
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

            # 4. Tampilkan Metrik Utama
            last_price = df['Close'].values[-1]
            diff = pred_price - last_price
            
            st.subheader(f"Hasil Analisis: {ticker}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Harga Terakhir", f"{last_price:,.2f}")
            col2.metric("Prediksi Besok", f"{pred_price:,.2f}", f"{diff:,.2f}")
            
            status = "BULLISH 🚀" if diff > 0 else "BEARISH 🔻"
            col3.markdown(f"**Sentimen:**\n### {status}")

            # 5. Dashboard Statistik 1 Tahun
            st.divider()
            st.subheader(f"📊 Tren Harga 1 Tahun Terakhir")
            st.line_chart(df['Close'])

            # Statistik Tambahan
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Tertinggi (1th)", f"{df['Close'].max():,.2f}")
            col_b.metric("Terendah (1th)", f"{df['Close'].min():,.2f}")
            col_c.metric("Rata-rata", f"{df['Close'].mean():,.2f}")

st.divider()
st.info("💡 **Tips:** Untuk saham Indonesia, gunakan akhiran `.JK` (Contoh: ASII.JK). Untuk kripto, gunakan `-USD` (Contoh: ETH-USD).")