from cmath import sqrt
from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import keras

def to_supervised(backward=5, forward=3):
  global df
  
  data, data_names = list(), list()

  src_df_label = df["debit_dayeuh_kolot"]
  src_df = df

  for i in range(backward, 0, -1):
    data.append(src_df.shift(i))
    data_names += [('%s(t-%d)' % (label, i)) for label in src_df.columns]

  for i in range(0, forward):
    data.append(src_df_label.shift(-i))

    if i == 0:
      data_names.append("debit_dayeuh_kolot(t)")
    else:
      data_names.append("debit_dayeuh_kolot(t+%d)" % (i))

  agg_data = pd.concat(data, axis=1)
  agg_data.columns = data_names
  agg_data.dropna(inplace=True)

  return agg_data

st.set_page_config(page_title="Evaluasi Model")
st.title("Evaluasi Model Regresi LSTM")
st.markdown("Pastikan data csv yang dimasukan memiliki jumlah kolom, dan nama yang sama dengan csv contoh di [google drive](https://drive.google.com/file/d/18uV7VIjBp7NIoeG0cQbMvdTprMGLE_BO/view?usp=sharing).")

choice = st.selectbox("Pilih jenis model", ["Prediksi 2 jam ke depan", "Prediksi 4 jam ke depan"])
source_data = st.file_uploader("Upload file data", "csv", accept_multiple_files=False)

backwards_period = 24
forward_period = 2 if choice == "Prediksi 2 jam ke depan" else 4

N_features = 14

if source_data is not None:
    with st.spinner("Menghitung prediksi dari data validasi..."):
        df = pd.read_csv(source_data, sep=None)

        df_timestamp = df['timestamp']
        df.drop(columns=['timestamp'], inplace=True)

        df.astype(np.float64)
        df['timestamp'] = pd.to_datetime(df_timestamp)

        df.set_index('timestamp', inplace=True)

        st.write("Sumber data sebelum di proses, 5 paling awal,", df.head())

        data = to_supervised(backwards_period, forward_period)
        values = data.values

        scaler = joblib.load("data_scaler.gz")
        values = scaler.fit_transform(values)

        n_obs = N_features * backwards_period

        validation_X, validation_Y = values[:, :n_obs], values[:, -forward_period:]
        validation_X = validation_X.reshape((validation_X.shape[0], backwards_period, N_features))

        model = keras.models.load_model(f"models/{'debit-regression-24b-2f.h5' if forward_period == 2 else 'debit-regression-24b-4f.h5'}")

        yhat = model.predict(validation_X)

        inv_validation_X = validation_X.reshape((validation_X.shape[0], backwards_period * N_features))

        inv_yhat = np.concatenate((inv_validation_X, yhat), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, -forward_period:]

        inv_validation_Y = validation_Y.reshape((len(validation_Y), forward_period))

        inv_y = np.concatenate((inv_validation_X, inv_validation_Y), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, -forward_period:]

        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

        st.write("RMSE : ", rmse)

        for i in range(1, forward_period + 1):
            t_df_temp = pd.DataFrame({f"Actual (t+{i})": inv_y[:, i - 1], f"Predicted (t+{i})": inv_yhat[:, i - 1]})

            st.header(f"Hasil prediksi t+{i}")
            st.line_chart(t_df_temp)

            rmse_temp = sqrt(mean_squared_error(inv_y[:, i - 1], inv_yhat[:, i - 1]))
            st.write(f"RMSE t+{i} :", rmse_temp)

    st.success("Selesai melakukan prediksi dari data validasi")