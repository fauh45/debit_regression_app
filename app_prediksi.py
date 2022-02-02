import streamlit as st
import pandas as pd
import numpy as np
import joblib
import keras

def app():
    def to_supervised(df, backward=5, forward=3):
        data, data_names = list(), list()
       
        src_df = df
        for i in range(backward, 0, -1):
           data.append(src_df.shift(i))
           data_names += [('%s(t-%d)' % (label, i)) for label in src_df.columns]

        agg_data = pd.concat(data, axis=1)
        agg_data.columns = data_names
        agg_data.dropna(inplace=True)

        for i in range(0, forward):
            if i == 0:
                agg_data["debit_dayeuh_kolot(t)"] = np.float64(0)
            else:
                agg_data["debit_dayeuh_kolot(t+%d)" % (i)] = np.float64(0)
        
        return agg_data
    
    st.title("Prediksi Model Regresi LSTM")
    st.markdown("Pastikan data csv yang dimasukan memiliki jumlah kolom, dan nama yang sama dengan csv contoh di [google drive](https://drive.google.com/file/d/18uV7VIjBp7NIoeG0cQbMvdTprMGLE_BO/view?usp=sharing).")

    choice = st.selectbox("Pilih jenis model", ["Prediksi 2 jam ke depan", "Prediksi 4 jam ke depan"])
    source_data = st.file_uploader("Upload file data", "csv", accept_multiple_files=False)

    backwards_period = 24
    forward_period = 2 if choice == "Prediksi 2 jam ke depan" else 4

    N_features = 14

    if source_data is not None:
        with st.spinner("Menghitung prediksi dari data yang diberikan..."):
            df = pd.read_csv(source_data, sep=None)

            df_timestamp = df['timestamp']
            df.drop(columns=['timestamp'], inplace=True)

            df.astype(np.float64)
            df['timestamp'] = pd.to_datetime(df_timestamp)

            df.set_index('timestamp', inplace=True)

            st.write("Sumber data sebelum di proses, 24 terakhir", df.tail(25))

            data = to_supervised(df.tail(25), backwards_period, forward_period)

            st.write(data)
            st.text("Abaikan kolom debit_dayeuh_kolot(t+n) karena itu yang akan dilakukan prediksi")

            values = data.values

            scaler = joblib.load(f"data_scaler_{forward_period}.gz")
            values = scaler.transform(values)

            n_obs = N_features * backwards_period

            validation_X, _ = values[:, :n_obs], values[:, -forward_period:]
            validation_X = validation_X.reshape((validation_X.shape[0], backwards_period, N_features))

            model = keras.models.load_model(f"models/{'debit-regression-24b-2f.h5' if forward_period == 2 else 'debit-regression-24b-4f.h5'}")

            yhat = model.predict(validation_X)

            inv_validation_X = validation_X.reshape((validation_X.shape[0], backwards_period * N_features))

            inv_yhat = np.concatenate((inv_validation_X, yhat), axis=1)
            inv_yhat = scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:, -forward_period:]

            st.header("Hasil Prediksi")
            st.write(pd.DataFrame(inv_yhat, columns=["Debit Dayeuh Kolot (t)" if i == 0 else "Debit Dayeuh Kolot (t+%d)" % i for i in range(0, forward_period)]))

        st.success("Selesai melakukan prediksi dari data yang diberikan")