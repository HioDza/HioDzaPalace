import streamlit as st
from models.L2R import L2Regressor
from nexgml.gradient_supported import BasicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.set_page_config(page_title="Prediksi pola angka", layout='wide')
st.title("📈 Prediksi pola angka dari data")
st.caption("Ringan dan mudah untuk digunakan.")
st.subheader("📂 Unggah Dataset (CSV)")
if st.button("ℹ️ Petunjuk Penggunaan"):
    st.info("""
    1. Siapkan dataset dalam format CSV dengan kolom fitur dan satu kolom target (angka).
    2. Unggah file CSV untuk fitting model. Opsional: unggah file CSV lain untuk prediksi.
    3. Pilih kolom target dan fitur yang diinginkan.
    4. Klik 'Proses' untuk melatih model dan melihat hasil prediksi.
    5. Unduh hasil prediksi sebagai file CSV.
    
    **Catatan:**
    - Pastikan dataset tidak melebihi batas ukuran dan dimensi yang ditentukan.
    - Gunakan fitur yang relevan untuk hasil terbaik.
    - Setiap sesi dibatasi hingga 5 kali proses fitting untuk menjaga kinerja aplikasi.
    """)
st.info(f"⚠️ Batas ukuran file: 5 MB | Batas baris: 50,000 | Batas kolom: 50. Batasan diberikan demi menjaga kinerja aplikasi tetap optimal.")

uploaded_train_file = st.file_uploader("Pilih file CSV untuk fitting", type="csv")
uploaded_test_file = st.file_uploader("Pilih file CSV untuk prediksi (opsional)", type="csv")

# Initialize session counter in session state
if 'session_counter' not in st.session_state:
    st.session_state['session_counter'] = 0

if uploaded_train_file is not None:
    # Check file size (5 MB limit)
    if uploaded_train_file.size > 5 * 1024 * 1024:
        st.error("Ukuran file melebihi batas 5 MB. Silakan unggah file yang lebih kecil.")
        st.stop()

    import pandas as pd
    data_train = pd.read_csv(uploaded_train_file)

    # Check dataset dimensions (50,000 rows, 50 columns)
    if data_train.shape[0] > 50000:
        st.error("Dataset memiliki lebih dari 50,000 baris. Silakan unggah dataset yang lebih kecil.")
        st.stop()
    if data_train.shape[1] > 50:
        st.error("Dataset memiliki lebih dari 50 kolom. Silakan unggah dataset yang lebih kecil.")
        st.stop()

    data_test = uploaded_test_file and pd.read_csv(uploaded_test_file) or None
    if data_test is not None:
        if data_train.shape[1] != data_test.shape[1]:
            st.error("Dataset fit dan prediksi harus memiliki jumlah kolom yang sama.")
            st.stop()

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    st.write("Pratinjau dataset fitting:")
    st.dataframe(data_train.head())

    st.subheader("⚙️ Konfigurasi Fitur dan Target")
    all_columns = data_train.columns.tolist()
    target_column = st.selectbox("Pilih Kolom Target", all_columns)
    feature_columns = st.multiselect("Pilih Kolom Fitur", [col for col in all_columns if col != target_column])

    # If user doesn't pick features, automatically use all non-target columns
    if not feature_columns:
        feature_columns = [col for col in all_columns if col != target_column]
        st.info(f"Tidak ada fitur yang dipilih — menggunakan semua fitur non-target ({len(feature_columns)} kolom)")
    
    if st.button("Proses"):
      st.info(f"Sisa batas sesi: {5 - st.session_state['session_counter']}")

      if st.session_state['session_counter'] - 5 == 0:
        st.error("Batas sesi tercapai. Silakan muat ulang halaman untuk memulai sesi baru.")
        st.stop()

      else:

        if not feature_columns:
            st.error("Dimohon untuk memilih minimal 1 fitur data.")

        else:
            # Build feature matrix by detecting text columns and vectorizing them
            from pandas.api.types import is_string_dtype
            import pandas as pd

            parts = []
            vectorizers = st.session_state.get('vectorizers', {}) or {}
            encoders = st.session_state.get('encoders', {}) or {}

            for col in feature_columns:
                if is_string_dtype(data_train[col]) or data_train[col].dtype == object:
                    unique_vals = data_train[col].nunique()
                    if unique_vals <= 10:
                        # Categorical: One-Hot Encoding
                        enc = encoders.get(col)
                        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore') if enc is None else enc
                        X_cat = enc.fit_transform(data_train[[col]])
                        encoders[col] = enc
                        parts.append(X_cat)
                        
                    else:
                        # Text: TF-IDF
                        vec = vectorizers.get(col)
                        vec = TfidfVectorizer(max_features=300) if vec is None else vec
                        X_text = vec.fit_transform(data_train[col].astype(str)).toarray()
                        vectorizers[col] = vec
                        parts.append(X_text)
                else:
                    # Numeric: coerce to numeric and fill NaNs with 0
                    col_vals = pd.to_numeric(data_train[col], errors='coerce').fillna(0).values.reshape(-1, 1)
                    parts.append(col_vals)

            # save vectorizers and encoders for reuse (e.g., future prediction runs)
            st.session_state['vectorizers'] = vectorizers
            st.session_state['encoders'] = encoders

            # horizontally stack all parts into X
            X = np.hstack(parts)
            # ensure target is numeric where possible
            y = pd.to_numeric(data_train[target_column], errors='coerce').fillna(0).values
            
            X_train_full = scaler_x.fit_transform(X)
            y_train_full = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

            if data_test is not None:
                if target_column not in data_test.columns:
                    st.error("Kolom target tidak ditemukan di dataset prediksi.")
                    st.stop()
                # Use provided test data
                # Process test data features
                parts_test = []
                for col in feature_columns:
                    if is_string_dtype(data_test[col]) or data_test[col].dtype == object:
                        unique_vals = data_test[col].nunique()
                        if unique_vals <= 10:
                            # Use fitted encoder
                            enc = encoders.get(col)
                            if enc is not None:
                                X_cat_test = enc.transform(data_test[[col]])
                                parts_test.append(X_cat_test)
                        else:
                            # Use fitted vectorizer
                            vec = vectorizers.get(col)
                            if vec is not None:
                                X_text_test = vec.transform(data_test[col].astype(str)).toarray()
                                parts_test.append(X_text_test)
                    else:
                        col_vals_test = pd.to_numeric(data_test[col], errors='coerce').fillna(0).values.reshape(-1, 1)
                        parts_test.append(col_vals_test)

                X_test_full = np.hstack(parts_test)
                X_test_full = scaler_x.transform(X_test_full)
                y_test_full = pd.to_numeric(data_test[target_column], errors='coerce').fillna(0).values
                y_test_full = scaler_y.transform(y_test_full.reshape(-1, 1)).ravel()

                X_train, y_train = X_train_full, y_train_full
                X_test, y_test = X_test_full, y_test_full
            else:
                # Split train data
                X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
        
            models = L2Regressor()

            start_time = time.time()
            models.fit(X_train, y_train)
            y_pred = models.predict(X_test)
            y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
            elapsed = time.time() - start_time

            mse = mean_squared_error(y_test_original, y_pred_original)
            mae = mean_absolute_error(y_test_original, y_pred_original)
            r2 = r2_score(y_test_original.astype(np.float32), y_pred_original)

            st.subheader("📊 Metrik evaluasi")
            st.write(f"Rata-rata kuadrat kesalahan: {mse:.4f}")
            st.write(f"Rata-rata absolut kesalahan: {mae:.4f}")
            st.write(f"Skor R^2: {r2:.4f}")
            st.write(f"Waktu pemrosesan: {elapsed:.2f} detik")

            st.subheader("🥇 Fitur berpengaruh")
            st.bar_chart(models.get_feature_importance().set_index('feature_index'))

            # Create predictions dataframe
            predictions_df = pd.DataFrame({
                'Actual': y_test_original,
                'Predicted': y_pred_original
            })

            st.subheader("📈 Pratinjau Prediksi")
            st.dataframe(predictions_df.head())

            # Download button
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
                key='download-csv'
            )

            st.info(f"TIPS: Kamu bisa mengunggah dataset yang bervariasi dan melihat perubahan hasil fitting model dengan memilih fitur yang berbeda.")
        
        if st.session_state['session_counter'] < 5:
          st.session_state['session_counter'] += 1

else:
        st.info("Unggah file CSV mu terlebih dahulu untuk proses fitting.")