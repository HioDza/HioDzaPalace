import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from models.L2C import L2Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pandas.api.types import is_string_dtype

# ===== FUNCTIONS =====

@st.cache_data
def load_and_validate_data(uploaded_train_file, uploaded_test_file):
    """Load and validate uploaded CSV files."""
    try:
        # Check file size (5 MB limit)
        if uploaded_train_file.size > 5 * 1024 * 1024:
            st.error("Ukuran file melebihi batas 5 MB. Silakan unggah file yang lebih kecil.", icon='⚠️')
            return None, None

        data_train = pd.read_csv(uploaded_train_file, encoding='cp1252')

        # Check dataset dimensions (50,000 rows, 50 columns)
        if data_train.shape[0] > 50000:
            st.error("Dataset memiliki lebih dari 50,000 baris. Silakan unggah dataset yang lebih kecil.", icon='⚠️')
            return None, None
        if data_train.shape[1] > 50:
            st.error("Dataset memiliki lebih dari 50 kolom. Silakan unggah dataset yang lebih kecil.", icon='⚠️')
            return None, None

        data_test = None
        if uploaded_test_file is not None:
            data_test = pd.read_csv(uploaded_test_file, encoding='cp1252')
            if data_train.shape[1] != data_test.shape[1]:
                st.error("Dataset fit dan prediksi harus memiliki jumlah kolom yang sama.", icon='⚠️')
                return None, None

        return data_train, data_test
    except Exception as e:
        st.error(f"Error loading data: {str(e)}", icon='⚠️')
        return None, None

def display_data_preview(data_train):
    """Display data preview."""
    st.write("Pratinjau dataset fitting:")
    st.dataframe(data_train.head())

def configure_features_and_target(data_train):
    """Configure feature and target columns."""
    st.subheader("⚙️ Konfigurasi Fitur dan Target")
    all_columns = data_train.columns.tolist()
    target_column = st.selectbox("Pilih Kolom Target", all_columns)
    feature_columns = st.multiselect("Pilih Kolom Fitur", [col for col in all_columns if col != target_column])

    # If user doesn't pick features, automatically use all non-target columns
    if not feature_columns:
        feature_columns = [col for col in all_columns if col != target_column]
        st.info(f"Tidak ada fitur yang dipilih — menggunakan semua fitur non-target ({len(feature_columns)} kolom)", icon="ℹ️")

    return target_column, feature_columns

def preprocess_features(data_train, data_test, feature_columns, target_column, test_size, _progress_bar):
    """Preprocess features for training and testing."""
    _progress_bar.progress(10)

    if not feature_columns:
        raise ValueError("Dimohon untuk memilih minimal 1 fitur data.")

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
                X_cat = enc.fit_transform(data_train[[col]].fillna('missing'))
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

    # Save vectorizers and encoders for reuse
    st.session_state['vectorizers'] = vectorizers
    st.session_state['encoders'] = encoders

    # Horizontally stack all parts into X
    X = np.hstack(parts)
    y = data_train[target_column].fillna('unknown').values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _progress_bar.progress(30)

    if data_test is not None:
        if target_column not in data_test.columns:
            raise ValueError("Kolom target tidak ditemukan di dataset prediksi.")

        # Process test data features
        parts_test = []
        for col in feature_columns:
            if is_string_dtype(data_test[col]) or data_test[col].dtype == object:
                unique_vals = data_test[col].nunique()
                if unique_vals <= 10:
                    # Use fitted encoder
                    enc = encoders.get(col)
                    if enc is not None:
                        X_cat_test = enc.transform(data_test[[col]].fillna('missing'))
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

        X_test = np.hstack(parts_test)
        X_test = scaler.transform(X_test)
        y_test = data_test[target_column].fillna('unknown').values

        X_train, y_train = X_scaled, y
    else:
        # Split train data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    _progress_bar.progress(50)
    return X_train, X_test, y_train, y_test, scaler

def train_and_evaluate(X_train, y_train, X_test, y_test, alpha, fit_intercept, _progress_bar):
    """Train model and evaluate performance."""
    _progress_bar.progress(60)

    model = L2Classifier(alpha=alpha, fit_intercept=fit_intercept)

    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = time.time() - start_time

    _progress_bar.progress(80)
    return model, y_pred, elapsed

def display_results(y_test, y_pred, model, elapsed, _progress_bar):
    """Display evaluation results."""
    _progress_bar.progress(90)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.subheader("📊 Metrik evaluasi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Akurasi", f"{accuracy:.4f}")
        st.metric("Waktu Pemrosesan", f"{elapsed:.2f} detik")
    with col2:
        st.text("Classification Report:")
        st.code(report)

    # Confusion Matrix Visualization
    st.subheader("📊 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.subheader("🥇 Fitur berpengaruh")
    feature_importance = model.get_feature_importance()
    if not feature_importance.empty:
        st.bar_chart(feature_importance.set_index('feature_index'))

    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })

    st.subheader("📈 Pratinjau Prediksi")
    st.dataframe(predictions_df.head())

    # Download button
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
        key='download-csv',
        icon="📥"
    )

    st.info("TIPS: Kamu bisa mengunggah dataset yang bervariasi dan melihat perubahan hasil fitting model dengan memilih fitur yang berbeda.", icon="💡")

    _progress_bar.progress(100)
    _progress_bar.empty()

# ===== STREAMLIT APP =====

st.set_page_config(page_title="Prediksi label data", 
                   layout='wide', 
                   initial_sidebar_state='collapsed',
                   page_icon="📈",
                   menu_items={
                       'About': "Aplikasi prediksi label data menggunakan model yang berbasis regulasi, yaitu ridge atau juga dikenal sebagai L2."
                              }
                       )

# Sidebar for configuration
st.sidebar.title("⚙️ Konfigurasi Model")
alpha = st.sidebar.slider("Alpha (Kekuatan regulasi model)", 1e-6, 1.0, 1e-4, format="%.6f")
fit_intercept = st.sidebar.checkbox("Fit Intercept", value=True)
test_size = st.sidebar.slider("Ukuran data tes (jika kamu tidak input file tes secara terpisah)", 0.1, 0.5, 0.2)

# Model persistence options
st.sidebar.subheader("💾 Model Persistence")
save_model = st.sidebar.button("💾 Save Trained Model")
load_model = st.sidebar.file_uploader("📂 Load Trained Model", type="pkl")
st.sidebar.info("""
                ## Penjelasan pengaturan:
                - **Alpha** mengontrol kekuatan regulasi model. Nilai lebih tinggi berarti regulasi lebih kuat yang membuat model lebih 'berhati-hati' saat fitting ke data mu.
                - **Fit Intercept** menentukan apakah model harus menghitung bias, efeknya model bisa lebih leluasa dalam memprediksi.
                - **Ukuran data tes** menentukan proporsi data yang digunakan untuk pengujian jika tidak ada file tes yang diunggah.
                """, icon="ℹ️")

st.title("📈 Prediksi label dari data")
st.caption("Ringan dan mudah untuk digunakan.")
st.subheader("📂 Unggah Dataset (CSV)")
if st.button("Petunjuk Penggunaan", icon="ℹ️"):
    st.info("""
    1. Siapkan dataset dalam format CSV dengan kolom fitur dan satu kolom target.
    2. Unggah file CSV untuk fitting model. Opsional: unggah file CSV lain untuk prediksi.
    3. Pilih kolom target dan fitur yang diinginkan.
    4. Klik 'Proses' untuk melatih model dan melihat hasil prediksi.
    5. Unduh hasil prediksi sebagai file CSV.
    
    **Catatan:**
    - Pastikan dataset tidak melebihi batas ukuran dan dimensi yang ditentukan.
    - Gunakan fitur yang relevan untuk hasil terbaik.
    - Setiap sesi dibatasi hingga 5 kali proses fitting untuk menjaga kinerja aplikasi.
    """, icon="📝")
st.info(f"Batas ukuran file: 5 MB | Batas baris: 50,000 | Batas kolom: 50. Batasan diberikan demi menjaga kinerja aplikasi tetap optimal.", icon="⚠️")

uploaded_train_file = st.file_uploader("Pilih file CSV untuk fitting", type="csv")
uploaded_test_file = st.file_uploader("Pilih file CSV untuk prediksi (opsional)", type="csv")

# Initialize session counter in session state
if 'session_counter' not in st.session_state:
    st.session_state['session_counter'] = 1

# Handle model persistence
if save_model and 'trained_model' in st.session_state:
    model_data = {
        'model': st.session_state['trained_model'],
        'vectorizers': st.session_state.get('vectorizers', {}),
        'encoders': st.session_state.get('encoders', {}),
        'scaler': st.session_state.get('scaler'),
        'feature_columns': st.session_state.get('feature_columns'),
        'target_column': st.session_state.get('target_column')
    }
    buffer = BytesIO()
    pickle.dump(model_data, buffer)
    buffer.seek(0)
    st.sidebar.download_button(
        label="Download Model",
        data=buffer,
        file_name="trained_model.pkl",
        mime="application/octet-stream",
        icon="📥"
    )

if load_model is not None:
    try:
        model_data = pickle.load(load_model)
        st.session_state['trained_model'] = model_data['model']
        st.session_state['vectorizers'] = model_data.get('vectorizers', {})
        st.session_state['encoders'] = model_data.get('encoders', {})
        st.session_state['scaler'] = model_data.get('scaler')
        st.session_state['feature_columns'] = model_data.get('feature_columns')
        st.session_state['target_column'] = model_data.get('target_column')
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}", icon='⚠️')

if uploaded_train_file is not None:
    data_train, data_test = load_and_validate_data(uploaded_train_file, uploaded_test_file)

    if data_train is not None:
        display_data_preview(data_train)

        target_column, feature_columns = configure_features_and_target(data_train)

        if st.button("Proses", icon="🚀"):

            if st.session_state['session_counter'] > 5:
                st.error("Batas sesi tercapai. Silakan muat ulang halaman untuk memulai sesi baru.", icon="⚠️")
                st.stop()
            
            st.info(f"Sisa batas sesi: {5 - st.session_state['session_counter']}", icon="ℹ️")

            with st.spinner("Memproses data..."):
                progress_bar = st.progress(0)

                try:
                    X_train, X_test, y_train, y_test, scaler = preprocess_features(
                        data_train, data_test, feature_columns, target_column, test_size, progress_bar
                    )

                    model, y_pred, elapsed = train_and_evaluate(
                        X_train, y_train, X_test, y_test, alpha, fit_intercept, progress_bar
                    )

                    display_results(y_test, y_pred, model, elapsed, progress_bar)

                    # Store model in session state
                    st.session_state['trained_model'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['feature_columns'] = feature_columns
                    st.session_state['target_column'] = target_column

                    st.session_state['session_counter'] += 1

                except Exception as e:
                    st.error(f"Error during processing: {str(e)}", icon='⚠️')
                    progress_bar.empty()
else:
    st.info("Unggah file CSV mu terlebih dahulu untuk proses fitting.", icon="‼️")
