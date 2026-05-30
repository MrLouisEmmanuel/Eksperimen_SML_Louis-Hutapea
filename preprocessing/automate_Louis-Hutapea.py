"""
Automate Preprocessing - California Housing Dataset
====================================================
Script otomasi untuk preprocessing dataset California Housing.
Dibuat oleh: Louis-Hutapea
Tanggal: 2026-05-30

Script ini mengotomasi seluruh pipeline preprocessing:
1. Load data dari sklearn atau file CSV
2. Eksplorasi data (EDA ringkas)
3. Preprocessing (outlier removal, scaling, train-test split)
4. Simpan hasil preprocessing ke file CSV
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data(raw_data_dir=None):
    """
    Memuat dataset California Housing.
    
    Pertama mencoba membaca dari file CSV yang sudah ada.
    Jika tidak ditemukan, mengambil langsung dari sklearn dan menyimpan sebagai CSV.
    
    Parameters:
    -----------
    raw_data_dir : str, optional
        Path ke direktori penyimpanan data mentah.
        Default: ../california_housing_raw/
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame berisi fitur dan target (MedHouseVal)
    """
    if raw_data_dir is None:
        # Relatif terhadap lokasi script ini
        script_dir = os.path.dirname(os.path.abspath(__file__))
        raw_data_dir = os.path.join(script_dir, "..", "california_housing_raw")
    
    raw_csv_path = os.path.join(raw_data_dir, "housing_raw.csv")
    
    if os.path.exists(raw_csv_path):
        print(f"[INFO] Membaca data dari file: {raw_csv_path}")
        df = pd.read_csv(raw_csv_path)
    else:
        print("[INFO] File CSV tidak ditemukan. Mengambil data dari sklearn...")
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame  # sudah include target column 'MedHouseVal'
        
        # Buat direktori jika belum ada
        os.makedirs(raw_data_dir, exist_ok=True)
        df.to_csv(raw_csv_path, index=False)
        print(f"[INFO] Data mentah disimpan ke: {raw_csv_path}")
    
    print(f"[INFO] Dataset berhasil dimuat. Shape: {df.shape}")
    return df


# =============================================================================
# 2. EXPLORE DATA (EDA Ringkas)
# =============================================================================
def explore_data(df):
    """
    Menampilkan informasi eksplorasi data dasar.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame yang akan dieksplorasi
    """
    print("\n" + "=" * 60)
    print("EKSPLORASI DATA (EDA)")
    print("=" * 60)
    
    # Informasi umum
    print("\n--- Informasi Dataset ---")
    print(f"Jumlah baris   : {df.shape[0]}")
    print(f"Jumlah kolom   : {df.shape[1]}")
    print(f"Nama kolom     : {list(df.columns)}")
    
    # Tipe data
    print("\n--- Tipe Data ---")
    print(df.dtypes)
    
    # Missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing)
    total_missing = missing.sum()
    if total_missing == 0:
        print("Tidak ada missing values! ✓")
    else:
        print(f"Total missing values: {total_missing}")
    
    # Statistik deskriptif
    print("\n--- Statistik Deskriptif ---")
    print(df.describe().round(4))
    
    # Cek duplikat
    duplicates = df.duplicated().sum()
    print(f"\n--- Duplikat ---")
    print(f"Jumlah baris duplikat: {duplicates}")
    
    print("\n" + "=" * 60)
    print("EDA selesai.")
    print("=" * 60)
    
    return df


# =============================================================================
# 3. PREPROCESS DATA
# =============================================================================
def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Melakukan preprocessing lengkap pada dataset:
    - Penanganan outlier dengan metode IQR
    - Feature scaling dengan StandardScaler
    - Train-test split (80:20)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame mentah
    test_size : float
        Proporsi data test (default: 0.2)
    random_state : int
        Random seed untuk reproducibility
    
    Returns:
    --------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : pd.Series
    y_test : pd.Series
    df_preprocessed : pd.DataFrame
        DataFrame setelah preprocessing (sebelum split)
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    
    df_clean = df.copy()
    
    # --- 3a. Penanganan Outlier dengan IQR ---
    print("\n--- Penanganan Outlier (Metode IQR) ---")
    print(f"Jumlah data sebelum penanganan outlier: {len(df_clean)}")
    
    # Kolom numerik untuk deteksi outlier (semua kecuali target)
    feature_cols = [col for col in df_clean.columns if col != "MedHouseVal"]
    
    for col in feature_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        
        if outliers_count > 0:
            print(f"  {col}: {outliers_count} outlier(s) terdeteksi "
                  f"(batas: [{lower_bound:.4f}, {upper_bound:.4f}])")
        
        # Clip outlier ke batas IQR (winsorization) agar tidak kehilangan terlalu banyak data
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"Jumlah data setelah penanganan outlier (clip): {len(df_clean)}")
    
    # --- 3b. Pisahkan fitur dan target ---
    X = df_clean.drop(columns=["MedHouseVal"])
    y = df_clean["MedHouseVal"]
    
    print(f"\nFitur (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    
    # --- 3c. Train-Test Split ---
    print(f"\n--- Train-Test Split ({int((1-test_size)*100)}:{int(test_size*100)}) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")
    
    # --- 3d. Feature Scaling (StandardScaler) ---
    print("\n--- Feature Scaling (StandardScaler) ---")
    scaler = StandardScaler()
    
    # Fit pada training data, transform pada training dan test data
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("Scaling selesai (fit pada train, transform pada train & test).")
    print(f"Mean X_train (setelah scaling): {X_train_scaled.mean().round(6).tolist()}")
    print(f"Std X_train (setelah scaling) : {X_train_scaled.std().round(4).tolist()}")
    
    # --- 3e. Gabungkan data preprocessed (untuk disimpan juga) ---
    df_preprocessed = pd.concat([
        pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        ),
        y.reset_index(drop=True)
    ], axis=1)
    
    print("\n" + "=" * 60)
    print("Preprocessing selesai!")
    print("=" * 60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, df_preprocessed


# =============================================================================
# 4. SAVE PREPROCESSED DATA
# =============================================================================
def save_preprocessed_data(X_train, X_test, y_train, y_test, df_preprocessed=None, output_dir=None):
    """
    Menyimpan data hasil preprocessing ke file CSV.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Fitur training dan testing
    y_train, y_test : pd.Series
        Target training dan testing
    df_preprocessed : pd.DataFrame, optional
        DataFrame gabungan hasil preprocessing
    output_dir : str, optional
        Direktori output. Default: california_housing_preprocessing/
    """
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "california_housing_preprocessing")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[INFO] Menyimpan data preprocessing ke: {output_dir}")
    
    # Simpan masing-masing split
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False, header=True)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=True)
    
    print("  ✓ X_train.csv")
    print("  ✓ X_test.csv")
    print("  ✓ y_train.csv")
    print("  ✓ y_test.csv")
    
    # Simpan data gabungan preprocessed
    if df_preprocessed is not None:
        df_preprocessed.to_csv(
            os.path.join(output_dir, "housing_preprocessed.csv"), index=False
        )
        print("  ✓ housing_preprocessed.csv")
    
    print(f"\n[INFO] Semua file berhasil disimpan! Total: {len(os.listdir(output_dir))} file")


# =============================================================================
# 5. MAIN - Orkestrasi Pipeline
# =============================================================================
def main():
    """
    Fungsi utama yang menjalankan seluruh pipeline preprocessing:
    1. Load data
    2. Eksplorasi data
    3. Preprocessing (outlier handling, scaling, splitting)
    4. Simpan hasil
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   AUTOMATE PREPROCESSING - CALIFORNIA HOUSING DATASET  ║")
    print("║   Oleh: Louis-Hutapea                                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # Step 1: Load data
    print("\n[STEP 1/4] Memuat dataset...")
    df = load_data()
    
    # Step 2: Eksplorasi data
    print("\n[STEP 2/4] Eksplorasi data...")
    explore_data(df)
    
    # Step 3: Preprocessing
    print("\n[STEP 3/4] Preprocessing data...")
    X_train, X_test, y_train, y_test, df_preprocessed = preprocess_data(df)
    
    # Step 4: Simpan hasil
    print("\n[STEP 4/4] Menyimpan hasil preprocessing...")
    save_preprocessed_data(X_train, X_test, y_train, y_test, df_preprocessed)
    
    print("\n" + "=" * 60)
    print("Pipeline preprocessing selesai! ✓")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
