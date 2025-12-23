import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_file, output_file):
    print("=== Memulai Preprocessing Otomatis ===")
    
    # 1. Memuat Dataset
    print(f"Memuat data dari {input_file}...")
    df = pd.read_csv(input_file, encoding='latin-1')
    
    # 2. Membersihkan Data Dasar
    print("Membersihkan Missing Values dan Duplikasi...")
    df = df.dropna()
    df = df.drop_duplicates()
    
    #Drop kolom yang tidak diperlukan untuk training
    df = df.drop(columns=['Customer_ID'], errors='ignore')

    # 3. Remove Outliers (Menggunakan metode IQR)
    print("Mendeteksi dan menghapus Outliers (Metode IQR)...")
    
    # Daftar kolom numerik untuk pengecekan outlier
    numerical_cols_check = [
        'Age', 'Annual_Income', 'Total_Spend', 'Years_as_Customer', 
        'Num_of_Purchases', 'Average_Transaction_Amount', 
        'Num_of_Returns', 'Num_of_Support_Contacts', 
        'Satisfaction_Score', 'Last_Purchase_Days_Ago'
    ]
    
    # Pastikan hanya kolom yang ada di df yang dicek
    existing_cols = [c for c in numerical_cols_check if c in df.columns]
    
    # Hitung IQR
    Q1 = df[existing_cols].quantile(0.25)
    Q3 = df[existing_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    # Filter data: Ambil data yang BUKAN outlier
    condition = ~((df[existing_cols] < (Q1 - 1.5 * IQR)) | (df[existing_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_clean = df[condition].copy()
    print(f"Data sebelum outlier removal: {len(df)}, sesudah: {len(df_clean)}")
    df = df_clean

    # 4. Feature Engineering: Binning Age
    print("Melakukan Binning pada kolom Age...")
    age_bins = [16, 25, 35, 45, 55, 65, float('inf')]
    age_labels = [
        "Remaja Akhir (17-25)",
        "Dewasa Awal (26-35)",
        "Dewasa Akhir (36-45)",
        "Lansia Awal (46-55)",
        "Lansia Akhir (56-65)",
        "Manula (>65)"
    ]
    
    # Membuat kategori umur
    age_binned = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=True)
    
    # 5. Encoding (Mengubah Kategori menjadi Angka)
    print("Melakukan Label Encoding...")
    le_age = LabelEncoder()
    df["Age"] = le_age.fit_transform(age_binned)
    
    # Encode kolom kategori lainnya
    categorical_cols = ['Gender', 'Promotion_Response', 'Email_Opt_In']
    
    encoders = {}
    encoders['Age'] = le_age
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # 6. Scaling (Standarisasi Nilai Numerik)
    print("Melakukan Scaling variabel numerik...")
    scaler = StandardScaler()
    
    numerical_cols_scale = [
        'Annual_Income', 'Total_Spend', 'Years_as_Customer', 
        'Num_of_Purchases', 'Average_Transaction_Amount', 
        'Num_of_Returns', 'Num_of_Support_Contacts', 
        'Satisfaction_Score', 'Last_Purchase_Days_Ago'
    ]
    
    cols_to_scale = [c for c in numerical_cols_scale if c in df.columns]
    
    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # 7. Menyimpan Hasil
    print(f"Menyimpan data hasil preprocessing ke {output_file}...")
    df.to_csv(output_file, index=False)
    print("Preprocessing selesai! Data siap dilatih.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(
        BASE_DIR,
        "..",
        "online_retail_customer_churn_raw",
        "online_retail_customer_churn.csv"
    )

    output_path = os.path.join(
        BASE_DIR,
        "..",
        "preprocessing",
        "online_retail_customer_churn_preprocessing.csv"
    )

    preprocess_data(input_path, output_path)
