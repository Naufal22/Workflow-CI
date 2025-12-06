import pandas as pd
import os
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. KONFIGURASI DAGSHUB ---
os.environ["DAGSHUB_USER_TOKEN"] = "ce3238b3a7c35717e39d5ea8b431f6ddebfc92c6"
dagshub.init(repo_owner='Naufal22', repo_name='Eksperimen_SML_MuhammadNaufalAqil', mlflow=True)

def main():
    print("🚀 Memulai Training Model (Mode: AUTOLOG)...")

    # 2. LOAD DATA
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    train_x_path = os.path.join(data_dir, 'train_clean.csv')
    train_y_path = os.path.join(data_dir, 'train_target.csv')
    test_x_path = os.path.join(data_dir, 'test_clean.csv')
    test_y_path = os.path.join(data_dir, 'test_target.csv')

    if not os.path.exists(train_x_path):
        print(f"❌ Error: File data tidak ditemukan di {data_dir}")
        return

    print("📖 Membaca data...")
    X_train = pd.read_csv(train_x_path)
    y_train = pd.read_csv(train_y_path).iloc[:, 0]
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path).iloc[:, 0]

    # 3. AKTIFKAN AUTOLOG
    mlflow.sklearn.autolog(log_models=True)

    # 4. TRAINING
    mlflow.set_experiment("Eksperimen_Telco_Churn_Final")
    
    with mlflow.start_run():
        print("🧠 Sedang melatih model...")
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Akurasi: {acc:.4f}")
        
        print("🎉 Selesai!.")

if __name__ == "__main__":
    main()