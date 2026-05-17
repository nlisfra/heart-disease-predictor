import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv("heart_data.csv")

df.columns = df.columns.str.strip().str.lstrip('\ufeff')

print("Kolom:", df.columns.tolist())
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# ── 2. Pisahkan fitur & target ────────────────────────────────
X = df.drop("target", axis=1)
y = df["target"]

# ── 3. Split data ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Scaling ────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 5. Train model ────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ── 6. Evaluasi ───────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Akurasi: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 7. Simpan model & scaler ──────────────────────────────────
joblib.dump(model,  "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n💾 model.pkl dan scaler.pkl berhasil disimpan!")
