import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Load data
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

target_col = "Churn"
id_col = "ID"

X = train.drop(columns=[target_col])
y = train[target_col]

test_ids = test[id_col]
X_test = test.drop(columns=[id_col])

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Preprocessing
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

if id_col in numeric_features:
    numeric_features.remove(id_col)
if id_col in categorical_features:
    categorical_features.remove(id_col)

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Train + validate
clf.fit(X_train, y_train)
val_proba = clf.predict_proba(X_val)[:, 1]
print("Validation ROC AUC:", roc_auc_score(y_val, val_proba))

# Train on full training set
clf.fit(X, y)

# Predict on test
test_proba = clf.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    id_col: test_ids,
    target_col: test_proba
})

out_path = "./data/submission.csv"
submission.to_csv(out_path, index=False)

print(f"Saved submission to: {out_path}")
