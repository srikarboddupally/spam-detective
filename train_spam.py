import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# Data loading
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
colnames = [
    f"word_freq_{i}" for i in range(48)
] + [f"char_freq_{i}" for i in range(6)] + [
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
    "spam"
]
df = pd.read_csv(url, header=None, names=colnames)

# Split features and label
X = df.drop("spam", axis=1)
y = df["spam"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=35)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Balance dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_selected, y_train)

# XGBoost with grid search
param_grid = {
    'n_estimators':[100, 250],
    'max_depth':[3, 5, 7],
    'learning_rate':[0.05, 0.1],
    'scale_pos_weight':[1, 2]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid = GridSearchCV(xgb, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=2)
grid.fit(X_resampled, y_resampled)

print("Best Params:", grid.best_params_)

# Evaluate
y_pred = grid.predict(X_test_selected)
print(classification_report(y_test, y_pred))

# Save model and processors
joblib.dump(grid.best_estimator_, "spam_xgb_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(selector, "selector.joblib")
