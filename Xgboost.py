import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

# 1. Load dataset
df = pd.read_csv("ps.csv")

# 2. Extract features and target
feature_cols = [col for col in df.columns if col not in ['start_time', 'rainfall_12h_future']]
X = df[feature_cols].values
y = df['rainfall_12h_future'].values

# 3. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Train XGBoost regressor
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 6. Evaluate the model with clipping to avoid negative rainfall
y_pred_raw = xgb_model.predict(X_test)
y_pred = np.maximum(0, y_pred_raw)  # Ensure no negative predictions

mae = mean_absolute_error(y_test, y_pred)
print(f"XGBoost Test MAE: {mae:.2f} mm")

# 7. Plot predictions
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("True Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("XGBoost Predictions vs Actual")
plt.grid()
plt.show()

# 8. Print some example predictions
for i in range(10):
    print(f"True: {y_test[i]:.2f} mm, Predicted: {y_pred[i]:.2f} mm")

# 9. Focus on range 10–30 mm
count = 0
for true_val, pred_val in zip(y_test, y_pred):
    if 10 <= true_val <= 30:
        print(f"True: {true_val:.2f} mm, Predicted: {pred_val:.2f} mm")
        count += 1
        if count >= 20:
            break

# 10. Save model and scaler
xgb_model.save_model("xgboost_rain_model.json")
joblib.dump(scaler, "scaler.save")