import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load dataset
df = pd.read_csv("ps.csv")

# 2. Extract features and target
feature_cols = [col for col in df.columns if col not in ['start_time', 'rainfall_12h_future']]
X = df[feature_cols].values
y = df['rainfall_12h_future'].values

# 3. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Reshape for CNN: (samples, timesteps, features)
X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# 5. Train/test split (random)
X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y, test_size=0.2, random_state=42
)

# 6. Build CNN model
model = models.Sequential([
    layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Allow free prediction, clip negatives during inference
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 7. Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=32,
    verbose=1
)

# 8. Evaluate on test set
y_pred = model.predict(X_test).flatten()
y_pred = np.maximum(0, y_pred)  # Clip to non-negative values
mae = mean_absolute_error(y_test, y_pred)
print(f"Test MAE: {mae:.2f} mm")

# 9. Optional: Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Training vs Validation MAE")
plt.legend()
plt.grid()
plt.show()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred_lr = lr.predict(X_test.reshape(X_test.shape[0], -1))
print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lr))

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("True Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("CNN Predictions vs Actual")
plt.grid()
plt.show()


for i in range(10):
    print(f"True: {y_test[i]:.2f} mm, Predicted: {y_pred[i]:.2f} mm")

    count = 0
    for true_val, pred_val in zip(y_test, y_pred):
        if 10 <= true_val <= 30:
            print(f"True: {true_val:.2f} mm, Predicted: {pred_val:.2f} mm")
            count += 1
            if count >= 20:
                break

# 13. Save model
model.save("cnn_rain_model.h5")
import joblib

joblib.dump(scaler, "scaler.save")