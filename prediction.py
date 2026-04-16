# predict_gui.py
import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
import tensorflow as tf
import xgboost as xgb

# 加载模型与 scaler
cnn_model = tf.keras.models.load_model("cnn_rain_model.h5", compile=False)
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgboost_rain_model.json")
scaler = joblib.load("scaler.save")

# 特征名（顺序必须与训练时一致）
feature_names = [
    'pow', 'RSSI', 'ambientHumidity', 'ambientTemperature', 'dewtemp',
    'Photosynthesis', 'pressure', 'soilCond', 'soilHumi', 'soilTemp',
    'TotalRadiation', 'windDirection', 'windScale', 'windSpeed'
]

# 创建窗口
window = tk.Tk()
window.title("Rainfall Prediction")

entries = {}

# 添加输入框
for i, name in enumerate(feature_names):
    label = tk.Label(window, text=name)
    label.grid(row=i, column=0, padx=5, pady=3, sticky="e")
    entry = tk.Entry(window)
    entry.grid(row=i, column=1, padx=5, pady=3)
    entries[name] = entry

# 预测函数
def predict():
    try:
        # 收集并处理输入
        input_values = [float(entries[name].get().strip()) for name in feature_names]
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # CNN 预测
        input_cnn = input_scaled.reshape((1, len(feature_names), 1))
        pred_cnn = cnn_model.predict(input_cnn)[0][0]

        # XGBoost 预测
        pred_xgb = xgb_model.predict(input_scaled)[0]

        # 显示结果
        result_text = (
            f"Rainfall prediction in next 12 hours:\n"
            f"• CNN model:      {pred_cnn:.2f} mm\n"
            f"• XGBoost model:  {pred_xgb:.2f} mm"
        )
        messagebox.showinfo("预测结果", result_text)

    except Exception as e:
        messagebox.showerror("错误", f"输入错误或格式问题：\n{e}")

# 按钮
predict_button = tk.Button(window, text="预测降雨量", command=predict)
predict_button.grid(row=len(feature_names), column=0, columnspan=2, pady=10)

window.mainloop()