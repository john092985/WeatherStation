import tkinter as tk
from tkinter import messagebox
from THOMAS_CNN import get_past_12_hours_data, plot_combined_rainfall
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import xgboost as xgb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 加载模型和Scaler
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgboost_rain_model.json")
cnn_model = load_model("cnn_rain_model.h5", compile=False)
scaler = joblib.load('scaler.save')

# 特征列
feature_cols = ['pow', 'RSSI', 'ambientHumidity', 'ambientTemperature', 'dewtemp',
                'Photosynthesis', 'pressure', 'soilCond', 'soilHumi', 'soilTemp',
                'TotalRadiation', 'windDirection', 'windScale', 'windSpeed']

def prepare_input(df_row):
    try:
        x = df_row[feature_cols].values.reshape(1, -1)
        x_scaled = scaler.transform(x)
        x_cnn = x_scaled.reshape((1, x_scaled.shape[1], 1))
        return x_cnn
    except Exception as e:
        print(f"❌ 输入准备失败：{e}")
        return None

def run_prediction():
    df_recent = get_past_12_hours_data()
    if df_recent is None or len(df_recent) < 6 * 24:
        messagebox.showwarning("Warning", "Insufficient data. Prediction cannot proceed.")
        return

    preds = []
    selected_rows = df_recent.iloc[-6 * 24::6].copy()
    for _, row in selected_rows.iterrows():
        x_cnn = prepare_input(row.to_frame().T)
        if x_cnn is not None:
            pred = cnn_model.predict(x_cnn).flatten()[0]
            preds.append(pred)

    now_rainfall = df_recent['rainfall'].iloc[-1]
    cnn_total = preds[-1] if preds else 0

    # XGBoost预测
    xgb_input = scaler.transform(selected_rows[feature_cols].values)
    xgb_preds = xgb_model.predict(xgb_input).tolist()
    xgb_total = xgb_preds[-1] if xgb_preds else 0

    # 展示结果
    result_text = (
        f"CNN predicted rainfall in next 12 hours: {cnn_total:.2f} mm\n"
        f"XGBoost predicted rainfall in next 12 hours: {xgb_total:.2f} mm\n"
        f"Current accumulated rainfall: {now_rainfall:.2f} mm"
    )
    result_label.config(text=result_text)

    # 绘图
    global canvas
    fig = plot_combined_rainfall(df_recent, preds, now_rainfall)
    if canvas:
        canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

# 创建主窗口
root = tk.Tk()
root.title("Rainfall Prediction System")
root.geometry("480x300")

# 标题
title_label = tk.Label(root, text="Next 12-Hour Rainfall Forecast", font=("Helvetica", 16))
title_label.pack(pady=20)

# 按钮
predict_btn = tk.Button(root, text="Fetch and Predict", command=run_prediction, font=("Helvetica", 12))
predict_btn.pack(pady=10)

# 刷新按钮
refresh_btn = tk.Button(root, text="Refresh", command=lambda: result_label.config(text=""), font=("Helvetica", 12))
refresh_btn.pack(pady=5)

# 显示结果
result_label = tk.Label(root, text="", font=("Helvetica", 12), justify=tk.LEFT)
result_label.pack(pady=10)

canvas = None

root.mainloop()