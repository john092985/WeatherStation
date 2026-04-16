import os
import requests
import time
from datetime import datetime, timedelta
import urllib.parse
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import xgboost as xgb
from tensorflow.keras.models import load_model

xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgboost_rain_model.json")

cnn_model = load_model("cnn_rain_model.h5",compile=False)

import joblib
scaler = joblib.load('scaler.save')

def compute_interval_rainfall(preds, now_rainfall):
    """
    将预测的累积降雨量转为各个半小时时段的实际降雨量
    """
    intervals = [max(0, preds[0] - now_rainfall)]  # 第一个时段
    for i in range(1, len(preds)):
        delta = preds[i] - preds[i - 1]
        intervals.append(max(0, delta))
    return intervals

def plot_combined_rainfall(df_recent, preds, now_rainfall, interval_minutes=30, start_time=None):
    """
    同时绘制过去12小时的真实降雨量 和 未来12小时的预测降雨量
    """
    if start_time is None:
        start_time = datetime.now()

    # 过去真实降雨处理
    df_rain = df_recent[['rainfall']].copy()
    df_rain.index = pd.to_datetime(df_rain.index)
    df_down = df_rain.resample(f'{interval_minutes}min').last().dropna()
    df_down['rainfall'] = df_down['rainfall'].diff().fillna(0)
    df_down.loc[df_down['rainfall'] < 0, 'rainfall'] = 0

    past_values = df_down['rainfall'].values
    past_times = df_down.index.strftime('%H:%M').tolist()

    # 未来预测值差分
    interval_preds = [max(0, preds[0] - now_rainfall)]
    interval_preds += [max(0, preds[i] - preds[i - 1]) for i in range(1, len(preds))]
    future_times = [(start_time + timedelta(minutes=interval_minutes * (i+1))).strftime("%H:%M") for i in range(len(interval_preds))]

    # 合并时间与数据
    all_times = past_times + future_times
    all_values = list(past_values) + interval_preds
    all_colors = ['skyblue'] * len(past_values) + ['royalblue'] * len(interval_preds)

    # 绘图
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(range(len(all_times)), all_values, color=all_colors)
    ax.set_xticks(range(len(all_times)))
    ax.set_xticklabels(all_times, rotation=90, fontsize=8)
    ax.set_xlabel('时间')
    ax.set_ylabel('降雨量 (mm)')
    ax.set_title("Total Prediction Rainfall")
    fig.tight_layout()
    return fig

def create_sliding_windows(df, feature_cols, window_size=6):
    data = df[feature_cols].values
    n_samples = data.shape[0] - window_size + 1
    if n_samples <= 0:
        print("数据长度不足以构建窗口")
        return None

    windows = []
    for i in range(n_samples):
        window = data[i:i + window_size].flatten()
        windows.append(window)
    return np.array(windows)


def predict_next_24_half_hours(df, feature_cols, interval_steps=6):
    """
    从历史数据中每隔 interval_steps（默认6，即30分钟）采样24个点，
    每个点使用标准Scaler做标准化后喂入XGBoost模型，预测未来某时刻的累积降雨量。
    返回预测结果列表。
    """
    if len(df) < interval_steps * 24:
        print("数据不足，无法采样24个间隔点")
        return None

    selected_rows = df.iloc[-interval_steps * 24::interval_steps].copy()
    X = selected_rows[feature_cols].values
    X_scaled = scaler.transform(X)
    preds = xgb_model.predict(X_scaled)
    return preds.tolist()


def get_new_token():
    login_url = 'http://47.92.208.0:8080/api/auth/login'

    # 添加环境变量设置来禁用代理
    os.environ['NO_PROXY'] = '47.92.208.0'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    login_data = {
        'username': 'zheng.dong@hd.kaiwenacademy.cn',
        'password': 'HDKWA_2025'
    }

    try:
        response = requests.post(
            login_url,
            json=login_data,
            headers=headers,
            verify=False,
            timeout=30
        )

        if response.status_code == 200:
            token = response.json()['token']
            print(f"获取到的token: {token[:30]}...")
            return token
        else:
            print(f"登录失败，状态码: {response.status_code}")
            return None

    except Exception as e:
        print(f"登录过程发生错误: {str(e)}")
        return None


def get_weather_data(token, start_date, end_date):
    base_url = 'http://47.92.208.0:8080/api/plugins/telemetry/DEVICE/ccbfbae0-59d8-11ef-bb79-bb5dd335888a/values/timeseries'

    params = {
        'keys': 'ambientHumidity,ambientTemperature,CO2,dewtemp,Photosynthesis,pow,pressure,rainfall,RSSI,soilCond,soilHumi,soilTemp,TotalRadiation,windDirection,windScale,windSpeed',
        'startTs': int(start_date.timestamp() * 1000),
        'endTs': int(end_date.timestamp() * 1000),
        'interval': 300000,  # 修改为5min间隔
        'limit': 10000,
        'agg': 'AVG'
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Authorization': f'Bearer {token}',
        'Connection': 'keep-alive',
        'Keep-Alive': 'timeout=60'
    }

    max_retries = 5
    retry_delay = 30

    for retry in range(max_retries):
        try:
            response = requests.get(
                base_url,
                params=params,
                headers=headers,
                verify=False,
                timeout=120
            )

            if response.status_code == 200:
                return response
            elif response.status_code == 503:
                print(f"服务器暂时不可用，第{retry + 1}次重试")
                if retry < max_retries - 1:
                    wait_time = retry_delay * (2 ** retry)
                    print(f"等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
            else:
                print(f"请求失败，状态码: {response.status_code}")
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
        except Exception as e:
            print(f"请求发生错误: {str(e)}")
            if retry < max_retries - 1:
                time.sleep(retry_delay)

    return None


def process_and_save_data(data, start_date, end_date):
    all_timestamps = set()
    measurements = defaultdict(dict)

    for key, values in data.items():
        for entry in values:
            ts = datetime.fromtimestamp(entry['ts'] / 1000)
            all_timestamps.add(ts)
            measurements[ts][key] = entry['value']

    df_data = []
    for ts in sorted(all_timestamps):
        row = {'时间戳': ts}
        row.update(measurements[ts])
        df_data.append(row)

    df = pd.DataFrame(df_data)
    df.set_index('时间戳', inplace=True)

    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        except:
            continue

    categories = {
        '温度相关': ['ambientTemperature', 'dewtemp', 'soilTemp'],
        '湿度相关': ['ambientHumidity', 'soilHumi'],
        '大气相关': ['pressure', 'CO2'],
        '光照相关': ['Photosynthesis', 'TotalRadiation'],
        '风相关': ['windDirection', 'windScale', 'windSpeed'],
        '其他指标': ['pow', 'rainfall', 'RSSI', 'soilCond']
    }

    file_name = f'WeatherData_{start_date.strftime("%Y%m%d")}-{end_date.strftime("%Y%m%d")}.xlsx'
    excel_path = f'/Users/lvjingxuan/Desktop/WeatherData/{file_name}'
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name='完整数据')

    print(f"数据已保存到: {excel_path}")
    return df


def main():
    # 设置总的时间范围
    start_global = datetime(2024, 5, 20, 22, 0, 0)
    end_global = datetime(2025, 5, 21, 10, 0, 0)  # 改为第二天

    # 修改为1天间隔
    delta = timedelta(days=1)

    # 获取token
    token = get_new_token()
    if not token:
        print("登录失败，无法获取token")
        return

    # 存储所有数据
    all_data = pd.DataFrame()
    current_start = start_global
    period_count = 1

    while current_start < end_global:
        current_end = min(current_start + delta, end_global)

        print(f"\n获取第{period_count}组数据:")
        print(f"开始时间: {current_start}")
        print(f"结束时间: {current_end}")

        try:
            response = get_weather_data(token, current_start, current_end)
            if response and response.status_code == 200:
                df = process_and_save_data(response.json(), current_start, current_end)
                if df is not None and not df.empty:
                    all_data = pd.concat([all_data, df])
                    print(f"成功获取数据点数: {len(df)}")
                    time.sleep(2)
                else:
                    print("获取的数据为空")
            else:
                print("获取数据失败")
        except Exception as e:
            print(f"处理数据时发生错误: {str(e)}")

        current_start = current_end
        period_count += 1
        time.sleep(30)  # 每次请求之间等待30秒

    # 保存完整数据集
    if not all_data.empty:
        complete_path = '/Users/lvjingxuan/PycharmProjects/PythonProject5/WeatherData_Complete.xlsx'
        os.makedirs(os.path.dirname(complete_path), exist_ok=True)
        with pd.ExcelWriter(complete_path) as writer:
            all_data.to_excel(writer, sheet_name='完整数据')

            categories = {
                '温度相关': ['ambientTemperature', 'dewtemp', 'soilTemp'],
                '湿度相关': ['ambientHumidity', 'soilHumi'],
                '大气相关': ['pressure', 'CO2'],
                '光照相关': ['Photosynthesis', 'TotalRadiation'],
                '风相关': ['windDirection', 'windScale', 'windSpeed'],
                '其他指标': ['pow', 'rainfall', 'RSSI', 'soilCond']
            }

            for category, params in categories.items():
                available_params = [p for p in params if p in all_data.columns]
                if available_params:
                    all_data[available_params].to_excel(writer, sheet_name=category)

        print(f"\n所有数据已合并保存到: {complete_path}")

def get_past_12_hours_data():
    token = get_new_token()
    if not token:
        print("无法获取 token")
        return None

    end_time = datetime.now()
    start_time = end_time - timedelta(hours=12)

    print(f"获取过去12小时数据：\n开始时间：{start_time}\n结束时间：{end_time}")

    response = get_weather_data(token, start_time, end_time)
    if response and response.status_code == 200:
        df = process_and_save_data(response.json(), start_time, end_time)
        if df is not None and not df.empty:
            print(f"✅ 获取成功，数据点数：{len(df)}")
            return df
        else:
            print("⚠️ 获取到的数据为空")
    else:
        print("⚠️ 获取失败")

    return None

def plot_rainfall_bars(df, interval_minutes=30):
    """
    绘制降雨量柱状图。interval_minutes 为柱状图时间间隔，可选 5, 10, 15, 30 等。
    """
    df_rain = df[['rainfall']].copy()
    df_rain.index = pd.to_datetime(df_rain.index)

    # 进行降采样（取最后值），并计算增量
    df_down = df_rain.resample(f'{interval_minutes}min').last().dropna()
    df_down['rainfall'] = df_down['rainfall'].diff().fillna(0)
    df_down.loc[df_down['rainfall'] < 0, 'rainfall'] = 0

    times = df_down.index.strftime('%H:%M')
    values = df_down['rainfall'].values

    plt.figure(figsize=(16, 6))
    plt.bar(times, values, width=1.0, color='skyblue')
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel('时间')
    plt.ylabel('降雨量 (mm)')
    plt.title(f'过去12小时降雨量（每{interval_minutes}分钟）')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df_recent = get_past_12_hours_data()  # 你已有的函数

    feature_cols = ['pow', 'RSSI', 'ambientHumidity', 'ambientTemperature', 'dewtemp',
                    'Photosynthesis', 'pressure', 'soilCond', 'soilHumi', 'soilTemp',
                    'TotalRadiation', 'windDirection', 'windScale', 'windSpeed']
    window_size = 6  # 半小时窗口

    if df_recent is not None and len(df_recent) >= window_size + 24 - 1:

        def prepare_input(df_row, feature_cols):
            try:
                x = df_row[feature_cols].values.reshape(1, -1)
                x_scaled = scaler.transform(x)
                x_cnn = x_scaled.reshape((1, x_scaled.shape[1], 1))
                return x_cnn
            except Exception as e:
                print(f"❌ 输入准备失败：{e}")
                return None

        preds = []
        step = 6
        selected_rows = df_recent.iloc[-step * 24::step].copy()
        for _, row in selected_rows.iterrows():
            x_cnn = prepare_input(row.to_frame().T, feature_cols)
            if x_cnn is not None:
                pred = cnn_model.predict(x_cnn).flatten()[0]
                preds.append(pred)

        now_rainfall = df_recent['rainfall'].iloc[-1]
        print(f"当前累计降雨量: {now_rainfall:.2f} mm")
        plot_combined_rainfall(df_recent, preds, now_rainfall, interval_minutes=30)
    else:
        print("数据不足，无法预测")