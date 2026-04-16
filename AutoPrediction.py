from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import requests
import time
from datetime import datetime, timedelta
import urllib.parse
import pandas as pd
import numpy as np
from collections import defaultdict

import xgboost as xgb

pd.set_option('display.max_columns', None)

# === 导入你同学爬虫中的函数 ===
# 确保你已经实现或粘贴以下函数：
# get_new_token(), get_weather_data(token, start_time, end_time), process_and_save_data(response_json, start_time, end_time)
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
# ========== 模型与Scaler加载 ==========
model = load_model("cnn_rain_model.h5", compile=False)
scaler = joblib.load("scaler.save")

# 加载 XGBoost 模型
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgboost_rain_model.json")

# ========== 获取最新处理好的数据 ==========
def get_latest_processed_data():
    try:
        token = get_new_token()
        now = datetime.now()
        ten_minutes_ago = now - timedelta(minutes=10)

        response = get_weather_data(token, ten_minutes_ago, now)
        if response and response.status_code == 200:
            df = process_and_save_data(response.json(), ten_minutes_ago, now)
            if df is not None and not df.empty:
                return df.tail(1)
            else:
                print("⚠️ 数据为空")
        else:
            print("⚠️ 请求失败")
    except Exception as e:
        print(f"❌ 异常：{e}")
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
    excel_path = f'/Users/lvjingxuan/PycharmProjects/PythonProject5/{file_name}'

    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name='完整数据')
        for category, params in categories.items():
            available_params = [p for p in params if p in df.columns]
            if available_params:
                df[available_params].to_excel(writer, sheet_name=category)

    print(f"数据已保存到: {excel_path}")
    return df


# ========== 准备模型输入 ==========
def prepare_input(df_row, feature_cols):
    try:
        x = df_row[feature_cols].values.reshape(1, -1)
        x_scaled = scaler.transform(x)
        x_cnn = x_scaled.reshape((1, x_scaled.shape[1], 1))
        return x_cnn
    except Exception as e:
        print(f"❌ 输入准备失败：{e}")
        return None

# ========== 主程序 ==========
if __name__ == "__main__":

    latest_data = get_latest_processed_data()
    if latest_data is not None:
        # 手动设置你训练时的特征列
        feature_cols = ['pow', 'RSSI', 'ambientHumidity', 'ambientTemperature', 'dewtemp',
    'Photosynthesis', 'pressure', 'soilCond', 'soilHumi', 'soilTemp',
    'TotalRadiation', 'windDirection', 'windScale', 'windSpeed']
        X = latest_data[feature_cols]
        print(X)
        input_data = prepare_input(latest_data, feature_cols)

        if input_data is not None:
            prediction = model.predict(input_data).flatten()[0]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] (CNN)Rainfall in 12 hours：{max(prediction,0):.2f} mm")
            # XGBoost 需要的是二维数组，无需reshape成 CNN 格式
            xgb_input = scaler.transform(latest_data[feature_cols].values)
            xgb_pred = xgb_model.predict(xgb_input)[0]
            print(f"[{timestamp}] (XGBoost) Rainfall in 12 hours：{max(xgb_pred,0):.2f} mm")
        else:
            print("Invalid Dataset")
    else:
        print("Invalid Dataset")