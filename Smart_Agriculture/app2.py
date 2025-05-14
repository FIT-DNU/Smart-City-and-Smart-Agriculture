# app.py
from flask import Flask, request, jsonify, send_from_directory
import threading, time, csv, requests, joblib, numpy as np
from datetime import datetime

app = Flask(__name__, static_folder='static')

ESP_IP        = "192.168.137.100"     # copy từ Serial Monitor ESP
THING_NAME    = "my-pump-thing"       # thing name trên dweet.io
DATA_FILE     = "sensor_data.csv"
THRESHOLD_VOL = 5.0                    # ngưỡng ml để bơm

# Load pipeline XGBoost
pipeline = joblib.load("xgb_reg_pipeline.joblib")

# Trạng thái cuối cùng để /status trả về
latest = {
    'timestamp':     None,
    'temperature':   None,
    'humidity_env':  None,
    'soil_moisture': None,
    'volume_ml':     None,
    'action':        None,
    'pump_status':   None
}

def fetch_and_store():
    """Background: pull Dweet mỗi 10s → append CSV."""
    # nếu chưa có file, tạo header
    try:
        open(DATA_FILE).close()
    except FileNotFoundError:
        with open(DATA_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","temp","hum","soil"])
    while True:
        try:
            r = requests.get(
                f"https://dweet.io/get/latest/dweet/for/{THING_NAME}", timeout=5
            ).json()
            e = r["with"][0]
            ts = e["created"]; cnt = e["content"]
            with open(DATA_FILE, "a", newline="") as f:
                csv.writer(f).writerow([ts, cnt.get("temp"), cnt.get("hum"), cnt.get("soil")])
            app.logger.info(f"Fetched @ {ts}: {cnt}")
        except Exception as ex:
            app.logger.error(f"Fetch error: {ex}")
        time.sleep(10)

def call_esp_async(vol: float):
    """Gọi ESP /pump?vol=… bất đồng bộ."""
    try:
        requests.get(f"http://{ESP_IP}/pump", params={'vol': vol}, timeout=3)
        app.logger.info(f"[Async] ESP pump called vol={vol}")
    except Exception as e:
        app.logger.error(f"[Async] ESP call error: {e}")

@app.route('/pump_and_predict', methods=['POST'])
def pump_and_predict():
    data = request.get_json() or {}
    t = data.get("temperature", 0)
    h = data.get("humidity_env", 0)
    s = data.get("soil_moisture", 0)

    # 1) Predict volume
    feats = np.array([[t, h, s]])
    vol = float(pipeline.predict(feats)[0])
    vol = float(np.clip(vol, 0, 50))
    action = "Cần tưới" if vol >= THRESHOLD_VOL else "Không cần tưới"

    # 2) Nếu cần bơm, spawn thread gọi ESP, nhưng không block
    if vol >= THRESHOLD_VOL:
        threading.Thread(target=call_esp_async, args=(vol,), daemon=True).start()
        pump_status = "sent"
    else:
        pump_status = "skipped (<5ml)"

    # 3) Cập nhật tình trạng để /status trả về
    latest.update({
        'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'temperature':   t,
        'humidity_env':  h,
        'soil_moisture': s,
        'volume_ml':     vol,
        'action':        action,
        'pump_status':   pump_status
    })

    return jsonify(latest)

@app.route('/status')
def status():
    return jsonify(latest)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    # 1) Chạy background thread pull Dweet
    threading.Thread(target=fetch_and_store, daemon=True).start()
    # 2) Chạy Flask
    app.run(host='0.0.0.0', port=5000, debug=True)
