from flask import Flask, request, render_template, jsonify
import joblib, requests, csv
from datetime import datetime

app = Flask(__name__, template_folder='templates')

# --- CẤU HÌNH ---
MODEL_PATH = 'xgb_reg_pipeline.joblib'
ESP_IP     = '192.168.137.244'   # Thay bằng IP của ESP trên mạng hotspot(mở cài đặt phần phát wifi để xem IP esp)
ESP_PORT   = 80
LOG_FILE   = 'sensor_data_log.csv'

# load model
pipeline = joblib.load(MODEL_PATH)

# khởi file log nếu chưa có header
with open(LOG_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp','temperature','humidity_env','soil_moisture','predicted_ml','action'])


def do_prediction(t, h, s):
    raw = pipeline.predict([[t, h, s]])[0]
    vol = float(max(0, min(50, raw)))   # clip 0–50 ml

    # luôn gửi lệnh volume về ESP
    try:
        url = f"http://{ESP_IP}:{ESP_PORT}/pump?volume={vol:.2f}"
        requests.get(url, timeout=2)
        action = 'sent'
    except:
        action = 'error'

    # ghi log
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), t, h, s, vol, action])

    return {
        'temperature':      t,
        'humidity_env':     h,
        'soil_moisture':    s,
        'predicted_volume': vol,
        'action':           action
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        t = float(request.form['temperature'])
        h = float(request.form['humidity_env'])
        s = float(request.form['soil_moisture'])
        result = do_prediction(t, h, s)
    return render_template('index.html', result=result)


@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    t, h, s = map(float, (data['temperature'], data['humidity_env'], data['soil_moisture']))
    return jsonify(do_prediction(t, h, s))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
# trên terminal: python app.py
