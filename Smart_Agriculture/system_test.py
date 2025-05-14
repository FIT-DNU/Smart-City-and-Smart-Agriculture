import csv
import random
import requests

# URL Flask API của bạn
FLASK_URL = "http://127.0.0.1:5000/pump_and_predict"

def load_samples(csv_path="test.csv"):
    samples = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "temperature":   float(row["temperature"]),
                "humidity_env":  float(row["humidity_env"]),
                "soil_moisture": float(row["soil_moisture"]),
                "volume_ml":     float(row["volume_ml"])
            })
    return samples

def choose_sample_by_volume(samples, vol_value):
    matches = [s for s in samples if s["volume_ml"] == vol_value]
    if not matches:
        return None
    return random.choice(matches)

def send_to_flask(sample):
    payload = {
        "temperature":   sample["temperature"],
        "humidity_env":  sample["humidity_env"],
        "soil_moisture": sample["soil_moisture"]
    }
    print(f"\\n→ Gửi payload: {payload}")
    try:
        r = requests.post(FLASK_URL, json=payload, timeout=5)
        r.raise_for_status()
        result = r.json()
        print("← Kết quả dự đoán của mô hình AI từ Flask:")
        print(f"   • volume_ml:   {result.get('volume_ml')}")
        print(f"   • action:      {result.get('action')}")
        print(f"   • pump_status: {result.get('pump_status')}")
    except Exception as e:
        print("⚠️ Lỗi khi gửi tới Flask:", e)

def main():
    samples = load_samples("test.csv")
    if not samples:
        print("Không tìm thấy mẫu nào trong test.csv")
        return

    vols = sorted(set(s["volume_ml"] for s in samples))
    print("Các giá trị volume_ml khả dụng:", vols)

    while True:
        entry = input("\\Gửi mỗi mẫu sau 5s, nhập volume_ml muốn gửi (hoặc 'exit' để thoát): ").strip()
        if entry.lower() in ("exit", "quit"):
            break
        try:
            vol = float(entry)
        except ValueError:
            print("Vui lòng nhập số hợp lệ.")
            continue

        sample = choose_sample_by_volume(samples, vol)
        if sample is None:
            print(f"Không tìm thấy mẫu với volume_ml = {vol}. Thử lại.")
            continue

        print("Chọn mẫu:", sample)
        send_to_flask(sample)
        print("-" * 50)

if __name__ == "__main__":
    main()