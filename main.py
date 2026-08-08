import random
from datetime import datetime

def load_data():
    # Egyszerű tesztadatok
    return [
        {"team": "A", "score": 1.2},
        {"team": "B", "score": 0.9},
        {"team": "C", "score": 1.5},
    ]

def predict(data):
    results = []
    for item in data:
        team = item.get("team", "Ismeretlen")
        score = item.get("score", 0)

        if score >= 1.3:
            prediction = "Erős"
        elif score >= 1.0:
            prediction = "Közepes"
        else:
            prediction = "Gyenge"

        results.append(f"{team}: {prediction} (score={score})")
    return results

def main():
    print("=== AI futás indult ===")
    print("Időpont:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    try:
        data = load_data()
        if not data:
            print("Nincs feldolgozható adat.")
            return

        results = predict(data)

        print("Eredmények:")
        for line in results:
            print("-", line)

        print("Kész.")

    except Exception as e:
        print("Hiba történt:", str(e))

if __name__ == "__main__":
    main()