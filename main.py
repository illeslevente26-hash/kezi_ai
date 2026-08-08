import csv
import json
import math
from datetime import datetime
from pathlib import Path

DATA_FILE = Path("matches.csv")
OUTPUT_FILE = Path("predictions.json")

# -----------------------------
# Utils
# -----------------------------
def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return default

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def normalize_probs(home, draw, away):
    total = home + draw + away
    if total <= 0:
        return 1/3, 1/3, 1/3
    return home / total, draw / total, away / total

# -----------------------------
# Data loading
# -----------------------------
def load_matches(file_path=DATA_FILE):
    matches = []

    if not file_path.exists():
        print(f"Hiba: nem található a fájl: {file_path}")
        return matches

    try:
        with file_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])

            required = {
                "home", "away", "competition",
                "home_form", "away_form", "h2h",
                "home_goals_avg", "away_goals_avg"
            }

            if not required.issubset(fieldnames):
                print("Hiba: hiányzó oszlopok a CSV-ben.")
                print("Szükséges oszlopok:")
                print("home, away, competition, home_form, away_form, h2h, home_goals_avg, away_goals_avg")
                return matches

            for row in reader:
                home = str(row.get("home", "")).strip()
                away = str(row.get("away", "")).strip()
                competition = str(row.get("competition", "")).strip().lower()

                if not home or not away:
                    print(f"Figyelmeztetés: üres csapatnév, kihagyva: {row}")
                    continue

                match = {
                    "home": home,
                    "away": away,
                    "competition": competition,
                    "home_form": safe_float(row.get("home_form")),
                    "away_form": safe_float(row.get("away_form")),
                    "h2h": clamp(safe_float(row.get("h2h")), 0.0, 1.0),
                    "home_goals_avg": safe_float(row.get("home_goals_avg")),
                    "away_goals_avg": safe_float(row.get("away_goals_avg")),
                }
                matches.append(match)

    except Exception as e:
        print(f"Hiba a CSV olvasás közben: {e}")

    return matches

# -----------------------------
# Competition logic
# -----------------------------
def competition_profile(comp):
    comp = (comp or "").lower()

    if comp in {"bl", "champions league", "ehf champions league"}:
        return {
            "mult": 1.15,
            "home_adv": 0.10,
            "draw_bias": 0.10,
            "goals_base": 48.5
        }

    if comp in {"el", "europe league", "ehf european league"}:
        return {
            "mult": 1.08,
            "home_adv": 0.08,
            "draw_bias": 0.12,
            "goals_base": 47.5
        }

    if comp in {"international", "national team", "world cup", "euro", "olympics"}:
        return {
            "mult": 1.10,
            "home_adv": 0.05,
            "draw_bias": 0.14,
            "goals_base": 49.0
        }

    return {
        "mult": 1.00,
        "home_adv": 0.07,
        "draw_bias": 0.12,
        "goals_base": 47.0
    }

# -----------------------------
# Prediction model
# -----------------------------
def team_power(home_form, away_form, h2h, goals_avg, side="home", home_adv=0.0):
    # side: "home" or "away"
    if side == "home":
        return (
            home_form * 0.35 +
            h2h * 0.22 +
            goals_avg * 0.18 +
            home_adv * 0.15 +
            (1.0 - away_form) * 0.10
        )
    else:
        return (
            away_form * 0.35 +
            (1.0 - h2h) * 0.22 +
            goals_avg * 0.18 +
            (1.0 - home_form) * 0.10
        )

def calculate_probabilities(match):
    comp = competition_profile(match["competition"])

    home_power = team_power(
        match["home_form"],
        match["away_form"],
        match["h2h"],
        match["home_goals_avg"],
        side="home",
        home_adv=comp["home_adv"]
    ) * comp["mult"]

    away_power = team_power(
        match["home_form"],
        match["away_form"],
        match["h2h"],
        match["away_goals_avg"],
        side="away",
        home_adv=comp["home_adv"]
    ) * comp["mult"]

    diff = home_power - away_power

    # 1X2 raw probabilities
    home_win = sigmoid(diff * 5.1)
    away_win = sigmoid(-diff * 5.1)

    # draw is more likely in balanced matches, especially internationals
    balance = abs(diff)
    draw = comp["draw_bias"] + (0.18 - balance * 0.42)
    draw = clamp(draw, 0.05, 0.25)

    home_win, draw, away_win = normalize_probs(home_win, draw, away_win)

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "diff": diff,
        "home_power": home_power,
        "away_power": away_power,
        "comp": comp,
    }

def estimate_goals(match, probs):
    comp = competition_profile(match["competition"])
    base = comp["goals_base"]

    form_factor = (match["home_form"] + match["away_form"]) / 2.0
    h2h_factor = abs(match["h2h"] - 0.5)
    pace = (match["home_goals_avg"] + match["away_goals_avg"]) / 2.0

    goals = (
        base
        + (pace * 0.35)
        + (form_factor * 2.5)
        + (h2h_factor * -1.2)
        + (abs(probs["diff"]) * 2.0)
    )

    return int(round(goals))

def over_markets(goals):
    return {
        "over_50_5": goals >= 51,
        "over_55_5": goals >= 56,
        "over_59_5": goals >= 60,
    }

def value_score(best_prob, confidence, goals):
    # egyszerű, stabil rangsorlogika
    return round((best_prob * 100) * 0.55 + confidence * 0.35 + max(0, goals - 50) * 0.10, 2)

def make_prediction(match):
    probs = calculate_probabilities(match)
    goals = estimate_goals(match, probs)

    options = [
        {"pick": "1", "prob": probs["home_win"]},
        {"pick": "X", "prob": probs["draw"]},
        {"pick": "2", "prob": probs["away_win"]},
    ]
    best = max(options, key=lambda x: x["prob"])
    confidence = round(best["prob"] * 100, 1)

    if best["pick"] == "1":
        tip_text = f"1 — hazai győzelem ({match['home']})"
    elif best["pick"] == "2":
        tip_text = f"2 — vendég győzelem ({match['away']})"
    else:
        tip_text = "X — döntetlen"

    overs = over_markets(goals)
    strongest_market = "1X2"

    prediction = {
        "match": f"{match['home']} - {match['away']}",
        "competition": match["competition"],
        "tip": tip_text,
        "pick": best["pick"],
        "confidence": confidence,
        "probabilities": {
            "1": round(probs["home_win"] * 100, 1),
            "X": round(probs["draw"] * 100, 1),
            "2": round(probs["away_win"] * 100, 1),
        },
        "goals": goals,
        "markets": {
            "over_50_5": overs["over_50_5"],
            "over_55_5": overs["over_55_5"],
            "over_59_5": overs["over_59_5"],
        },
        "home_power": round(probs["home_power"], 4),
        "away_power": round(probs["away_power"], 4),
        "diff": round(probs["diff"], 4),
        "value": value_score(best["prob"], confidence, goals),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strongest_market": strongest_market
    }

    return prediction

# -----------------------------
# Output
# -----------------------------
def save_json(predictions, file_path=OUTPUT_FILE):
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Figyelmeztetés: JSON mentés sikertelen: {e}")

def print_prediction(p):
    print(f"Meccs: {p['match']}")
    print(f"Verseny: {p['competition']}")
    print(f"Tipp: {p['tip']}")
    print(f"Bizalom: {p['confidence']}%")
    print(f"Esélyek: 1={p['probabilities']['1']}% | X={p['probabilities']['X']}% | 2={p['probabilities']['2']}%")
    print(f"Várható összgól: {p['goals']}")
    print(f"Piacok: O50.5={p['markets']['over_50_5']} | O55.5={p['markets']['over_55_5']} | O59.5={p['markets']['over_59_5']}")
    print(f"Value score: {p['value']}")
    print("-" * 55)

def main():
    print("=== Profi Kézilabda Tippelő AI (BL / Nemzetközi) ===")
    print("Indítva:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()

    matches = load_matches()
    if not matches:
        print("Nincs feldolgozható meccsadat.")
        return

    predictions = [make_prediction(m) for m in matches]

    # rangsorolás value szerint
    predictions.sort(key=lambda x: x["value"], reverse=True)

    for p in predictions:
        print_prediction(p)

    save_json(predictions)

    print(f"Elkészült: {len(predictions)} tipp")
    print(f"Mentve ide: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()