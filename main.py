import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime
from zoneinfo import ZoneInfo

TIMEZONE = "Europe/Budapest"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# --- ADATFORRÁSOK ---

def get_matches():
    """
    Egyszerű meccslista.
    Itt később API vagy scrape forrás jöhet.
    """
    return [
        {"home": "Magyarország", "away": "Szlovákia", "league": "Kézilabda EB"},
        {"home": "Dánia", "away": "Svédország", "league": "Nemzetközi barátságos"},
        {"home": "Németország", "away": "Franciaország", "league": "Világbajnokság"}
    ]

def get_team_stats(team):
    """
    Példa statisztika.
    Később ezt valós adatra kell cserélni.
    """
    stats = {
        "Magyarország": {"gf": 29.2, "ga": 27.8},
        "Szlovákia": {"gf": 26.1, "ga": 28.9},
        "Dánia": {"gf": 32.4, "ga": 26.0},
        "Svédország": {"gf": 30.1, "ga": 27.1},
        "Németország": {"gf": 31.0, "ga": 28.2},
        "Franciaország": {"gf": 30.4, "ga": 27.4},
    }
    return stats.get(team, {"gf": 28.0, "ga": 28.0})

# --- MODELL ---

def calculate_expected_goals(home, away):
    home_stats = get_team_stats(home)
    away_stats = get_team_stats(away)

    # Hazai/idegen előny egyszerű beépítése
    home_attack = home_stats["gf"]
    home_defense = home_stats["ga"]
    away_attack = away_stats["gf"]
    away_defense = away_stats["ga"]

    exp_home = (home_attack + away_defense) / 2 + 1.2
    exp_away = (away_attack + home_defense) / 2 - 0.8

    exp_home = max(20.0, min(exp_home, 40.0))
    exp_away = max(20.0, min(exp_away, 40.0))

    return exp_home, exp_away

def calculate_1x2(expected_home, expected_away):
    goals = np.arange(0, 71)
    matrix = np.outer(
        poisson.pmf(goals, expected_home),
        poisson.pmf(goals, expected_away)
    )

    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()
    total = home_win + draw + away_win

    return {
        "home": home_win / total,
        "draw": draw / total,
        "away": away_win / total,
    }

def calculate_over_under(expected_home, expected_away, lines=None):
    if lines is None:
        lines = [49.5, 51.5, 53.5, 55.5]

    total_expected = expected_home + expected_away
    results = {}

    for line in lines:
        under = poisson.cdf(int(line - 0.5), total_expected)
        over = 1 - under
        results[line] = {"over": over, "under": under}

    return results

def calculate_value(probability, odds):
    """
    Egyszerű value számítás.
    pozitív érték = jó ajánlat
    """
    return probability * odds - 1

def pick_best_bet(probs, ou, odds_data):
    candidates = []

    # 1X2 piac
    market_map = {
        "home": "1",
        "draw": "X",
        "away": "2"
    }

    for key, odd_key in market_map.items():
        if odd_key in odds_data:
            prob = probs[key]
            odds = odds_data[odd_key]
            value = calculate_value(prob, odds)
            candidates.append({
                "market": f"1X2 - {odd_key}",
                "probability": prob,
                "odds": odds,
                "value": value
            })

    # Over/Under piac
    for line, vals in ou.items():
        over_key = f"Over {line}"
        under_key = f"Under {line}"

        if over_key in odds_data:
            prob = vals["over"]
            odds = odds_data[over_key]
            value = calculate_value(prob, odds)
            candidates.append({
                "market": over_key,
                "probability": prob,
                "odds": odds,
                "value": value
            })

        if under_key in odds_data:
            prob = vals["under"]
            odds = odds_data[under_key]
            value = calculate_value(prob, odds)
            candidates.append({
                "market": under_key,
                "probability": prob,
                "odds": odds,
                "value": value
            })

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["value"], reverse=True)
    return candidates[0]

def percentage(value):
    return f"{value * 100:.2f}%"

# --- PÉLDA ODDS ---

def get_mock_odds(match):
    """
    Itt később valós odds API jön.
    """
    return {
        "1": 2.10,
        "X": 10.00,
        "2": 2.85,
        "Over 49.5": 1.85,
        "Under 49.5": 1.85,
        "Over 51.5": 1.95,
        "Under 51.5": 1.75,
        "Over 53.5": 2.05,
        "Under 53.5": 1.68,
        "Over 55.5": 2.25,
        "Under 55.5": 1.58,
    }

# --- ELEMZÉS ---

def analyze_match(match):
    home, away = match["home"], match["away"]

    print("\n" + "=" * 70)
    print(f"{home} - {away}")
    print(f"Verseny: {match['league']}")
    print("=" * 70)

    exp_home, exp_away = calculate_expected_goals(home, away)
    probs = calculate_1x2(exp_home, exp_away)
    ou = calculate_over_under(exp_home, exp_away)
    odds = get_mock_odds(match)
    best_bet = pick_best_bet(probs, ou, odds)

    print("\nVÁRHATÓ GÓLOK")
    print("-" * 40)
    print(f"{home}: {exp_home:.2f}")
    print(f"{away}: {exp_away:.2f}")
    print(f"Összesen: {exp_home + exp_away:.2f}")

    print("\n1X2 VALÓSZÍNŰSÉGEK")
    print("-" * 40)
    print(f"1 - {home}: {percentage(probs['home'])}")
    print(f"X - Döntetlen: {percentage(probs['draw'])}")
    print(f"2 - {away}: {percentage(probs['away'])}")

    print("\nOVER / UNDER")
    print("-" * 40)
    for line, vals in ou.items():
        print(f"Over {line}: {percentage(vals['over'])} | Under {line}: {percentage(vals['under'])}")

    if best_bet:
        confidence = best_bet["probability"] * 100
        print("\nAJÁNLOTT TIPP")
        print("-" * 40)
        print(f"Piac: {best_bet['market']}")
        print(f"Valószínűség: {confidence:.2f}%")
        print(f"Odds: {best_bet['odds']:.2f}")
        print(f"Value: {best_bet['value']:.3f}")

        if best_bet["value"] > 0.05:
            print("Javaslat: MEGÉRI MEGJÁTSZANI")
        else:
            print("Javaslat: NINCS ERŐS VALUE")

def main():
    now = datetime.now(ZoneInfo(TIMEZONE))
    print("\n" + "=" * 70)
    print("🤾 KEZI_AI - TIPPGENERÁTOR")
    print("=" * 70)
    print(f"Időpont: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    matches = get_matches()
    print(f"\nElemzendő mérkőzések száma: {len(matches)}")

    for match in matches:
        analyze_match(match)

    print("\n" + "=" * 70)
    print("✅ TIPPGENERÁLÁS BEFEJEZVE")
    print("=" * 70)

if __name__ == "__main__":
    main() 

.github/workflows/run.yml
name: Run Tip Generator

on:
  workflow_dispatch:
  schedule:
    - cron: "0 8 * * *"

jobs:
  run-bot:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install requests numpy scipy

      - name: Run main.py
        run: python main.py
