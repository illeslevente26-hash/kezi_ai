import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def get_today_ehf_matches():
    """Lekéri az EHF hivatalos rendszeréből az összes aznapi mérkőzést."""
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://api.ehfcl.com/v1/matches?date={today}"
    
    matches = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for match in data.get('matches', []):
                matches.append({
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'league': match.get('competitionName', 'EHF')
                })
    except Exception as e:
        print(f"Lekérdezési hiba: {e}")
        
    return matches

def calculate_odds(home_team, away_team):
    """Poisson-eloszlás alapú valószínűség számítás."""
    # Kézilabda átlagok (hazai ~29 gól, vendég ~27 gól)
    exp_home = 29.2
    exp_away = 27.5

    goals = np.arange(0, 50)
    matrix = np.outer(poisson.pmf(goals, exp_home), poisson.pmf(goals, exp_away))

    p_home = np.sum(np.tril(matrix, -1)) * 100
    p_draw = np.sum(np.diag(matrix)) * 100
    p_away = np.sum(np.triu(matrix, 1)) * 100

    return f"Esélyek: {home_team}: {p_home:.1f}% | Döntetlen: {p_draw:.1f}% | {away_team}: {p_away:.1f}%"

if __name__ == "__main__":
    print(f"=== AZNAPI KÉZILABDA MÉRKŐZÉSEK AI ELEMZÉSE ({datetime.now().strftime('%Y-%m-%d')}) ===\n")
    
    today_matches = get_today_ehf_matches()
    
    if not today_matches:
        print("A mai napon nincsenek hivatalos EHF mérkőzések a rendszerben.")
    else:
        for match in today_matches:
            print(f"[{match['league']}] {match['home_team']} vs {match['away_team']}")
            print(f"  -> {calculate_odds(match['home_team'], match['away_team'])}\n")
