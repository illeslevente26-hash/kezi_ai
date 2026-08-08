import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Szűrendő kulcsszavak (U18, B-divízió, EHF, BL, VB, EB, Olimpia)
TARGET_KEYWORDS = [
    "champions league", "ehf", "world championship", "european championship",
    "olympic", "u18", "u19", "u20", "u21", "u22", "m18", "w18",
    "championship", "euro", "division b"
]

def is_target_event(tournament_name):
    """Ellenőrzi, hogy a torna a szűrt kategóriákba tartozik-e."""
    name_lower = tournament_name.lower()
    return any(keyword in name_lower for keyword in TARGET_KEYWORDS)

def get_handball_matches():
    """Lekéri a mai nap kézilabda meccseit a Sofascore adatbázisából."""
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://api.sofascore.com/api/v1/sport/handball/scheduled-events/{today}"
    matches = []
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            events = response.json().get('events', [])
            for event in events:
                tournament = event.get('tournament', {}).get('name', 'Kézilabda Torna')
                category = event.get('tournament', {}).get('category', {}).get('name', '')
                full_tour_name = f"{category} {tournament}"
                
                # Szűrés a kiemelt és utánpótlás tornákra
                if is_target_event(full_tour_name):
                    home = event.get('homeTeam', {}).get('name', 'Hazai')
                    away = event.get('awayTeam', {}).get('name', 'Vendég')
                    
                    matches.append({
                        'home': home,
                        'away': away,
                        'league': full_tour_name
                    })
    except Exception as e:
        print(f"Lekérdezési hiba: {e}")

    # Tartalék lekérés nyílt adatsorokból, ha a GitHub szerverét átmenetileg korlátozzák
    if not matches:
        try:
            res = requests.get("https://site.api.espn.com/apis/site/v2/sports/handball/daily-schedule", headers=HEADERS, timeout=10)
            if res.status_code == 200:
                for ev in res.json().get('events', []):
                    tour = ev.get('name', 'Handball')
                    for comp in ev.get('competitions', []):
                        comps = comp.get('competitors', [])
                        if len(comps) == 2:
                            h = comps[0].get('team', {}).get('displayName', '')
                            a = comps[1].get('team', {}).get('displayName', '')
                            if h and a:
                                matches.append({
                                    'home': h,
                                    'away': a,
                                    'league': tour
                                })
        except Exception:
            pass

    return matches

def calculate_handball_odds(home, away):
    """Poisson-eloszlás alapú kézilabda valószínűség számítás."""
    exp_home = 28.5
    exp_away = 26.5

    goals = np.arange(0, 50)
    matrix = np.outer(poisson.pmf(goals, exp_home), poisson.pmf(goals, exp_away))

    p_home = np.sum(np.tril(matrix, -1)) * 100
    p_draw = np.sum(np.diag(matrix)) * 100
    p_away = np.sum(np.triu(matrix, 1)) * 100

    return f"  -> Esélyek: {home}: {p_home:.1f}% | Döntetlen: {p_draw:.1f}% | {away}: {p_away:.1f}%"

if __name__ == "__main__":
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"=== KÉZILABDA (BL, U18+ EB/VB/CHAMPIONSHIP) AI ELEMZÉS ({today_str}) ===\n")
    
    matches = get_handball_matches()
    
    if not matches:
        print("A mai napon nem található szűrt kézilabda mérkőzés a műsorban.")
    else:
        print(f"Összesen {len(matches)} mérkőzés található a műsorban:\n")
        for match in matches:
            print(f"[{match['league']}] {match['home']} vs {match['away']}")
            print(calculate_handball_odds(match['home'], match['away']) + "\n")
