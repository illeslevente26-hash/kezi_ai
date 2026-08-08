import requests
from datetime import datetime
import numpy as np
from scipy.stats import poisson

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Szűrendő kiemelt bajnokságok és kulcsszavak
TARGET_LEAGUES = [
    "champions league", "ehf cl", "world championship", "european championship",
    "olympics", "olympic games", "u18", "u19", "u20", "u21", "u22", "u23",
    "world cup", "euro", "handball cl"
]

def is_target_competition(tournament_name):
    """Kiszűri, hogy a meccs a Bajnokok Ligája vagy Világesemény kategóriába tartozik-e."""
    name_lower = tournament_name.lower()
    return any(keyword in name_lower for keyword in TARGET_LEAGUES)

def fetch_handball_data():
    """Lekéri az aznapi kézilabda mérkőzéseket az ESPN globális adatfolyamából."""
    url = "https://site.api.espn.com/apis/site/v2/sports/handball/daily-schedule"
    matches = []
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            
            for event in events:
                tournament = event.get('name', 'Kézilabda Torna')
                
                # Bajnokság/Kategória szűrése
                if is_target_competition(tournament):
                    competitions = event.get('competitions', [])
                    for comp in competitions:
                        competitors = comp.get('competitors', [])
                        if len(competitors) == 2:
                            home = competitors[0].get('team', {}).get('displayName', 'Hazai')
                            away = competitors[1].get('team', {}).get('displayName', 'Vendég')
                            
                            # Formastatisztikák lekérése a csapatoz
                            home_records = competitors[0].get('records', [])
                            away_records = competitors[1].get('records', [])
                            
                            home_stat = home_records[0].get('summary', '0-0') if home_records else "5-0"
                            away_stat = away_records[0].get('summary', '0-0') if away_records else "3-2"

                            matches.append({
                                'home': home,
                                'away': away,
                                'league': tournament,
                                'home_stat': home_stat,
                                'away_stat': away_stat
                            })
    except Exception as e:
        print(f"Lekérdezési hiba: {e}")
        
    return matches

def calculate_advanced_handball_odds(home, away, home_stat, away_stat):
    """
    Kézilabda esélybecslő Poisson-modell statisztikai súlyozással.
    A korábbi győzelmi mérleg alapján módosítja a várható gólszámokat.
    """
    base_home_goals = 29.0
    base_away_goals = 27.0
    
    # Statisztikai finomhangolás a csapatok mérlege alapján
    try:
        h_wins = int(home_stat.split('-')[0]) if '-' in home_stat else 3
        a_wins = int(away_stat.split('-')[0]) if '-' in away_stat else 2
        exp_home = base_home_goals + (h_wins - a_wins) * 0.8
        exp_away = base_away_goals + (a_wins - h_wins) * 0.8
    except Exception:
        exp_home, exp_away = base_home_goals, base_away_goals

    goals = np.arange(0, 50)
    matrix = np.outer(poisson.pmf(goals, exp_home), poisson.pmf(goals, exp_away))

    p_home = np.sum(np.tril(matrix, -1)) * 100
    p_draw = np.sum(np.diag(matrix)) * 100
    p_away = np.sum(np.triu(matrix, 1)) * 100

    return (f"  -> Statisztikai forma: {home} ({home_stat}) vs {away} ({away_stat})\n"
            f"  -> Várható gólok: {home}: {exp_home:.1f} | {away}: {exp_away:.1f}\n"
            f"  -> Esélyek: {home}: {p_home:.1f}% | Döntetlen: {p_draw:.1f}% | {away}: {p_away:.1f}%")

if __name__ == "__main__":
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"=== PRÉMIUM KÉZILABDA (BL, VB, EB, OLIMPIA U18+) AI ELEMZÉS ({today_str}) ===\n")
    
    matches = fetch_handball_data()
    
    if not matches:
        print("A mai napon nem található Bajnokok Ligája vagy Világesemény kézilabda mérkőzés.")
    else:
        print(f"Összesen {len(matches)} kiemelt mérkőzés található a mai napon:\n")
        for match in matches:
            print(f"[{match['league']}] {match['home']} vs {match['away']}")
            print(calculate_advanced_handball_odds(
                match['home'], match['away'], match['home_stat'], match['away_stat']
            ) + "\n")
