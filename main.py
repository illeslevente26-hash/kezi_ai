import requests
from datetime import datetime
import numpy as np
from scipy.stats import poisson

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Szűrendő tornák és kulcsszavak (U18 Euro, Championship B divízió, BL, VB, EB)
TARGET_KEYWORDS = [
    "champions league", "ehf", "world championship", "european championship",
    "olympics", "u18", "u19", "u20", "u21", "u22", "m18", "w18",
    "championship i", "championship ii", "euro"
]

def is_target_event(tournament_name):
    """Lekéri, hogy az adott torna a szűrt kategóriákba (U18+ EB/VB/BL/Championship) tartozik-e."""
    name_lower = tournament_name.lower()
    return any(keyword in name_lower for keyword in TARGET_KEYWORDS)

def fetch_handball_matches():
    """Lekéri az aznapi kézilabda mérkőzéseket kibővített kategóriákkal."""
    url = "https://site.api.espn.com/apis/site/v2/sports/handball/daily-schedule"
    matches = []
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            
            for event in events:
                tournament = event.get('name', 'Kézilabda Torna')
                competitions = event.get('competitions', [])
                
                for comp in competitions:
                    competitors = comp.get('competitors', [])
                    if len(competitors) == 2:
                        home = competitors[0].get('team', {}).get('displayName', 'Hazai')
                        away = competitors[1].get('team', {}).get('displayName', 'Vendég')
                        
                        # Ha az ESPN nem választja szét jól a neveket
                        if home == "Hazai" or away == "Vendég":
                            title = event.get('name', '')
                            if " vs " in title or " at " in title:
                                parts = title.replace(" at ", " vs ").split(" vs ")
                                home, away = parts[0].strip(), parts[1].strip()

                        # Formastatisztika lekérése
                        home_records = competitors[0].get('records', [])
                        away_records = competitors[1].get('records', [])
                        home_stat = home_records[0].get('summary', '3-1') if home_records else "3-1"
                        away_stat = away_records[0].get('summary', '2-2') if away_records else "2-2"

                        # Szűrés
                        if is_target_event(tournament) or is_target_event(event.get('shortName', '')):
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

def calculate_handball_odds(home, away, home_stat, away_stat):
    """Poisson-alapú esélybecslő modell kézilabdára."""
    base_home_goals = 28.5
    base_away_goals = 26.5
    
    try:
        h_wins = int(home_stat.split('-')[0]) if '-' in home_stat else 2
        a_wins = int(away_stat.split('-')[0]) if '-' in away_stat else 2
        exp_home = base_home_goals + (h_wins - a_wins) * 0.7
        exp_away = base_away_goals + (a_wins - h_wins) * 0.7
    except Exception:
        exp_home, exp_away = base_home_goals, base_away_goals

    goals = np.arange(0, 50)
    matrix = np.outer(poisson.pmf(goals, exp_home), poisson.pmf(goals, exp_away))

    p_home = np.sum(np.tril(matrix, -1)) * 100
    p_draw = np.sum(np.diag(matrix)) * 100
    p_away = np.sum(np.triu(matrix, 1)) * 100

    return (f"  -> Várható gólszám: {home}: {exp_home:.1f} | {away}: {exp_away:.1f}\n"
            f"  -> Esélyek: {home}: {p_home:.1f}% | Döntetlen: {p_draw:.1f}% | {away}: {p_away:.1f}%")

if __name__ == "__main__":
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"=== PRÉMIUM ÉS UTÁNPÓTLÁS KÉZILABDA (U18+ EB/VB, BL) AI ELEMZÉS ({today_str}) ===\n")
    
    matches = fetch_handball_matches()
    
    if not matches:
        print("A mai napon nem található szűrt kategóriájú kézilabda mérkőzés az API műsorában.")
    else:
        print(f"Összesen {len(matches)} mérkőzés található a műsorban:\n")
        for match in matches:
            print(f"[{match['league']}] {match['home']} vs {match['away']}")
            print(calculate_handball_odds(
                match['home'], match['away'], match['home_stat'], match['away_stat']
            ) + "\n")
