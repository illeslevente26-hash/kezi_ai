import requests
from bs4 import BeautifulSoup
import numpy as np
from scipy.stats import poisson
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_live_real_handball_matches():
    """Tisztított RSS feldolgozó a kézilabda meccsekhez."""
    url = "https://www.scorespro.com/rss2/live-handball.xml"
    matches = []
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            # BeautifulSoup kezeli a specifikus karaktereket az XML-ben
            soup = BeautifulSoup(response.content, 'html.parser')
            items = soup.find_all('item')
            
            for item in items:
                title = item.find('title').text if item.find('title') else ""
                if " vs " in title:
                    parts = title.split(" vs ")
                    home = parts[0].replace("(*)", "").strip()
                    away = parts[1].replace("(*)", "").strip()
                    
                    category = item.find('category').text if item.find('category') else "Kézilabda Bajnokság"
                    matches.append({
                        'home_team': home,
                        'away_team': away,
                        'league': category
                    })
    except Exception as e:
        print(f"Lekérdezési hiba: {e}")
        
    return matches

def calculate_handball_odds(home_team, away_team):
    """Poisson-eloszlású kézilabda elemzés."""
    exp_home = 28.5
    exp_away = 26.8

    goals = np.arange(0, 50)
    matrix = np.outer(poisson.pmf(goals, exp_home), poisson.pmf(goals, exp_away))

    p_home = np.sum(np.tril(matrix, -1)) * 100
    p_draw = np.sum(np.diag(matrix)) * 100
    p_away = np.sum(np.triu(matrix, 1)) * 100

    return f"Esélyek: {home_team}: {p_home:.1f}% | Döntetlen: {p_draw:.1f}% | {away_team}: {p_away:.1f}%"

if __name__ == "__main__":
    today_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"=== ÉLŐ/AZNAPI KÉZILABDA MÉRKŐZÉSEK ELEMZÉSE ({today_str}) ===\n")
    
    today_matches = get_live_real_handball_matches()
    
    if not today_matches:
        print("Jelenleg egyetlen élő/aznapi kézilabda mérkőzés sem érhető el az adatfolyamban.")
    else:
        print(f"Összesen {len(today_matches)} valódi mérkőzés található az élő adatfolyamban:\n")
        for match in today_matches[:15]:
            print(f"[{match['league']}] {match['home_team']} vs {match['away_team']}")
            print(f"  -> {calculate_handball_odds(match['home_team'], match['away_team'])}\n")
