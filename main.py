import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
from scipy.stats import poisson
from datetime import datetime

def get_live_real_handball_matches():
    """Valódi aznapi és élő kézilabda meccseket kér le nyílt adatfolyamból."""
    url = "https://www.scorespro.com/rss2/live-handball.xml"
    matches = []
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        for item in root.findall('.//item'):
            title = item.find('title').text if item.find('title') is not None else ""
            # Címsor formátuma: "Team A vs Team B"
            if " vs " in title:
                parts = title.split(" vs ")
                home = parts[0].replace("(*)", "").strip()
                away = parts[1].replace("(*)", "").strip()
                
                category = item.find('category').text if item.find('category') is not None else "Kézilabda Bajnokság"
                matches.append({
                    'home_team': home,
                    'away_team': away,
                    'league': category
                })
    except Exception as e:
        print(f"Adatlekérdezési hiba: {e}")
        
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
        for match in today_matches[:15]: # Az első 15 legfrissebb meccs
            print(f"[{match['league']}] {match['home_team']} vs {match['away_team']}")
            print(f"  -> {calculate_handball_odds(match['home_team'], match['away_team'])}\n")
