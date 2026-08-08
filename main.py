import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from scipy.stats import poisson

# Böngésző szimuláció (User-Agent), hogy a weboldal ne blokkolja a kérést
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def scrape_handball_matches(url):
    """
    Közvetlenül kiszedi a meccsek adatait a megadott weboldal HTML szerkezetéből.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"Hiba az oldal letöltésekor: Status {response.status_code}")
            return pd.DataFrame()

        soup = BeautifulSoup(response.text, 'html.parser')
        matches = []

        # Megkeressük a meccs-blokkokat a HTML-ben (példa struktúra alapján)
        for item in soup.select('.match-row, .game-item'):
            try:
                home = item.select_one('.home-team, .team-home').text.strip()
                away = item.select_one('.away-team, .team-away').text.strip()
                score_text = item.select_one('.score, .result').text.strip()
                
                # Gólok szétválasztása (pl. "31 - 28")
                if '-' in score_text:
                    h_goals, a_goals = map(int, score_text.split('-'))
                    matches.append({
                        'home_team': home,
                        'away_team': away,
                        'home_goals': h_goals,
                        'away_goals': a_goals
                    })
            except Exception:
                continue

        return pd.DataFrame(matches)

    except Exception as e:
        print(f"Hiba történt a scraping során: {e}")
        return pd.DataFrame()

def predict_match(df, home_team, away_team):
    """Kiszámolja a meccs kimeneteli valószínűségeit a gyűjtött adatokból."""
    if df.empty or len(df) < 3:
        return "Nincs elég felhalmozott adat a megbízható elemzéshez."

    lh_avg = df['home_goals'].mean()
    la_avg = df['away_goals'].mean()

    # Csapatok átlagai
    h_matches = df[df['home_team'] == home_team]
    a_matches = df[df['away_team'] == away_team]

    exp_home = (h_matches['home_goals'].mean() if not h_matches.empty else lh_avg)
    exp_away = (a_matches['away_goals'].mean() if not a_matches.empty else la_avg)

    # Valószínűségi mátrix generálása
    goals = np.arange(0, 50)
    matrix = np.outer(poisson.pmf(goals, exp_home), poisson.pmf(goals, exp_away))

    p_home = np.sum(np.tril(matrix, -1)) * 100
    p_draw = np.sum(np.diag(matrix)) * 100
    p_away = np.sum(np.triu(matrix, 1)) * 100

    return (f"Várható gólok: {home_team} {exp_home:.1f} - {exp_away:.1f} {away_team} | "
            f"Hazai: {p_home:.1f}% | Döntetlen: {p_draw:.1f}% | Vendég: {p_away:.1f}%")

if __name__ == "__main__":
    # Céloldal URL (példa nyilvános kézilabda eredmény oldal)
    URL_EHF = "https://www.eurohandball.com/en/matches/"
    
    print("=== ADATOK GYŰJTÉSE A WEBOLDALRÓL ===")
    df_matches = scrape_handball_matches(URL_EHF)
    
    # Ha a scraping épp nem talált adatot a megadott szelektorokkal, tartalék mintát használ
    if df_matches.empty:
        print("A weboldal szerkezete megváltozott vagy nincs élő meccs, tartalék adatkészlet betöltése...")
        df_matches = pd.DataFrame({
            'home_team': ['Veszprém', 'Magdeburg', 'Szeged', 'Füchse Berlin'],
            'away_team': ['Barcelona', 'PSG', 'Kielce', 'THW Kiel'],
            'home_goals': [32, 30, 28, 31],
            'away_goals': [29, 28, 27, 28]
        })

    print(f"Sikeresen feldolgozva: {len(df_matches)} mérkőzés.\n")
    print("=== AI ELŐREJELZÉS ===")
    print(predict_match(df_matches, 'Veszprém', 'Barcelona'))
