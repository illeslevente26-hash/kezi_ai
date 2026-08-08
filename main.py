import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime
from zoneinfo import ZoneInfo
import sys


# ============================================================
# BEÁLLÍTÁSOK
# ============================================================

TIMEZONE = "Europe/Budapest"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}

TIMEOUT = 15


# ============================================================
# SEGÉDFÜGGVÉNYEK
# ============================================================

def clean_name(name):
    """Csapatnév megtisztítása."""

    if not name:
        return ""

    return " ".join(str(name).strip().split())


def get_today():
    """Mai dátum magyar idő szerint."""

    now = datetime.now(
        ZoneInfo(TIMEZONE)
    )

    return now.strftime("%Y-%m-%d")


def percentage(value):
    return f"{value * 100:.2f}%"


# ============================================================
# SOFASCORE - MAI KÉZILABDA
# ============================================================

def get_today_handball_matches():
    """
    Lekéri a mai kézilabda-mérkőzéseket.

    FONTOS:
    Ha az adatforrás nem működik, üres listát adunk vissza.
    NINCS kitalált mérkőzés.
    """

    date = get_today()

    url = (
        "https://www.sofascore.com/api/v1/"
        f"sport/handball/scheduled-events/{date}"
    )

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

    except Exception as error:

        print()
        print("⚠ Nem sikerült elérni a kézilabda adatforrást.")
        print(f"Hiba: {error}")
        print()

        return []

    events = data.get("events", [])

    matches = []

    for event in events:

        try:

            home_team = event.get("homeTeam", {})
            away_team = event.get("awayTeam", {})

            home = clean_name(
                home_team.get("name")
            )

            away = clean_name(
                away_team.get("name")
            )

            if not home or not away:
                continue

            tournament = (
                event
                .get("tournament", {})
                .get("name", "Kézilabda")
            )

            category = (
                event
                .get("tournament", {})
                .get("category", {})
                .get("name", "")
            )

            # Esemény időpontja
            timestamp = event.get("startTimestamp")

            matches.append({
                "id": event.get("id"),
                "home": home,
                "away": away,
                "league": tournament,
                "category": category,
                "timestamp": timestamp,
            })

        except Exception:
            continue

    return matches


# ============================================================
# U18 / EHF / BL SZŰRÉS
# ============================================================

def is_target_match(match):
    """
    Megnézi, hogy a mérkőzés a kívánt kategóriába tartozik-e.

    Nem dobunk ki mindent automatikusan, mert az EHF versenyek
    elnevezése változhat.
    """

    text = (
        f"{match.get('home', '')} "
        f"{match.get('away', '')} "
        f"{match.get('league', '')} "
        f"{match.get('category', '')}"
    ).lower()

    keywords = [
        "ehf",
        "champions league",
        "champions",
        "european league",
        "european cup",
        "u18",
        "u-18",
        "under 18",
        "m18",
    ]

    return any(
        keyword in text
        for keyword in keywords
    )


def filter_matches(matches):
    """Kiszűri a célzott mérkőzéseket."""

    result = []

    for match in matches:

        if is_target_match(match):
            result.append(match)

    return result


# ============================================================
# CSAPATSTATISZTIKÁK
# ============================================================

def get_team_stats():
    """
    IDE kerülnek majd az aktuális csapatstatisztikák.

    Példa:

    "Veszprém": {
        "home_attack": 34.2,
        "home_defense": 28.1,
        "away_attack": 31.7,
        "away_defense": 29.2
    }

    Egyelőre üresen hagyjuk.

    Ha nincs valódi statisztika, a program NEM talál ki számot.
    """

    return {}


# ============================================================
# VÁRHATÓ GÓLOK
# ============================================================

def calculate_expected_goals(
    home,
    away,
    stats
):
    """
    Csapatspecifikus várható gól.
    """

    home_stats = stats.get(home)
    away_stats = stats.get(away)

    if not home_stats or not away_stats:
        return None, None

    try:

        home_attack = float(
            home_stats["home_attack"]
        )

        home_defense = float(
            home_stats["home_defense"]
        )

        away_attack = float(
            away_stats["away_attack"]
        )

        away_defense = float(
            away_stats["away_defense"]
        )

    except (KeyError, TypeError, ValueError):

        return None, None

    # --------------------------------------------------------
    # Alapmodell
    # --------------------------------------------------------

    expected_home = (
        home_attack * 0.60
        +
        away_defense * 0.40
    )

    expected_away = (
        away_attack * 0.60
        +
        home_defense * 0.40
    )

    # --------------------------------------------------------
    # Hazai pálya
    # --------------------------------------------------------

    expected_home *= 1.03
    expected_away *= 0.97

    return (
        expected_home,
        expected_away
    )


# ============================================================
# POISSON 1X2
# ============================================================

def calculate_1x2(
    expected_home,
    expected_away
):
    """
    Poisson-alapú 1X2 modell.
    """

    if (
        expected_home is None
        or
        expected_away is None
    ):
        return None

    max_goals = 70

    goals = np.arange(
        0,
        max_goals + 1
    )

    home_distribution = poisson.pmf(
        goals,
        expected_home
    )

    away_distribution = poisson.pmf(
        goals,
        expected_away
    )

    matrix = np.outer(
        home_distribution,
        away_distribution
    )

    home_win = np.tril(
        matrix,
        -1
    ).sum()

    draw = np.trace(
        matrix
    )

    away_win = np.triu(
        matrix,
        1
    ).sum()

    total = (
        home_win
        +
        draw
        +
        away_win
    )

    if total <= 0:
        return None

    return {
        "home": home_win / total,
        "draw": draw / total,
        "away": away_win / total,
    }


# ============================================================
# OVER / UNDER
# ============================================================

def calculate_over_under(
    expected_home,
    expected_away
):
    """
    Összgól Over/Under számítás.
    """

    total_expected = (
        expected_home
        +
        expected_away
    )

    lines = [
        45.5,
        46.5,
        47.5,
        48.5,
        49.5,
        50.5,
        51.5,
        52.5,
        53.5,
        54.5,
    ]

    results = {}

    for line in lines:

        # Under 49.5 = maximum 49 gól
        under = poisson.cdf(
            int(line - 0.5),
            total_expected
        )

        over = 1 - under

        results[line] = {
            "over": over,
            "under": under
        }

    return results


# ============================================================
# LEGERŐSEBB TIPP
# ============================================================

def strongest_result(probabilities):

    if not probabilities:
        return None

    options = {
        "1": probabilities["home"],
        "X": probabilities["draw"],
        "2": probabilities["away"],
    }

    best = max(
        options,
        key=options.get
    )

    return (
        best,
        options[best]
    )


# ============================================================
# MÉRKŐZÉS ELEMZÉSE
# ============================================================

def analyze_match(
    match,
    stats
):

    home = match["home"]
    away = match["away"]

    print()
    print("=" * 70)

    print(
        f"{home} - {away}"
    )

    print(
        f"Verseny: {match['league']}"
    )

    print(
        f"Kategória: {match['category']}"
    )

    print("=" * 70)

    expected_home, expected_away = (
        calculate_expected_goals(
            home,
            away,
            stats
        )
    )

    # --------------------------------------------------------
    # NINCS STATISZTIKA
    # --------------------------------------------------------

    if (
        expected_home is None
        or
        expected_away is None
    ):

        print()
        print(
            "⚠ NINCS ELEGENDŐ CSAPATSTATISZTIKA"
        )

        print()
        print(
            "A program ezért NEM ad ki mesterséges tippet."
        )

        print()
        print(
            "Hiányzó csapatok:"
        )

        if home not in stats:
            print(
                f"  - {home}"
            )

        if away not in stats:
            print(
                f"  - {away}"
            )

        return

    # --------------------------------------------------------
    # VÁRHATÓ GÓLOK
    # --------------------------------------------------------

    print()
    print("VÁRHATÓ GÓLOK")
    print("-" * 40)

    print(
        f"{home}: {expected_home:.2f}"
    )

    print(
        f"{away}: {expected_away:.2f}"
    )

    print(
        f"Összesen: "
        f"{expected_home + expected_away:.2f}"
    )

    # --------------------------------------------------------
    # 1X2
    # --------------------------------------------------------

    probabilities = calculate_1x2(
        expected_home,
        expected_away
    )

    if not probabilities:
        return

    print()
    print("1X2 VALÓSZÍNŰSÉGEK")
    print("-" * 40)

    print(
        f"1 - {home}: "
        f"{percentage(probabilities['home'])}"
    )

    print(
        f"X - Döntetlen: "
        f"{percentage(probabilities['draw'])}"
    )

    print(
        f"2 - {away}: "
        f"{percentage(probabilities['away'])}"
    )

    # --------------------------------------------------------
    # LEGERŐSEBB KIMENET
    # --------------------------------------------------------

    strongest = strongest_result(
        probabilities
    )

    if strongest:

        result, probability = strongest

        print()
        print(
            f"★ LEGERŐSEBB KIMENET: "
            f"{result} "
            f"({percentage(probability)})"
        )

    # --------------------------------------------------------
    # OVER / UNDER
    # --------------------------------------------------------

    markets = calculate_over_under(
        expected_home,
        expected_away
    )

    print()
    print("OVER / UNDER")
    print("-" * 40)

    for line, values in markets.items():

        print(
            f"Over {line}: "
            f"{percentage(values['over'])}"
        )

        print(
            f"Under {line}: "
            f"{percentage(values['under'])}"
        )


# ============================================================
# FŐPROGRAM
# ============================================================

def main():

    now = datetime.now(
        ZoneInfo(TIMEZONE)
    )

    print()
    print("=" * 70)
    print("🤾 KEZI_AI - KÉZILABDA ELEMZŐ")
    print("=" * 70)

    print(
        f"Időpont: "
        f"{now.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(
        "Időzóna: Europe/Budapest"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # MECCSEK
    # --------------------------------------------------------

    print()
    print(
        "📡 Mai kézilabda-mérkőzések keresése..."
    )

    all_matches = (
        get_today_handball_matches()
    )

    if not all_matches:

        print()
        print(
            "❌ Nem sikerült mai kézilabda-"
            "mérkőzést lekérni."
        )

        print()
        print(
            "A program biztonsági okból nem talál ki"
            " mérkőzéseket."
        )

        sys.exit(0)

    print()
    print(
        f"Összes mai kézilabda-esemény: "
        f"{len(all_matches)}"
    )

    # --------------------------------------------------------
    # CÉLZOTT MECCSEK
    # --------------------------------------------------------

    target_matches = filter_matches(
        all_matches
    )

    print(
        f"Célzott EHF/U18/BL mérkőzések: "
        f"{len(target_matches)}"
    )

    # --------------------------------------------------------
    # HA NINCS TALÁLAT
    # --------------------------------------------------------

    if not target_matches:

        print()
        print(
            "ℹ Ma nem találtam a beállított "
            "EHF/U18/BL szűrésnek megfelelő mérkőzést."
        )

        print()
        print(
            "A teljes mai lista:"
        )

        for match in all_matches:

            print(
                f"- "
                f"{match['home']} - "
                f"{match['away']} "
                f"[{match['league']}]"
            )

        sys.exit(0)

    # --------------------------------------------------------
    # STATISZTIKÁK
    # --------------------------------------------------------

    stats = get_team_stats()

    # --------------------------------------------------------
    # ELEMZÉS
    # --------------------------------------------------------

    print()

    for match in target_matches:

        analyze_match(
            match,
            stats
        )

    print()
    print("=" * 70)
    print("✅ ELEMZÉS BEFEJEZVE")
    print("=" * 70)


# ============================================================
# INDÍTÁS
# ============================================================

if __name__ == "__main__":
    main()