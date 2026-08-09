import os
import math
import json
import random
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone

import requests


# ============================================================
# HANDBALL AI ANALYZER
# Version 1.0
#
# A program célja:
# - kézilabda mérkőzések elemzése
# - korábbi eredményekből csapaterősség becslése
# - támadó / védekező teljesítmény
# - forma
# - hazai pálya
# - Monte-Carlo szimuláció
# - 1X2 valószínűségek
# - gólpiac valószínűségek
# - bookmaker odds -> implied probability
# - value számítás
#
# FONTOS:
# Az oddsokat és meccseket API-ból kell betölteni.
# API nélkül a program demo adatokkal is működik.
# ============================================================


# ============================================================
# BEÁLLÍTÁSOK
# ============================================================

API_KEY = os.getenv("HANDBALL_API_KEY", "")

# Ha van saját API-d, ide kerülhet az endpoint.
# Később könnyen cserélhető.
API_URL = os.getenv(
    "HANDBALL_API_URL",
    ""
)

SIMULATIONS = 100_000

FORM_MATCHES = 10

# Súlyok
FORM_WEIGHT = 0.35
SEASON_WEIGHT = 0.65

HOME_ADVANTAGE = 1.025

MIN_VALUE = 0.03


# ============================================================
# HTTP
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 "
        "Handball-AI-Analyzer/1.0"
    ),
    "Accept": "application/json",
})


def api_get(url, params=None):

    try:

        response = SESSION.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        return response.json()

    except Exception as exc:

        print(f"[API ERROR] {exc}")

        return None


# ============================================================
# ADATSTRUKTÚRA
# ============================================================

class TeamStats:

    def __init__(self, name):

        self.name = name

        self.games = 0

        self.goals_for = []
        self.goals_against = []

        self.home_games = 0
        self.away_games = 0

        self.home_goals_for = []
        self.home_goals_against = []

        self.away_goals_for = []
        self.away_goals_against = []

        self.points = 0

        self.form = deque(maxlen=FORM_MATCHES)

    @property
    def avg_goals_for(self):

        if not self.goals_for:
            return 0

        return statistics.mean(self.goals_for)

    @property
    def avg_goals_against(self):

        if not self.goals_against:
            return 0

        return statistics.mean(self.goals_against)

    @property
    def avg_goal_difference(self):

        if not self.goals_for:
            return 0

        return statistics.mean(
            a - b
            for a, b in zip(
                self.goals_for,
                self.goals_against
            )
        )

    @property
    def form_score(self):

        if not self.form:
            return 0.5

        values = []

        for result in self.form:

            if result == "W":
                values.append(1.0)

            elif result == "D":
                values.append(0.5)

            else:
                values.append(0.0)

        return statistics.mean(values)


# ============================================================
# ADATBETÖLTÉS
# ============================================================

def load_matches_from_json(path="matches.json"):

    """
    A matches.json formátuma:

    [
        {
            "date": "2026-08-01",
            "home": "Veszprem",
            "away": "Kiel",
            "home_score": 34,
            "away_score": 31,
            "odds": {
                "home": 1.50,
                "draw": 9.00,
                "away": 3.80,
                "over_55.5": 1.85,
                "under_55.5": 1.85
            }
        }
    ]
    """

    if not os.path.exists(path):

        print(
            f"[INFO] {path} nem található."
        )

        return []

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as exc:

        print(
            f"[JSON ERROR] {exc}"
        )

        return []


# ============================================================
# CSAPATSTATISZTIKA
# ============================================================

def build_team_database(matches):

    teams = {}

    def get_team(name):

        if name not in teams:

            teams[name] = TeamStats(name)

        return teams[name]

    for match in matches:

        home = match.get("home")
        away = match.get("away")

        hs = match.get("home_score")
        aws = match.get("away_score")

        if (
            not home
            or not away
            or hs is None
            or aws is None
        ):
            continue

        home_team = get_team(home)
        away_team = get_team(away)

        # HOME

        home_team.games += 1

        home_team.home_games += 1

        home_team.goals_for.append(hs)
        home_team.goals_against.append(aws)

        home_team.home_goals_for.append(hs)
        home_team.home_goals_against.append(aws)

        # AWAY

        away_team.games += 1

        away_team.away_games += 1

        away_team.goals_for.append(aws)
        away_team.goals_against.append(hs)

        away_team.away_goals_for.append(aws)
        away_team.away_goals_against.append(hs)

        # FORM

        if hs > aws:

            home_team.form.append("W")
            away_team.form.append("L")

            home_team.points += 2

        elif hs == aws:

            home_team.form.append("D")
            away_team.form.append("D")

            home_team.points += 1
            away_team.points += 1

        else:

            home_team.form.append("L")
            away_team.form.append("W")

            away_team.points += 2

    return teams


# ============================================================
# LIGA ÁTLAG
# ============================================================

def calculate_league_averages(matches):

    goals = []

    home_goals = []
    away_goals = []

    for match in matches:

        hs = match.get("home_score")
        aws = match.get("away_score")

        if hs is None or aws is None:
            continue

        goals.append(hs + aws)

        home_goals.append(hs)
        away_goals.append(aws)

    if not goals:

        return {
            "total": 56.0,
            "home": 28.0,
            "away": 28.0
        }

    return {
        "total": statistics.mean(goals),
        "home": statistics.mean(home_goals),
        "away": statistics.mean(away_goals)
    }


# ============================================================
# CSAPATERŐSSÉG
# ============================================================

def calculate_strength(team, league):

    if team.games < 2:

        return {
            "attack": 1.0,
            "defense": 1.0
        }

    attack = (
        team.avg_goals_for
        / max(league["total"] / 2, 1)
    )

    defense = (
        team.avg_goals_against
        / max(league["total"] / 2, 1)
    )

    return {
        "attack": attack,
        "defense": defense
    }


# ============================================================
# FORMA KORREKCIÓ
# ============================================================

def form_multiplier(team):

    score = team.form_score

    # 0.5 = semleges
    # 1.0 = nagyon jó forma
    # 0.0 = nagyon rossz forma

    return 0.90 + score * 0.20


# ============================================================
# VÁRHATÓ GÓLOK
# ============================================================

def expected_goals(
    home_team,
    away_team,
    league
):

    home_strength = calculate_strength(
        home_team,
        league
    )

    away_strength = calculate_strength(
        away_team,
        league
    )

    home_attack = home_strength["attack"]
    home_defense = home_strength["defense"]

    away_attack = away_strength["attack"]
    away_defense = away_strength["defense"]

    league_home = league["home"]
    league_away = league["away"]

    home_form = form_multiplier(home_team)
    away_form = form_multiplier(away_team)

    home_expected = (
        league_home
        * home_attack
        * away_defense
        * HOME_ADVANTAGE
        * home_form
    )

    away_expected = (
        league_away
        * away_attack
        * home_defense
        * away_form
    )

    # Biztonsági korlát
    home_expected = max(
        10,
        min(home_expected, 50)
    )

    away_expected = max(
        10,
        min(away_expected, 50)
    )

    return (
        home_expected,
        away_expected
    )


# ============================================================
# POISSON
# ============================================================

def poisson_probability(k, lam):

    if lam <= 0:
        return 0

    return (
        math.exp(-lam)
        * lam ** k
        / math.factorial(k)
    )


# ============================================================
# SCORE MATRIX
# ============================================================

def score_matrix(
    home_lambda,
    away_lambda
):

    matrix = {}

    # Kézilabdánál 60 körüli összgól gyakori,
    # ezért 0-70 tartományt használunk.
    for home in range(0, 71):

        for away in range(0, 71):

            p_home = poisson_probability(
                home,
                home_lambda
            )

            p_away = poisson_probability(
                away,
                away_lambda
            )

            matrix[(home, away)] = (
                p_home * p_away
            )

    total = sum(matrix.values())

    if total > 0:

        matrix = {
            score: probability / total
            for score, probability
            in matrix.items()
        }

    return matrix


# ============================================================
# 1X2 VALÓSZÍNŰSÉG
# ============================================================

def calculate_1x2(matrix):

    home = 0
    draw = 0
    away = 0

    for (
        (home_score, away_score),
        probability
    ) in matrix.items():

        if home_score > away_score:

            home += probability

        elif home_score == away_score:

            draw += probability

        else:

            away += probability

    return {
        "home": home,
        "draw": draw,
        "away": away
    }


# ============================================================
# GÓLPIAC
# ============================================================

def total_goals_probability(
    matrix,
    line
):

    over = 0
    under = 0

    for (
        (home_score, away_score),
        probability
    ) in matrix.items():

        total = home_score + away_score

        if total > line:

            over += probability

        else:

            under += probability

    return {
        "over": over,
        "under": under
    }


# ============================================================
# MONTE CARLO
# ============================================================

def monte_carlo(
    home_lambda,
    away_lambda,
    simulations=SIMULATIONS
):

    home_wins = 0
    draws = 0
    away_wins = 0

    totals = []

    for _ in range(simulations):

        home_score = random.poisson(
            home_lambda
        ) if hasattr(random, "poisson") else None

        # Python random modulban nincs poisson,
        # ezért saját Poisson mintavétel.
        if home_score is None:

            home_score = poisson_sample(
                home_lambda
            )

        away_score = poisson_sample(
            away_lambda
        )

        totals.append(
            home_score + away_score
        )

        if home_score > away_score:

            home_wins += 1

        elif home_score == away_score:

            draws += 1

        else:

            away_wins += 1

    return {

        "home": home_wins / simulations,

        "draw": draws / simulations,

        "away": away_wins / simulations,

        "average_total":
            statistics.mean(totals)

    }


def poisson_sample(lam):

    """
    Knuth-féle Poisson mintavétel.
    """

    L = math.exp(-lam)

    k = 0

    p = 1.0

    while p > L:

        k += 1

        p *= random.random()

    return k - 1


# ============================================================
# ODDS
# ============================================================

def implied_probability(odds):

    if odds is None or odds <= 1:

        return 0

    return 1 / odds


def value_percentage(
    model_probability,
    odds
):

    if odds is None or odds <= 1:

        return 0

    fair_odds = 1 / model_probability

    value = (
        model_probability * odds
    ) - 1

    return value


# ============================================================
# BOOKMAKER MARGIN
# ============================================================

def remove_margin(odds):

    probabilities = {}

    for key, odd in odds.items():

        if odd and odd > 1:

            probabilities[key] = (
                1 / odd
            )

    total = sum(probabilities.values())

    if total <= 0:

        return {}

    return {
        key: value / total
        for key, value
        in probabilities.items()
    }


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_score(
    home_team,
    away_team,
    model_probability,
    market_probability
):

    score = 50

    # Modell és piac különbsége
    edge = (
        model_probability
        - market_probability
    )

    score += edge * 100

    # Minta nagysága
    sample = (
        home_team.games
        + away_team.games
    )

    if sample >= 30:
        score += 10

    elif sample >= 20:
        score += 7

    elif sample >= 10:
        score += 4

    # Forma
    form_difference = abs(
        home_team.form_score
        - away_team.form_score
    )

    score += form_difference * 10

    return max(
        0,
        min(
            100,
            round(score)
        )
    )


# ============================================================
# EGY MECCS ELEMZÉSE
# ============================================================

def analyze_match(
    match,
    teams,
    league
):

    home_name = match["home"]
    away_name = match["away"]

    if home_name not in teams:

        return None

    if away_name not in teams:

        return None

    home_team = teams[home_name]
    away_team = teams[away_name]

    home_lambda, away_lambda = expected_goals(
        home_team,
        away_team,
        league
    )

    matrix = score_matrix(
        home_lambda,
        away_lambda
    )

    probabilities = calculate_1x2(
        matrix
    )

    mc = monte_carlo(
        home_lambda,
        away_lambda
    )

    # A matematikai modell + Monte Carlo átlaga
    final_probabilities = {

        "home": (
            probabilities["home"]
            + mc["home"]
        ) / 2,

        "draw": (
            probabilities["draw"]
            + mc["draw"]
        ) / 2,

        "away": (
            probabilities["away"]
            + mc["away"]
        ) / 2

    }

    result = {

        "match":
            f"{home_name} - {away_name}",

        "expected_home_goals":
            round(home_lambda, 2),

        "expected_away_goals":
            round(away_lambda, 2),

        "expected_total_goals":
            round(
                home_lambda + away_lambda,
                2
            ),

        "probabilities": {

            "home":
                round(
                    final_probabilities["home"]
                    * 100,
                    2
                ),

            "draw":
                round(
                    final_probabilities["draw"]
                    * 100,
                    2
                ),

            "away":
                round(
                    final_probabilities["away"]
                    * 100,
                    2
                )

        }

    }

    # ========================================================
    # ODDS
    # ========================================================

    odds = match.get("odds", {})

    if odds:

        market = remove_margin(odds)

        value = {}

        for selection in [
            "home",
            "draw",
            "away"
        ]:

            if selection in odds:

                model_probability = (
                    final_probabilities[
                        selection
                    ]
                )

                market_probability = (
                    market.get(
                        selection,
                        implied_probability(
                            odds[selection]
                        )
                    )
                )

                value[selection] = {

                    "odds":
                        odds[selection],

                    "model_probability":
                        round(
                            model_probability
                            * 100,
                            2
                        ),

                    "market_probability":
                        round(
                            market_probability
                            * 100,
                            2
                        ),

                    "value":
                        round(
                            value_percentage(
                                model_probability,
                                odds[selection]
                            ) * 100,
                            2
                        ),

                    "confidence":
                        confidence_score(
                            home_team,
                            away_team,
                            model_probability,
                            market_probability
                        )

                }

        result["value"] = value

        # Gólpiacok
        for key, odd in odds.items():

            if key.startswith(
                "over_"
            ):

                try:

                    line = float(
                        key.split("_")[1]
                    )

                except Exception:

                    continue

                probabilities_goals = (
                    total_goals_probability(
                        matrix,
                        line
                    )
                )

                result.setdefault(
                    "goal_markets",
                    {}
                )

                over_probability = (
                    probabilities_goals["over"]
                )

                result[
                    "goal_markets"
                ][key] = {

                    "odds": odd,

                    "model_probability":
                        round(
                            over_probability
                            * 100,
                            2
                        ),

                    "value":
                        round(
                            value_percentage(
                                over_probability,
                                odd
                            ) * 100,
                            2
                        )

                }

    return result


# ============================================================
# TOP VALUE KERESÉSE
# ============================================================

def find_best_values(results):

    opportunities = []

    for result in results:

        for selection, data in result.get(
            "value",
            {}
        ).items():

            if data["value"] >= MIN_VALUE * 100:

                opportunities.append({

                    "match":
                        result["match"],

                    "selection":
                        selection,

                    "odds":
                        data["odds"],

                    "model_probability":
                        data[
                            "model_probability"
                        ],

                    "market_probability":
                        data[
                            "market_probability"
                        ],

                    "value":
                        data["value"],

                    "confidence":
                        data["confidence"]

                })

    opportunities.sort(
        key=lambda x: (
            x["value"],
            x["confidence"]
        ),
        reverse=True
    )

    return opportunities


# ============================================================
# KIÍRÁS
# ============================================================

def print_analysis(result):

    print()
    print("=" * 70)

    print(
        f"🏐 {result['match']}"
    )

    print("=" * 70)

    print(
        f"Várható gólok: "
        f"{result['expected_home_goals']} - "
        f"{result['expected_away_goals']}"
    )

    print(
        f"Várható összgól: "
        f"{result['expected_total_goals']}"
    )

    p = result["probabilities"]

    print()

    print(
        f"🏠 Hazai: {p['home']}%"
    )

    print(
        f"🤝 Döntetlen: {p['draw']}%"
    )

    print(
        f"✈️ Vendég: {p['away']}%"
    )

    if "value" in result:

        print()
        print("--- VALUE ---")

        for selection, data in result[
            "value"
        ].items():

            print(
                f"{selection.upper():8} "
                f"odds={data['odds']:.2f} "
                f"model={data['model_probability']:.1f}% "
                f"value={data['value']:+.2f}% "
                f"confidence={data['confidence']}/100"
            )

    if "goal_markets" in result:

        print()
        print("--- GÓLPIAC ---")

        for market, data in result[
            "goal_markets"
        ].items():

            print(
                f"{market:12} "
                f"odds={data['odds']:.2f} "
                f"model={data['model_probability']:.1f}% "
                f"value={data['value']:+.2f}%"
            )


# ============================================================
# DEMO ADATOK
# ============================================================

def create_demo_matches():

    return [

        {
            "date": "2026-07-20",
            "home": "Veszprem",
            "away": "Szeged",
            "home_score": 34,
            "away_score": 30
        },

        {
            "date": "2026-07-25",
            "home": "Szeged",
            "away": "Veszprem",
            "home_score": 31,
            "away_score": 33
        },

        {
            "date": "2026-07-28",
            "home": "Veszprem",
            "away": "Kiel",
            "home_score": 35,
            "away_score": 31
        },

        {
            "date": "2026-07-30",
            "home": "Kiel",
            "away": "Szeged",
            "home_score": 29,
            "away_score": 30
        },

        {
            "date": "2026-08-01",
            "home": "Veszprem",
            "away": "Kiel",
            "home_score": 32,
            "away_score": 29
        },

        {
            "date": "2026-08-03",
            "home": "Szeged",
            "away": "Kiel",
            "home_score": 30,
            "away_score": 30
        },

        # Közelgő mérkőzés
        {
            "date": "2026-08-10",

            "home": "Veszprem",

            "away": "Szeged",

            "odds": {

                "home": 1.55,

                "draw": 9.00,

                "away": 3.80,

                "over_55.5": 1.85,

                "under_55.5": 1.85

            }

        }

    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("        HANDBALL AI BETTING ANALYZER")
    print("=" * 70)
    print()

    matches = load_matches_from_json()

    # Ha nincs saját adatfájl:
    if not matches:

        print(
            "[INFO] Demo adatbázis használata."
        )

        matches = create_demo_matches()

    historical = [
        match
        for match in matches
        if match.get("home_score") is not None
        and match.get("away_score") is not None
    ]

    upcoming = [
        match
        for match in matches
        if match.get("home_score") is None
        or match.get("away_score") is None
    ]

    if not historical:

        print(
            "Nincs elegendő történelmi adat."
        )

        return

    teams = build_team_database(
        historical
    )

    league = calculate_league_averages(
        historical
    )

    print(
        f"Liga átlagos összgól: "
        f"{league['total']:.2f}"
    )

    print(
        f"Átlag hazai gól: "
        f"{league['home']:.2f}"
    )

    print(
        f"Átlag vendég gól: "
        f"{league['away']:.2f}"
    )

    results = []

    for match in upcoming:

        result = analyze_match(
            match,
            teams,
            league
        )

        if result:

            results.append(result)

            print_analysis(
                result
            )

    if not results:

        print(
            "\nNincs elemezhető közelgő mérkőzés."
        )

        return

    # ========================================================
    # TOP VALUE
    # ========================================================

    opportunities = find_best_values(
        results
    )

    print()
    print("=" * 70)
    print("                 TOP VALUE")
    print("=" * 70)

    if not opportunities:

        print(
            "Nem találtam megfelelő value lehetőséget."
        )

    else:

        for index, opportunity in enumerate(
            opportunities[:10],
            1
        ):

            print(
                f"\n#{index} "
                f"{opportunity['match']}"
            )

            print(
                f"   Tipp: "
                f"{opportunity['selection'].upper()}"
            )

            print(
                f"   Odds: "
                f"{opportunity['odds']:.2f}"
            )

            print(
                f"   Modell: "
                f"{opportunity['model_probability']:.2f}%"
            )

            print(
                f"   Piac: "
                f"{opportunity['market_probability']:.2f}%"
            )

            print(
                f"   VALUE: "
                f"{opportunity['value']:+.2f}%"
            )

            print(
                f"   CONFIDENCE: "
                f"{opportunity['confidence']}/100"
            )

    print()
    print("=" * 70)
    print(
        "Elemzés befejezve."
    )
    print("=" * 70)


if __name__ == "__main__":

    main()