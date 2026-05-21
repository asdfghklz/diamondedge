"""
DiamondEdge MLB Engine v3
==========================
Fully automated ensemble model.

Data sources (all free, no manual downloads):
  - FanGraphs      : ATC, THE BAT X, Steamer projections (batting + pitching)
  - Baseball Savant: Stuff+ / pitch metrics for starting pitchers
  - MLB Stats API  : Schedule, lineups, bullpen usage, umpires (no key needed)
  - The Odds API   : Live moneylines (free key)
  - OpenWeatherMap : Live weather per ballpark (free key)
  - model_lookup   : Park factors, team home/away WR, umpire bias (33k game trained)

Runs 3x per day via GitHub Actions:
  7:00 AM CT  - morning run (probable pitchers, opening odds)
  11:30 AM CT - lineup run (confirmed lineups injected)
  5:30 PM CT  - evening run (final lineups, updated odds, weather)
"""

import os, sys, json, math, time, datetime, requests, csv, io
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
ODDS_KEY    = os.environ.get("ODDS_API_KEY", "")
WEATHER_KEY = os.environ.get("OPENWEATHER_KEY", "")
OUT_PATH    = Path(__file__).parent.parent / "picks.json"
LOOKUP_PATH = Path(__file__).parent / "model_lookup.json"

MLB_API     = "https://statsapi.mlb.com/api/v1"
ODDS_API    = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
WEATHER_API = "https://api.openweathermap.org/data/2.5/weather"

# Ensemble weights
W_ATC     = 0.40   # ATC batting/pitching
W_BATX    = 0.30   # THE BAT X batting
W_STEAMER = 0.20   # Steamer pitching
W_STUFF   = 0.10   # Stuff+ / pitch modeling

# Situational modifier caps
FATIGUE_PER_RP  = 0.030   # per fatigued reliever
FATIGUE_MAX     = 0.120
EV_MIN          = 0.040   # minimum edge to flag

# Ballpark coordinates for weather lookup
PARK_COORDS = {
    "ARI": (33.4453, -112.0667), "ATL": (33.8908, -84.4681),
    "BAL": (39.2838, -76.6216),  "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),  "CWS": (41.8300, -87.6339),
    "CIN": (39.0979, -84.5082),  "CLE": (41.4962, -81.6852),
    "COL": (39.7559, -104.9942), "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),  "KC":  (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827), "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197),  "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2778),  "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),  "ATH": (37.7516, -122.2005),
    "PHI": (39.9061, -75.1665),  "PIT": (40.4468, -80.0057),
    "SD":  (32.7076, -117.1570), "SF":  (37.7786, -122.3893),
    "SEA": (47.5914, -122.3323), "STL": (38.6226, -90.1928),
    "TB":  (27.7683, -82.6534),  "TEX": (32.7473, -97.0845),
    "TOR": (43.6414, -79.3894),  "WSH": (38.8730, -77.0074),
}

# Park IDs for model lookup
TEAM_PARK = {
    "ARI":"PHO01","ATL":"ATL03","BAL":"BAL12","BOS":"BOS07","CHC":"CHC11",
    "CWS":"CHI12","CIN":"CIN09","CLE":"CLE08","COL":"DEN02","DET":"DET02",
    "HOU":"HOU03","KC":"KC01","LAA":"LAA01","LAD":"LAD01","MIA":"MIA02",
    "MIL":"MIL06","MIN":"MIN04","NYM":"NYC21","NYY":"NYC20","ATH":"OAK01",
    "PHI":"PHI13","PIT":"PIT01","SD":"SAN02","SF":"SFO03","SEA":"SEA03",
    "STL":"STL10","TB":"STP01","TEX":"ARL02","TOR":"TOR02","WSH":"WAS11",
}

# ── LOAD TRAINED LOOKUP ───────────────────────────────────────────────────────
print("Loading trained model lookup (33,292 games 2010-2024)...")
with open(LOOKUP_PATH) as f:
    LOOKUP = json.load(f)

PARK_FACTORS = {k: v["factor"] for k, v in LOOKUP["park_factors"].items()}
TEAM_HOME_WR = {k: v["wr"] for k, v in LOOKUP["team_home_wr"].items()}
TEAM_AWAY_WR = {k: v["wr"] for k, v in LOOKUP["team_away_wr"].items()}
UMP_BIAS     = {k: v["bias"] for k, v in LOOKUP["ump_bias"].items()}
LEAGUE_HOME  = LOOKUP["meta"]["league_home_wr"]
LEAGUE_RUNS  = LOOKUP["meta"]["league_avg_runs"]
print(f"  Loaded: {LOOKUP['meta']['train_games']:,} games | {LOOKUP['meta']['train_seasons']}")

# ── HTTP HELPERS ──────────────────────────────────────────────────────────────
def get(url, params={}, timeout=15, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": "DiamondEdge/3.0"})
            r.raise_for_status()
            return r
        except Exception as e:
            if i == retries - 1:
                print(f"  [WARN] GET failed: {url[:60]} → {e}")
                return None
            time.sleep(2 ** i)

def mlb(path, params={}):
    r = get(f"{MLB_API}{path}", params)
    return r.json() if r else {}

# ── FANGRAPHS PROJECTIONS ─────────────────────────────────────────────────────
FG_BATTING_URLS = {
    "atc":     "https://www.fangraphs.com/projections.aspx?pos=all&stats=bat&type=atc&team=0&lg=all&players=0&csv=1",
    "batx":    "https://www.fangraphs.com/projections.aspx?pos=all&stats=bat&type=thebatx&team=0&lg=all&players=0&csv=1",
    "steamer": "https://www.fangraphs.com/projections.aspx?pos=all&stats=bat&type=steamer&team=0&lg=all&players=0&csv=1",
}
FG_PITCHING_URLS = {
    "atc":     "https://www.fangraphs.com/projections.aspx?pos=all&stats=pit&type=atc&team=0&lg=all&players=0&csv=1",
    "steamer": "https://www.fangraphs.com/projections.aspx?pos=all&stats=pit&type=steamer&team=0&lg=all&players=0&csv=1",
}

def fetch_fg_csv(url):
    """Download a FanGraphs projection CSV and return list of dicts."""
    r = get(url, timeout=20)
    if not r:
        return []
    try:
        reader = csv.DictReader(io.StringIO(r.text))
        return list(reader)
    except Exception as e:
        print(f"  [WARN] FG CSV parse failed: {e}")
        return []

def load_fg_projections():
    """
    Pull ATC, THE BAT X, and Steamer from FanGraphs.
    Returns batting_proj and pitching_proj dicts keyed by playerid.
    """
    print("\n[FanGraphs] Pulling batting projections...")
    bat_raw = {}
    for sys_name, url in FG_BATTING_URLS.items():
        rows = fetch_fg_csv(url)
        print(f"  {sys_name.upper()}: {len(rows)} batters")
        bat_raw[sys_name] = {r.get("playerid", r.get("PlayerId", "")): r for r in rows if r.get("playerid") or r.get("PlayerId")}

    print("[FanGraphs] Pulling pitching projections...")
    pit_raw = {}
    for sys_name, url in FG_PITCHING_URLS.items():
        rows = fetch_fg_csv(url)
        print(f"  {sys_name.upper()}: {len(rows)} pitchers")
        pit_raw[sys_name] = {r.get("playerid", r.get("PlayerId", "")): r for r in rows if r.get("playerid") or r.get("PlayerId")}

    # Build ensemble batting projection per player
    # Key metric: wOBA (best single predictor of run scoring)
    batting_proj = {}
    all_player_ids = set()
    for sys_rows in bat_raw.values():
        all_player_ids.update(sys_rows.keys())

    for pid in all_player_ids:
        if not pid: continue
        vals = {}
        for metric in ["wOBA", "woba", "WOBA"]:
            atc_row  = bat_raw.get("atc",  {}).get(pid, {})
            batx_row = bat_raw.get("batx", {}).get(pid, {})
            stm_row  = bat_raw.get("steamer", {}).get(pid, {})
            def sf(row, keys):
                for k in keys:
                    v = row.get(k)
                    try:
                        fv = float(v)
                        if 0.1 < fv < 0.6: return fv
                    except: pass
                return None

            atc_woba  = sf(atc_row,  ["wOBA","woba","WOBA"])
            batx_woba = sf(batx_row, ["wOBA","woba","WOBA"])
            stm_woba  = sf(stm_row,  ["wOBA","woba","WOBA"])

            # Build weighted ensemble — only use systems that have data
            weights_used = []
            woba_sum = 0
            if atc_woba:  woba_sum += atc_woba  * (W_ATC/(W_ATC+W_BATX+W_STEAMER)); weights_used.append("atc")
            if batx_woba: woba_sum += batx_woba * (W_BATX/(W_ATC+W_BATX+W_STEAMER)); weights_used.append("batx")
            if stm_woba:  woba_sum += stm_woba  * (W_STEAMER/(W_ATC+W_BATX+W_STEAMER)); weights_used.append("steamer")

            if woba_sum > 0:
                name = atc_row.get("Name") or batx_row.get("Name") or stm_row.get("Name") or pid
                team = atc_row.get("Team") or batx_row.get("Team") or stm_row.get("Team") or ""
                batting_proj[pid] = {
                    "name": name, "team": team,
                    "ensemble_woba": round(woba_sum, 4),
                    "atc_woba": atc_woba, "batx_woba": batx_woba, "steamer_woba": stm_woba,
                    "systems": weights_used,
                }
            break  # found at least one wOBA variant

    # Build ensemble pitching projection per pitcher
    pitching_proj = {}
    all_pit_ids = set()
    for sys_rows in pit_raw.values():
        all_pit_ids.update(sys_rows.keys())

    for pid in all_pit_ids:
        if not pid: continue
        atc_row = pit_raw.get("atc",  {}).get(pid, {})
        stm_row = pit_raw.get("steamer", {}).get(pid, {})

        def pf(row, keys, lo=0.5, hi=10.0):
            for k in keys:
                v = row.get(k)
                try:
                    fv = float(v)
                    if lo < fv < hi: return fv
                except: pass
            return None

        atc_era  = pf(atc_row, ["ERA","era"])
        stm_era  = pf(stm_row, ["ERA","era"])
        atc_fip  = pf(atc_row, ["FIP","fip"])
        stm_fip  = pf(stm_row, ["FIP","fip"])
        atc_k9   = pf(atc_row, ["K/9","K9","SO9"], 1, 16)
        stm_k9   = pf(stm_row, ["K/9","K9","SO9"], 1, 16)
        atc_bb9  = pf(atc_row, ["BB/9","BB9"], 0.5, 8)
        stm_bb9  = pf(stm_row, ["BB/9","BB9"], 0.5, 8)

        # Weighted ensemble ERA (ATC 60%, Steamer 40% for pitching)
        eras = [(atc_era, 0.6), (stm_era, 0.4)]
        era_vals = [(v, w) for v, w in eras if v]
        if era_vals:
            total_w = sum(w for _, w in era_vals)
            ens_era = sum(v * w for v, w in era_vals) / total_w
            fips = [(atc_fip, 0.6), (stm_fip, 0.4)]
            fip_vals = [(v, w) for v, w in fips if v]
            ens_fip = sum(v*w for v,w in fip_vals)/sum(w for _,w in fip_vals) if fip_vals else ens_era
            k9s = [(atc_k9, 0.6), (stm_k9, 0.4)]
            k9_vals = [(v,w) for v,w in k9s if v]
            ens_k9 = sum(v*w for v,w in k9_vals)/sum(w for _,w in k9_vals) if k9_vals else 8.0
            bb9s = [(atc_bb9,0.6),(stm_bb9,0.4)]
            bb9_vals = [(v,w) for v,w in bb9s if v]
            ens_bb9 = sum(v*w for v,w in bb9_vals)/sum(w for _,w in bb9_vals) if bb9_vals else 3.2

            name = atc_row.get("Name") or stm_row.get("Name") or pid
            team = atc_row.get("Team") or stm_row.get("Team") or ""
            pitching_proj[pid] = {
                "name": name, "team": team,
                "ensemble_era": round(ens_era, 2),
                "ensemble_fip": round(ens_fip, 2),
                "ensemble_k9":  round(ens_k9, 2),
                "ensemble_bb9": round(ens_bb9, 2),
                "atc_era": atc_era, "steamer_era": stm_era,
            }

    print(f"  Ensemble batting: {len(batting_proj)} players")
    print(f"  Ensemble pitching: {len(pitching_proj)} pitchers")
    return batting_proj, pitching_proj

# ── BASEBALL SAVANT STUFF+ ────────────────────────────────────────────────────
def fetch_stuff_plus(pitcher_id):
    """Pull Stuff+ metrics from Baseball Savant for a pitcher."""
    if not pitcher_id:
        return {"stuff_plus": 100, "provisional": True}
    try:
        url = f"https://baseballsavant.mlb.com/player-services/summary?player_id={pitcher_id}&position=P&type=batter&year=2026"
        r = get(url, timeout=10)
        if not r:
            return {"stuff_plus": 100, "provisional": True}
        data = r.json()
        # Extract Stuff+ from Savant payload
        stats = data.get("player_stats", {})
        sp = stats.get("stuff_plus") or stats.get("stuffPlus")
        if sp:
            return {"stuff_plus": float(sp), "provisional": False}
    except:
        pass
    return {"stuff_plus": 100, "provisional": True}

# ── MLB STATS API ─────────────────────────────────────────────────────────────
def get_schedule(date_str):
    data = mlb("/schedule", {
        "sportId": 1, "date": date_str,
        "hydrate": "probablePitcher,team,weather,officials,venue,lineups,linescore"
    })
    games = []
    for db in data.get("dates", []):
        for g in db.get("games", []):
            if g.get("status", {}).get("abstractGameCode") not in ("F", "DR"):
                games.append(g)
    return games

def get_confirmed_lineup(game):
    """Extract confirmed batting lineup from game data."""
    lineups = game.get("lineups", {})
    home_lineup = [p.get("id") for p in lineups.get("homePlayers", [])]
    away_lineup = [p.get("id") for p in lineups.get("awayPlayers", [])]
    return home_lineup, away_lineup

def get_bullpen_fatigue(team_id, date_str):
    """Count relievers who pitched in last 48 hours."""
    game_date = datetime.date.fromisoformat(date_str)
    fatigued = 0
    for days_back in [1, 2]:
        check = (game_date - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
        data = mlb("/schedule", {
            "sportId": 1, "teamId": team_id, "date": check,
            "hydrate": "linescore,boxscore"
        })
        for db in data.get("dates", []):
            for g in db.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                # Check if high-leverage relievers pitched
                box = g.get("boxscore", {})
                for side in ["home", "away"]:
                    team_data = box.get("teams", {}).get(side, {})
                    if team_data.get("team", {}).get("id") == team_id:
                        pitchers = team_data.get("pitchers", [])
                        # Relievers = pitchers after index 0 (starter)
                        if len(pitchers) > 1:
                            fatigued += min(len(pitchers) - 1, 2)
    return {
        "fatigued_count": fatigued,
        "tax": min(fatigued * FATIGUE_PER_RP, FATIGUE_MAX)
    }

def get_umpire(game):
    for official in game.get("officials", []):
        if official.get("officialType") == "Home Plate":
            uid = str(official.get("official", {}).get("id", ""))
            name = official.get("official", {}).get("fullName", "")
            return uid, name
    return None, None

# ── LIVE ODDS ─────────────────────────────────────────────────────────────────
def get_odds():
    if not ODDS_KEY:
        print("  [INFO] No ODDS_API_KEY — skipping odds")
        return {}
    r = get(ODDS_API, {
        "apiKey": ODDS_KEY, "regions": "us",
        "markets": "h2h", "oddsFormat": "american"
    })
    if not r:
        return {}
    result = {}
    for game in r.json():
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        for bk in game.get("bookmakers", []):
            if bk["key"] in ("draftkings", "fanduel", "betmgm"):
                for mkt in bk.get("markets", []):
                    if mkt["key"] == "h2h":
                        oc = {o["name"]: o["price"] for o in mkt["outcomes"]}
                        result[f"{home}|{away}"] = {
                            "home_ml": oc.get(home),
                            "away_ml": oc.get(away),
                            "book": bk.get("title", bk["key"])
                        }
                break
    print(f"  Odds: {len(result)} games")
    return result

# ── WEATHER ───────────────────────────────────────────────────────────────────
def get_weather(team_abbr):
    coords = PARK_COORDS.get(team_abbr)
    if not coords or not WEATHER_KEY:
        return {"temp_f": 72, "wind_mph": 0, "wind_dir": "", "condition": "unknown"}
    r = get(WEATHER_API, {
        "lat": coords[0], "lon": coords[1],
        "appid": WEATHER_KEY, "units": "imperial"
    })
    if not r:
        return {"temp_f": 72, "wind_mph": 0, "wind_dir": "", "condition": "unknown"}
    data = r.json()
    temp   = data.get("main", {}).get("temp", 72)
    wind   = data.get("wind", {})
    wspeed = wind.get("speed", 0)
    wdeg   = wind.get("deg", 0)
    # Convert wind degrees to direction
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    wdir = dirs[round(wdeg / 45) % 8]
    cond = data.get("weather", [{}])[0].get("main", "Clear")
    return {
        "temp_f":    round(temp, 1),
        "wind_mph":  round(wspeed, 1),
        "wind_dir":  wdir,
        "condition": cond,
    }

# ── WOBA → RUNS CONVERSION ────────────────────────────────────────────────────
def woba_to_runs_per_game(team_woba, park_factor=1.0, is_2026=True):
    """
    Convert team wOBA to projected runs per game.
    Uses linear weights approximation:
    League avg wOBA ~.318, avg runs/game ~4.41 (2024-25 avg)
    """
    league_woba = 0.318
    league_rpg  = 4.41
    # Run value sensitivity: each .010 wOBA ≈ 0.18 runs/game
    rpg = league_rpg + (team_woba - league_woba) * 18.0
    rpg = rpg * park_factor
    # 2023+ shift ban slightly boosted offense
    if is_2026:
        rpg *= 1.02
    return max(2.0, min(10.0, round(rpg, 2)))

def runs_to_win_prob(home_runs, away_runs):
    """
    Pythagorean win probability from projected runs.
    Uses exponent of 1.83 (baseball standard).
    """
    exp = 1.83
    if home_runs <= 0 or away_runs <= 0:
        return 0.5
    p = (home_runs ** exp) / (home_runs ** exp + away_runs ** exp)
    return round(p, 4)

# ── WEATHER ADJUSTMENTS ───────────────────────────────────────────────────────
def apply_weather(home_rpg, away_rpg, weather, park_id):
    """Adjust run projections for weather conditions."""
    temp   = weather.get("temp_f", 72)
    wind   = weather.get("wind_mph", 0)
    wdir   = weather.get("wind_dir", "")
    cond   = weather.get("condition", "")

    # Temperature effect: every 10°F above 72 adds ~2% to runs
    temp_mult = 1.0 + (temp - 72) * 0.002

    # Wind effect on total runs (not team-specific)
    wind_run_adj = 0.0
    if wind > 8:
        if wdir in ("N", "NE", "NW") and park_id in ("CHC11",):
            # Wrigley wind out to center/right
            wind_run_adj = wind * 0.04
        elif wdir in ("S", "SE", "SW") and park_id in ("CHC11",):
            wind_run_adj = -wind * 0.03
        elif wdir in ("E", "NE") and park_id in ("SFO03",):
            wind_run_adj = -wind * 0.035  # wind in from bay
        else:
            # Generic: out = +, in = -
            if wdir in ("N","NE","NW","E"):
                wind_run_adj = wind * 0.025
            else:
                wind_run_adj = -wind * 0.02

    # Rain/dome effect
    rain_mult = 0.97 if cond in ("Rain", "Drizzle", "Thunderstorm") else 1.0

    home_adj = home_rpg * temp_mult * rain_mult + wind_run_adj * 0.5
    away_adj = away_rpg * temp_mult * rain_mult + wind_run_adj * 0.5
    return round(home_adj, 2), round(away_adj, 2)

# ── MAIN MODEL ────────────────────────────────────────────────────────────────
def run_model(game, batting_proj, pitching_proj, odds_map, date_str):
    home_d  = game.get("teams", {}).get("home", {})
    away_d  = game.get("teams", {}).get("away", {})
    home_abbr = home_d.get("team", {}).get("abbreviation", "HM")
    away_abbr = away_d.get("team", {}).get("abbreviation", "AW")
    home_name = home_d.get("team", {}).get("name", "Home")
    away_name = away_d.get("team", {}).get("name", "Away")
    home_id   = home_d.get("team", {}).get("id")
    away_id   = away_d.get("team", {}).get("id")
    game_pk   = str(game.get("gamePk", ""))
    game_time = game.get("gameDate", "")
    venue     = game.get("venue", {}).get("name", "")
    game_id   = f"{date_str}-{away_abbr}-{home_abbr}"

    sp_home   = home_d.get("probablePitcher", {})
    sp_away   = away_d.get("probablePitcher", {})
    sp_home_id   = str(sp_home.get("id", ""))
    sp_away_id   = str(sp_away.get("id", ""))
    sp_home_name = sp_home.get("fullName", "TBD")
    sp_away_name = sp_away.get("fullName", "TBD")

    print(f"\n  {away_abbr} @ {home_abbr} | {sp_away_name} vs {sp_home_name}")

    # ── Confirmed lineups ──────────────────────────────────────────────────────
    home_lineup_ids, away_lineup_ids = get_confirmed_lineup(game)
    lineup_confirmed = len(home_lineup_ids) >= 8 and len(away_lineup_ids) >= 8

    # ── Build team wOBA from confirmed lineup or team average ──────────────────
    def team_woba_from_lineup(lineup_ids, team_abbr):
        """Average ensemble wOBA across confirmed lineup."""
        wobas = []
        for pid in lineup_ids:
            proj = batting_proj.get(str(pid))
            if proj and proj.get("ensemble_woba"):
                wobas.append(proj["ensemble_woba"])
        if len(wobas) >= 6:
            return round(sum(wobas) / len(wobas), 4), True
        # Fall back to team average from all proj batters
        team_wobas = [p["ensemble_woba"] for p in batting_proj.values()
                      if p.get("team","").upper() == team_abbr.upper()
                      and p.get("ensemble_woba")]
        if team_wobas:
            return round(sum(team_wobas[:9]) / min(len(team_wobas), 9), 4), False
        return 0.318, False  # league average fallback

    home_woba, h_lineup_used = team_woba_from_lineup(home_lineup_ids, home_abbr)
    away_woba, a_lineup_used = team_woba_from_lineup(away_lineup_ids, away_abbr)

    # ── SP projections (ATC + Steamer ensemble + Stuff+) ──────────────────────
    def sp_proj(sp_id, sp_name):
        # Try FanGraphs pitching proj first
        fg = pitching_proj.get(sp_id, {})
        if not fg:
            # Search by name
            for pid, p in pitching_proj.items():
                if p.get("name","").lower() == sp_name.lower():
                    fg = p; break
        # Stuff+
        stuff = fetch_stuff_plus(sp_id if sp_id else None)
        sp100 = stuff.get("stuff_plus", 100)

        era  = fg.get("ensemble_era", 4.50)
        fip  = fg.get("ensemble_fip", era)
        k9   = fg.get("ensemble_k9", 8.0)
        bb9  = fg.get("ensemble_bb9", 3.2)

        # Stuff+ adjustment: 110 Stuff+ ≈ elite, 90 ≈ below avg
        # Each 10 points of Stuff+ above/below 100 adjusts ERA by ~0.25
        stuff_era_adj = (100 - sp100) * 0.025
        adj_era = max(2.0, min(7.5, era + stuff_era_adj))

        return {
            "era": round(adj_era, 2), "fip": round(fip, 2),
            "k9": round(k9, 2), "bb9": round(bb9, 2),
            "stuff_plus": sp100,
            "fg_era": era, "provisional": not bool(fg),
        }

    hsp = sp_proj(sp_home_id, sp_home_name)
    asp = sp_proj(sp_away_id, sp_away_name)

    # ── SP quality adjustment to opponent wOBA ─────────────────────────────────
    # Better SP = suppress opponent wOBA further
    # ERA 3.00 pitcher holds opponents to ~.280 wOBA; ERA 5.00 → ~.340
    def sp_woba_suppression(era):
        return .318 - (4.50 - era) * 0.010

    home_opp_woba = sp_woba_suppression(hsp["era"])  # what home SP allows
    away_opp_woba = sp_woba_suppression(asp["era"])  # what away SP allows

    # Blend lineup wOBA with opponent's SP suppression
    # 60% lineup projection, 40% SP suppression
    home_eff_allowed = 0.60 * away_woba + 0.40 * home_opp_woba  # runs home team allows
    away_eff_allowed = 0.60 * home_woba + 0.40 * away_opp_woba  # runs away team allows

    # ── Park factor ────────────────────────────────────────────────────────────
    park_id  = TEAM_PARK.get(home_abbr, "default")
    park_fac = PARK_FACTORS.get(park_id, 1.0)

    # ── Weather ────────────────────────────────────────────────────────────────
    weather = get_weather(home_abbr)

    # ── Convert wOBA → runs ────────────────────────────────────────────────────
    home_proj_runs = woba_to_runs_per_game(away_eff_allowed, park_fac)  # runs scored BY home
    away_proj_runs = woba_to_runs_per_game(home_eff_allowed, park_fac)  # runs scored BY away
    home_proj_runs, away_proj_runs = apply_weather(home_proj_runs, away_proj_runs, weather, park_id)
    proj_total = round(home_proj_runs + away_proj_runs, 1)

    # ── F5 projection (SPs matter more, bullpen not yet in play) ──────────────
    f5_home = round(home_proj_runs * 0.48, 2)
    f5_away = round(away_proj_runs * 0.48, 2)

    # ── Base win probability from Pythagorean ──────────────────────────────────
    base_home_p = runs_to_win_prob(home_proj_runs, away_proj_runs)

    # ── Proprietary adjustments (from 33k game trained lookup) ────────────────
    h_home_wr = TEAM_HOME_WR.get(home_abbr, LEAGUE_HOME)
    a_away_wr = TEAM_AWAY_WR.get(away_abbr, 1 - LEAGUE_HOME)
    team_delta = (h_home_wr - LEAGUE_HOME) - (a_away_wr - (1 - LEAGUE_HOME))
    prop_adj = team_delta * 0.40  # proprietary component gets 40% weight

    # ── Umpire bias ────────────────────────────────────────────────────────────
    ump_id, ump_name = get_umpire(game)
    ump_b = UMP_BIAS.get(ump_id, 0.0) if ump_id else 0.0

    # ── Bullpen fatigue ────────────────────────────────────────────────────────
    h_bull = get_bullpen_fatigue(home_id, date_str) if home_id else {"tax": 0}
    a_bull = get_bullpen_fatigue(away_id, date_str) if away_id else {"tax": 0}

    # ── Ensemble final win probability ────────────────────────────────────────
    # Pythagorean provides the run-based foundation
    # Proprietary (team historical splits) adjusts
    # Umpire bias adjusts
    # Fatigue tax adjusts
    fair_home_p = base_home_p + prop_adj + ump_b - h_bull["tax"] + a_bull["tax"]
    fair_home_p = max(0.18, min(0.88, fair_home_p))
    fair_away_p = 1 - fair_home_p

    # F5 win prob (SP matters much more, no bullpen)
    f5_base = runs_to_win_prob(f5_home, f5_away)
    fair_f5_p = max(0.18, min(0.88, f5_base + prop_adj * 0.5 + ump_b * 0.3))

    # ── Fair odds ──────────────────────────────────────────────────────────────
    def to_ml(p):
        p = max(0.01, min(0.99, p))
        return round(-(p/(1-p))*100) if p >= 0.5 else round(((1-p)/p)*100)

    fair_home_ml = to_ml(fair_home_p)
    fair_away_ml = to_ml(fair_away_p)

    # ── Market comparison ──────────────────────────────────────────────────────
    def impl(ml):
        if not ml: return 0.5
        return abs(ml)/(abs(ml)+100) if ml < 0 else 100/(ml+100)

    key = f"{home_name}|{away_name}"
    mkt = odds_map.get(key, {})
    mkt_home_ml = mkt.get("home_ml")
    mkt_away_ml = mkt.get("away_ml")
    book = mkt.get("book", "N/A")

    if mkt_home_ml and mkt_away_ml:
        rh = impl(mkt_home_ml); ra = impl(mkt_away_ml); tt = rh + ra
        mkt_home_p = rh / tt; mkt_away_p = ra / tt
    else:
        mkt_home_p = mkt_away_p = None

    # ── EV & pick ──────────────────────────────────────────────────────────────
    def kelly(ev, wp, ml):
        if not ml or ev <= 0: return "0%"
        b = 100/abs(ml) if ml < 0 else ml/100
        k = max(0, min((b*wp-(1-wp))/b, 0.25))
        return f"{round(k*25, 1)}%"

    best_bet = None; ev_pct = 0.0; kstake = "0%"; bet_side = None; f5_bet = None
    if mkt_home_p:
        home_ev = fair_home_p - mkt_home_p
        away_ev = fair_away_p - mkt_away_p
        f5_home_ev = fair_f5_p - mkt_home_p
        f5_away_ev = (1-fair_f5_p) - mkt_away_p

        if home_ev >= away_ev and home_ev >= EV_MIN:
            best_bet = f"{home_abbr} ML {mkt_home_ml:+d}"
            ev_pct = round(home_ev*100, 1); bet_side = "home"
            kstake = kelly(home_ev, fair_home_p, mkt_home_ml)
        elif away_ev >= EV_MIN:
            best_bet = f"{away_abbr} ML {mkt_away_ml:+d}"
            ev_pct = round(away_ev*100, 1); bet_side = "away"
            kstake = kelly(away_ev, fair_away_p, mkt_away_ml)

        # F5 flag — is edge larger in first 5?
        if bet_side == "home" and f5_home_ev >= EV_MIN and f5_home_ev > home_ev + 0.02:
            f5_bet = f"F5: {home_abbr} — edge stronger before bullpen"
        elif bet_side == "away" and f5_away_ev >= EV_MIN and f5_away_ev > away_ev + 0.02:
            f5_bet = f"F5: {away_abbr} — edge stronger before bullpen"

    # ── Signals ────────────────────────────────────────────────────────────────
    signals = []
    if not lineup_confirmed:          signals.append("LINEUP_UNCONFIRMED")
    if h_lineup_used:                 signals.append(f"LINEUP_CONFIRMED_{home_abbr}")
    if a_lineup_used:                 signals.append(f"LINEUP_CONFIRMED_{away_abbr}")
    if h_bull["tax"] > 0.06:         signals.append(f"BULLPEN_FATIGUE_{home_abbr}")
    if a_bull["tax"] > 0.06:         signals.append(f"BULLPEN_FATIGUE_{away_abbr}")
    if abs(ump_b) > 0.04:            signals.append(f"UMP_BIAS_{ump_b:+.2f}_{ump_name}")
    if park_fac > 1.10:              signals.append(f"HIGH_PARK_{park_fac:.2f}x")
    if park_fac < 0.93:              signals.append("PITCHERS_PARK")
    if weather["wind_mph"] > 10:     signals.append(f"WIND_{weather['wind_mph']}mph_{weather['wind_dir']}")
    if weather["temp_f"] > 85:       signals.append("HOT_WEATHER")
    if weather["condition"] in ("Rain","Thunderstorm"): signals.append("RAIN_RISK")
    if hsp.get("provisional"):       signals.append(f"SP_{home_abbr}_PROVISIONAL")
    if asp.get("provisional"):       signals.append(f"SP_{away_abbr}_PROVISIONAL")
    if hsp["stuff_plus"] > 115:      signals.append(f"ELITE_STUFF_{home_abbr}_{hsp['stuff_plus']:.0f}")
    if asp["stuff_plus"] > 115:      signals.append(f"ELITE_STUFF_{away_abbr}_{asp['stuff_plus']:.0f}")

    # ── Tier ──────────────────────────────────────────────────────────────────
    if   ev_pct >= 7 and best_bet: tier = "TOP"
    elif ev_pct >= 4 and best_bet: tier = "GOOD"
    elif ev_pct >= 2 and best_bet: tier = "WATCH"
    else:                           tier = "SKIP"

    return {
        "game_metadata": {
            "game_id":    game_id, "game_pk": game_pk,
            "home": home_abbr, "away": away_abbr,
            "home_full": home_name, "away_full": away_name,
            "home_sp": sp_home_name, "away_sp": sp_away_name,
            "start_time": game_time, "venue": venue,
            "park_factor": round(park_fac, 3),
            "weather": weather, "umpire": ump_name or "TBD",
            "lineup_confirmed": lineup_confirmed,
            "status": "scheduled", "score": "Scheduled",
        },
        "projections": {
            "fair_home_ml": fair_home_ml, "fair_away_ml": fair_away_ml,
            "fair_home_p":  round(fair_home_p, 3),
            "fair_away_p":  round(fair_away_p, 3),
            "fair_f5_p":    round(fair_f5_p, 3),
            "proj_total":   proj_total,
            "home_proj_runs": home_proj_runs, "away_proj_runs": away_proj_runs,
            "home_woba":    home_woba, "away_woba": away_woba,
            "sp_home_era":  hsp["era"], "sp_away_era": asp["era"],
            "sp_home_fip":  hsp["fip"], "sp_away_fip": asp["fip"],
            "sp_home_k9":   hsp["k9"], "sp_away_k9":  asp["k9"],
            "sp_home_stuff": hsp["stuff_plus"], "sp_away_stuff": asp["stuff_plus"],
            "h_bull_tax":   h_bull["tax"], "a_bull_tax": a_bull["tax"],
            "h_home_wr":    round(h_home_wr, 3), "a_away_wr": round(a_away_wr, 3),
            "ump_bias":     round(ump_b, 4), "park_fac": round(park_fac, 3),
            "market_home_ml": mkt_home_ml, "market_away_ml": mkt_away_ml,
            "market_home_p":  round(mkt_home_p, 3) if mkt_home_p else None,
            "market_away_p":  round(mkt_away_p, 3) if mkt_away_p else None,
            "odds_source": book,
        },
        "market_edge": {
            "best_bet":    best_bet or "NO EDGE",
            "f5_bet":      f5_bet,
            "ev_percent":  ev_pct,
            "kelly_stake": kstake,
            "tier":        tier,
            "bet_side":    bet_side,
            "signals":     signals,
        },
        "math": {
            "base_pyth_p":   round(base_home_p, 4),
            "prop_adj":      round(prop_adj, 4),
            "ump_adj":       round(ump_b, 4),
            "h_fatigue_tax": round(h_bull["tax"], 3),
            "a_fatigue_tax": round(a_bull["tax"], 3),
            "final_home_p":  round(fair_home_p, 4),
            "mkt_home_p":    round(mkt_home_p, 4) if mkt_home_p else None,
            "h_home_wr":     round(h_home_wr, 3),
            "a_away_wr":     round(a_away_wr, 3),
            "park_factor":   round(park_fac, 3),
            "ump_bias":      round(ump_b, 4),
        }
    }

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().strftime("%Y-%m-%d")
    run_type = sys.argv[2] if len(sys.argv) > 2 else "full"

    print(f"\n{'='*60}")
    print(f"  DiamondEdge Engine v3 | {date_str} | {run_type}")
    print(f"  Sources: FanGraphs (ATC+BATX+Steamer) + Savant + MLB API")
    print(f"  + Odds API + OpenWeatherMap + 33k-game trained lookup")
    print(f"{'='*60}")

    # Pull all projection data
    batting_proj, pitching_proj = load_fg_projections()

    print("\n[MLB API] Fetching schedule...")
    games = get_schedule(date_str)
    print(f"  {len(games)} games")

    print("\n[Odds API] Fetching moneylines...")
    odds_map = get_odds()

    print("\n[Model] Running ensemble...")
    picks = []
    for game in games:
        try:
            result = run_model(game, batting_proj, pitching_proj, odds_map, date_str)
            picks.append(result)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()

    # Sort by tier then EV
    to = {"TOP":0,"GOOD":1,"WATCH":2,"SKIP":3}
    picks.sort(key=lambda x: (to.get(x["market_edge"]["tier"],3), -x["market_edge"]["ev_percent"]))

    output = {
        "date":       date_str,
        "run_type":   run_type,
        "generated":  datetime.datetime.utcnow().isoformat() + "Z",
        "game_count": len(picks),
        "model": {
            "version":    "v3.0",
            "sources":    ["ATC","THE BAT X","Steamer","Stuff+","MLB API","Odds API","OpenWeatherMap","Retrosheet Lookup"],
            "ensemble":   {"atc":0.40,"batx":0.30,"steamer":0.20,"stuff_plus":0.10},
            "train_games": LOOKUP["meta"]["train_games"],
            "ev_min":     EV_MIN,
        },
        "picks": picks,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    ev_picks = [p for p in picks if p["market_edge"]["tier"] in ("TOP","GOOD")]
    print(f"\n✅ {len(picks)} games | {len(ev_picks)} value picks (EV ≥ {EV_MIN*100:.0f}%)")
    print(f"   Saved → {OUT_PATH}")
    for p in ev_picks:
        me = p["market_edge"]; gm = p["game_metadata"]
        print(f"   [{me['tier']}] {gm['away']}@{gm['home']} | {me['best_bet']} | EV={me['ev_percent']}% | Kelly={me['kelly_stake']}")

if __name__ == "__main__":
    main()
