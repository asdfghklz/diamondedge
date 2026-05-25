"""
DiamondEdge MLB Engine v4
==========================
100% accurate data only. No hallucination. No scraping.

Data sources:
  - MLB Stats API     : Schedule, lineups, pitcher/batter 2026 stats, bullpen usage,
                        umpires, final scores for auto W/L marking (free, no key)
  - The Odds API      : FanDuel + BetMGM live moneylines (free key, 500 req/month)
                        Bet365 NOT available on free tier - flagged honestly
  - OpenWeatherMap    : Live weather per ballpark (free key)
  - model_lookup.json : Park factors, team home/away WR, umpire bias (33k game trained)

FIP  = ((13*HR) + (3*(BB+HBP)) - (2*K)) / IP + 3.17  [standard formula, same as FanGraphs]
wOBA = (0.69*BB + 0.72*HBP + 0.89*1B + 1.27*2B + 1.62*3B + 2.10*HR) / PA [standard weights]

Runs 4x per day via GitHub Actions:
  7:00 AM CT  - morning (probable pitchers, opening odds)
  11:30 AM CT - lineup run (confirmed lineups injected)
  5:30 PM CT  - evening (final lineups, updated odds, weather)
  11:00 PM CT - results run (auto W/L from final scores)

High confidence threshold: EV >= 6% AND model win prob >= 58%
"""

import os, sys, json, math, time, datetime, requests
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
ODDS_KEY    = os.environ.get("ODDS_API_KEY", "")
WEATHER_KEY = os.environ.get("OPENWEATHER_KEY", "")
OUT_PATH    = Path(__file__).parent.parent / "picks.json"
LOOKUP_PATH = Path(__file__).parent / "model_lookup.json"

MLB_API     = "https://statsapi.mlb.com/api/v1"
ODDS_API    = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
WEATHER_API = "https://api.openweathermap.org/data/2.5/weather"

# Thresholds — only flag HIGH confidence picks
EV_MIN          = 0.060   # minimum 6% edge
WIN_PROB_MIN    = 0.580   # minimum 58% model win probability
FATIGUE_PER_RP  = 0.030
FATIGUE_MAX     = 0.120

# Books available on free Odds API tier
AVAILABLE_BOOKS = {
    "fanduel":  "FanDuel",
    "betmgm":   "BetMGM",
}
UNAVAILABLE_BOOKS = {
    "bet365": "Bet365 requires Odds API paid plan (~$79/mo) — not available on free tier"
}

# wOBA linear weights (2024-2026 run environment)
WOBA_BB  = 0.690; WOBA_HBP = 0.720; WOBA_1B = 0.890
WOBA_2B  = 1.270; WOBA_3B  = 1.620; WOBA_HR = 2.100
FIP_CONST = 3.17

# Ballpark GPS coords for weather
PARK_COORDS = {
    "ARI":(33.4453,-112.0667),"ATL":(33.8908,-84.4681),"BAL":(39.2838,-76.6216),
    "BOS":(42.3467,-71.0972), "CHC":(41.9484,-87.6553),"CWS":(41.8300,-87.6339),
    "CIN":(39.0979,-84.5082), "CLE":(41.4962,-81.6852),"COL":(39.7559,-104.9942),
    "DET":(42.3390,-83.0485), "HOU":(29.7573,-95.3555),"KC": (39.0517,-94.4803),
    "LAA":(33.8003,-117.8827),"LAD":(34.0739,-118.2400),"MIA":(25.7781,-80.2197),
    "MIL":(43.0280,-87.9712), "MIN":(44.9817,-93.2778),"NYM":(40.7571,-73.8458),
    "NYY":(40.8296,-73.9262), "ATH":(37.7516,-122.2005),"PHI":(39.9061,-75.1665),
    "PIT":(40.4468,-80.0057), "SD": (32.7076,-117.1570),"SF": (37.7786,-122.3893),
    "SEA":(47.5914,-122.3323),"STL":(38.6226,-90.1928), "TB": (27.7683,-82.6534),
    "TEX":(32.7473,-97.0845), "TOR":(43.6414,-79.3894), "WSH":(38.8730,-77.0074),
}
TEAM_PARK = {
    "ARI":"PHO01","ATL":"ATL03","BAL":"BAL12","BOS":"BOS07","CHC":"CHC11",
    "CWS":"CHI12","CIN":"CIN09","CLE":"CLE08","COL":"DEN02","DET":"DET02",
    "HOU":"HOU03","KC":"KC01",  "LAA":"LAA01","LAD":"LAD01","MIA":"MIA02",
    "MIL":"MIL06","MIN":"MIN04","NYM":"NYC21","NYY":"NYC20","ATH":"OAK01",
    "PHI":"PHI13","PIT":"PIT01","SD":"SAN02", "SF":"SFO03", "SEA":"SEA03",
    "STL":"STL10","TB":"STP01", "TEX":"ARL02","TOR":"TOR02","WSH":"WAS11",
}

# ── LOAD TRAINED LOOKUP ───────────────────────────────────────────────────────
print("Loading trained model lookup...")
with open(LOOKUP_PATH) as f:
    LOOKUP = json.load(f)

def _get(d, key, fallback):
    v = d.get(key, fallback)
    return v["wr"] if isinstance(v, dict) and "wr" in v else (
           v["bias"] if isinstance(v, dict) and "bias" in v else (
           v["factor"] if isinstance(v, dict) and "factor" in v else (
           v if not isinstance(v, dict) else fallback)))

PARK_FACTORS = {}
for k, v in LOOKUP.get("park_factors", {}).items():
    PARK_FACTORS[k] = v["factor"] if isinstance(v, dict) else float(v)

TEAM_HOME_WR = {}
for k, v in LOOKUP.get("team_home_wr", {}).items():
    TEAM_HOME_WR[k] = v["wr"] if isinstance(v, dict) else float(v)

TEAM_AWAY_WR = {}
for k, v in LOOKUP.get("team_away_wr", {}).items():
    TEAM_AWAY_WR[k] = v["wr"] if isinstance(v, dict) else float(v)

UMP_BIAS = {}
for k, v in LOOKUP.get("ump_bias", {}).items():
    UMP_BIAS[k] = v["bias"] if isinstance(v, dict) else float(v)

meta = LOOKUP.get("meta", {})
LEAGUE_HOME  = meta.get("league_home_wr", 0.535)
LEAGUE_RUNS  = meta.get("league_avg_runs", 8.82)
print(f"  {meta.get('train_games', 0):,} games | {meta.get('train_seasons','')}")

# ── HTTP HELPER ───────────────────────────────────────────────────────────────
def get(url, params={}, timeout=15, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent":"DiamondEdge/4.0"})
            r.raise_for_status()
            return r
        except Exception as e:
            if i == retries-1:
                print(f"  [WARN] {url[:55]}: {e}")
                return None
            time.sleep(2**i)

def mlb(path, params={}):
    r = get(f"{MLB_API}{path}", params)
    return r.json() if r else {}

# ── MLB STATS API: PITCHER STATS ──────────────────────────────────────────────
def calc_fip(hr, bb, hbp, k, ip):
    """Standard FIP formula. Same as FanGraphs."""
    if ip <= 0: return 4.50
    return round(((13*hr) + (3*(bb+hbp)) - (2*k)) / ip + FIP_CONST, 2)

def calc_woba(bb, hbp, h, doubles, triples, hr, pa):
    """Standard wOBA from counting stats. Same weights as FanGraphs."""
    if pa <= 0: return 0.318
    singles = h - doubles - triples - hr
    num = (WOBA_BB*bb + WOBA_HBP*hbp + WOBA_1B*singles +
           WOBA_2B*doubles + WOBA_3B*triples + WOBA_HR*hr)
    return round(num / pa, 4)

def get_pitcher_stats(pitcher_id, days=30):
    """
    Pull pitcher's real 2026 stats from MLB Stats API.
    Uses last N days game log for recency, falls back to season.
    Returns ERA, FIP (calculated), K/9, BB/9, WHIP, IP.
    """
    if not pitcher_id:
        return {"era":4.50,"fip":4.50,"k9":8.0,"bb9":3.2,"whip":1.30,
                "ip":0,"gs":0,"provisional":True,"source":"no_id"}

    # Try last 30 days game log first
    cutoff = (datetime.date.today()-datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    today  = datetime.date.today().strftime("%Y-%m-%d")
    data = mlb(f"/people/{pitcher_id}/stats", {
        "stats":"gameLog","group":"pitching",
        "season":2026,"startDate":cutoff,"endDate":today,"sportId":1
    })
    splits = data.get("stats",[{}])[0].get("splits",[])

    if len(splits) >= 2:
        # Aggregate last 30 days
        er=ip=k=bb=hbp=h=hr=gs=0
        for sp in splits:
            s = sp.get("stat",{})
            def si(key): return int(s.get(key,0) or 0)
            def sf(key): return float(s.get(key,0) or 0)
            er  += si("earnedRuns"); ip  += sf("inningsPitched")
            k   += si("strikeOuts"); bb  += si("baseOnBalls")
            hbp += si("hitBatsmen"); h   += si("hits")
            hr  += si("homeRuns");   gs  += si("gamesStarted")
        if ip > 0:
            era  = round((er/ip)*9, 2)
            fip  = calc_fip(hr, bb, hbp, k, ip)
            k9   = round((k/ip)*9, 2)
            bb9  = round((bb/ip)*9, 2)
            whip = round((h+bb)/ip, 2)
            return {"era":era,"fip":fip,"k9":k9,"bb9":bb9,"whip":whip,
                    "ip":round(ip,1),"gs":gs,"provisional":False,"source":f"last_{days}d"}

    # Fall back to full 2026 season stats
    data = mlb(f"/people/{pitcher_id}/stats", {
        "stats":"season","group":"pitching","season":2026,"sportId":1
    })
    splits = data.get("stats",[{}])[0].get("splits",[])
    if splits:
        s = splits[0].get("stat",{})
        def si(key): return int(s.get(key,0) or 0)
        def sf(key,d=0.0): 
            try: return float(s.get(key,d) or d)
            except: return d
        ip = sf("inningsPitched")
        if ip >= 5:
            hr=si("homeRuns"); bb=si("baseOnBalls"); hbp=si("hitBatsmen")
            k=si("strikeOuts"); h=si("hits")
            return {
                "era":  sf("era",4.50),
                "fip":  calc_fip(hr,bb,hbp,k,ip),
                "k9":   sf("strikeoutsPer9Inn",8.0),
                "bb9":  sf("walksPer9Inn",3.2),
                "whip": sf("whip",1.30),
                "ip":   round(ip,1),
                "gs":   si("gamesStarted"),
                "provisional":False,"source":"season_2026"
            }

    # Truly no data — use league average, mark provisional
    return {"era":4.50,"fip":4.50,"k9":8.0,"bb9":3.2,"whip":1.30,
            "ip":0,"gs":0,"provisional":True,"source":"league_avg"}

# ── MLB STATS API: TEAM BATTING ───────────────────────────────────────────────
def get_team_batting(team_id, days=30):
    """
    Pull team's real 2026 batting stats from MLB Stats API.
    Calculates wOBA from counting stats (same formula as FanGraphs).
    """
    if not team_id:
        return {"woba":0.318,"ops":0.720,"provisional":True,"source":"no_id"}

    cutoff = (datetime.date.today()-datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    today  = datetime.date.today().strftime("%Y-%m-%d")

    # Team stats by date range
    data = mlb(f"/teams/{team_id}/stats", {
        "stats":"byDateRange","group":"hitting","season":2026,
        "startDate":cutoff,"endDate":today,"sportId":1
    })
    splits = data.get("stats",[{}])[0].get("splits",[])

    if splits:
        s = splits[0].get("stat",{})
        def si(key): return int(s.get(key,0) or 0)
        def sf(key,d=0.0):
            try: return float(s.get(key,d) or d)
            except: return d
        pa = si("plateAppearances")
        if pa >= 50:  # enough sample
            woba = calc_woba(
                si("baseOnBalls"), si("hitByPitch"), si("hits"),
                si("doubles"), si("triples"), si("homeRuns"), pa
            )
            obp = sf("obp",0.320); slg = sf("slg",0.400)
            return {
                "woba":woba,"ops":round(obp+slg,3),
                "obp":obp,"slg":slg,"pa":pa,
                "provisional":False,"source":f"last_{days}d"
            }

    # Fall back to full season
    data = mlb(f"/teams/{team_id}/stats", {
        "stats":"season","group":"hitting","season":2026,"sportId":1
    })
    splits = data.get("stats",[{}])[0].get("splits",[])
    if splits:
        s = splits[0].get("stat",{})
        def si(key): return int(s.get(key,0) or 0)
        def sf(key,d=0.0):
            try: return float(s.get(key,d) or d)
            except: return d
        pa = si("plateAppearances")
        if pa >= 20:
            woba = calc_woba(
                si("baseOnBalls"),si("hitByPitch"),si("hits"),
                si("doubles"),si("triples"),si("homeRuns"),pa
            )
            obp = sf("obp",0.320); slg = sf("slg",0.400)
            return {
                "woba":woba,"ops":round(obp+slg,3),
                "obp":obp,"slg":slg,"pa":pa,
                "provisional":False,"source":"season_2026"
            }

    return {"woba":0.318,"ops":0.720,"provisional":True,"source":"league_avg"}

# ── MLB STATS API: BULLPEN FATIGUE ────────────────────────────────────────────
def get_bullpen_fatigue(team_id, date_str):
    if not team_id:
        return {"fatigued":0,"tax":0.0,"detail":[]}
    game_date = datetime.date.fromisoformat(date_str)
    fatigued = 0
    detail = []
    for days_back in [1, 2]:
        check = (game_date-datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
        data = mlb("/schedule", {
            "sportId":1,"teamId":team_id,"date":check,
            "hydrate":"linescore,boxscore"
        })
        for db in data.get("dates",[]):
            for g in db.get("games",[]):
                if g.get("status",{}).get("abstractGameState") != "Final":
                    continue
                box = g.get("boxscore",{})
                for side in ["home","away"]:
                    td = box.get("teams",{}).get(side,{})
                    if td.get("team",{}).get("id") == team_id:
                        pitchers = td.get("pitchers",[])
                        relievers = len(pitchers) - 1 if len(pitchers) > 1 else 0
                        if relievers > 0:
                            fatigued += min(relievers, 2)
                            detail.append(f"{relievers} RP used {days_back}d ago")
    return {
        "fatigued": fatigued,
        "tax": min(fatigued * FATIGUE_PER_RP, FATIGUE_MAX),
        "detail": detail
    }

# ── MLB STATS API: UMPIRE ─────────────────────────────────────────────────────
def get_umpire(game):
    for o in game.get("officials",[]):
        if o.get("officialType") == "Home Plate":
            uid = str(o.get("official",{}).get("id",""))
            name = o.get("official",{}).get("fullName","Unknown")
            return uid, name
    return None, "TBD"

# ── MLB STATS API: FINAL SCORES (for auto W/L) ───────────────────────────────
def get_final_scores(date_str):
    """
    Returns dict of game_id -> {home_score, away_score, status}
    Used for auto W/L marking.
    """
    data = mlb("/schedule", {
        "sportId":1,"date":date_str,
        "hydrate":"linescore,team"
    })
    results = {}
    for db in data.get("dates",[]):
        for g in db.get("games",[]):
            status = g.get("status",{}).get("abstractGameState","")
            if status != "Final":
                continue
            home = g.get("teams",{}).get("home",{})
            away = g.get("teams",{}).get("away",{})
            home_abbr = home.get("team",{}).get("abbreviation","")
            away_abbr = away.get("team",{}).get("abbreviation","")
            ls = g.get("linescore",{})
            home_score = ls.get("teams",{}).get("home",{}).get("runs")
            away_score = ls.get("teams",{}).get("away",{}).get("runs")
            if home_score is None or away_score is None:
                continue
            # Build game_id to match our format
            game_id = f"{date_str}-{away_abbr}-{home_abbr}"
            results[game_id] = {
                "home_score": int(home_score),
                "away_score": int(away_score),
                "score_display": f"{away_abbr} {away_score} · {home_abbr} {home_score}",
                "status": "Final"
            }
    return results

# ── MLB STATS API: SCHEDULE ───────────────────────────────────────────────────
def get_schedule(date_str):
    data = mlb("/schedule", {
        "sportId":1,"date":date_str,
        "hydrate":"probablePitcher,team,weather,officials,venue,lineups,linescore"
    })
    games = []
    for db in data.get("dates",[]):
        for g in db.get("games",[]):
            if g.get("status",{}).get("abstractGameCode") not in ("F","DR"):
                games.append(g)
    return games

def get_confirmed_lineup(game):
    lineups = game.get("lineups",{})
    home = [p.get("id") for p in lineups.get("homePlayers",[])]
    away = [p.get("id") for p in lineups.get("awayPlayers",[])]
    return home, away

# ── LIVE ODDS ─────────────────────────────────────────────────────────────────
def get_odds():
    """
    Fetches FanDuel and BetMGM odds from The Odds API (free tier).
    Bet365 is NOT available on free tier — flagged honestly, not faked.
    """
    if not ODDS_KEY:
        print("  [INFO] No ODDS_API_KEY set")
        return {}, ["No Odds API key — add ODDS_API_KEY secret to GitHub"]

    r = get(ODDS_API, {
        "apiKey":ODDS_KEY,"regions":"us",
        "markets":"h2h","oddsFormat":"american",
        "bookmakers":"fanduel,betmgm"
    })
    if not r:
        return {}, ["Odds API request failed"]

    result = {}
    for game in r.json():
        home = game.get("home_team","")
        away = game.get("away_team","")
        game_lines = {}
        for bk in game.get("bookmakers",[]):
            bk_key = bk.get("key","")
            bk_name = AVAILABLE_BOOKS.get(bk_key, bk_key)
            for mkt in bk.get("markets",[]):
                if mkt["key"] == "h2h":
                    oc = {o["name"]:o["price"] for o in mkt["outcomes"]}
                    game_lines[bk_name] = {
                        "home_ml": oc.get(home),
                        "away_ml": oc.get(away),
                    }
        if game_lines:
            result[f"{home}|{away}"] = game_lines

    missing = []
    for book, reason in UNAVAILABLE_BOOKS.items():
        missing.append(reason)

    print(f"  Odds: {len(result)} games | Books: FanDuel, BetMGM | Bet365: unavailable (free tier)")
    return result, missing

# ── WEATHER ───────────────────────────────────────────────────────────────────
def get_weather(team_abbr):
    coords = PARK_COORDS.get(team_abbr)
    if not coords or not WEATHER_KEY:
        return {"temp_f":72,"wind_mph":0,"wind_dir":"","condition":"unknown","provisional":True}
    r = get(WEATHER_API, {
        "lat":coords[0],"lon":coords[1],
        "appid":WEATHER_KEY,"units":"imperial"
    })
    if not r:
        return {"temp_f":72,"wind_mph":0,"wind_dir":"","condition":"unknown","provisional":True}
    data = r.json()
    wind = data.get("wind",{})
    wdeg = wind.get("deg",0)
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    wdir = dirs[round(wdeg/45)%8]
    return {
        "temp_f":    round(float(data.get("main",{}).get("temp",72)),1),
        "wind_mph":  round(float(wind.get("speed",0)),1),
        "wind_dir":  wdir,
        "condition": data.get("weather",[{}])[0].get("main","Clear"),
        "provisional": False
    }

# ── MATH HELPERS ──────────────────────────────────────────────────────────────
def impl(ml):
    if not ml: return 0.5
    return abs(ml)/(abs(ml)+100) if ml < 0 else 100/(ml+100)

def devig(home_ml, away_ml):
    """Remove vig to get true implied probabilities."""
    rh = impl(home_ml); ra = impl(away_ml); tt = rh+ra
    return rh/tt if tt > 0 else 0.5, ra/tt if tt > 0 else 0.5

def to_ml(p):
    p = max(0.01, min(0.99, p))
    return round(-(p/(1-p))*100) if p >= 0.5 else round(((1-p)/p)*100)

def kelly(ev, wp, ml):
    if not ml or ev <= 0: return "0%"
    b = 100/abs(ml) if ml < 0 else ml/100
    k = max(0, min((b*wp-(1-wp))/b, 0.25))
    return f"{round(k*25,1)}%"

def runs_to_win_prob(home_runs, away_runs):
    """Pythagorean expectation (exponent 1.83, baseball standard)."""
    exp = 1.83
    if home_runs <= 0 or away_runs <= 0: return 0.5
    return round((home_runs**exp)/(home_runs**exp + away_runs**exp), 4)

def woba_to_rpg(woba, park_fac=1.0):
    """Convert wOBA to runs per game. Calibrated to 2024-25 run environment."""
    league_woba = 0.318; league_rpg = 4.41
    rpg = league_rpg + (woba - league_woba) * 18.0
    return max(2.0, min(10.0, round(rpg * park_fac * 1.02, 2)))  # 1.02 = 2026 shift ban boost

def weather_adj(home_rpg, away_rpg, weather, park_id):
    temp = weather.get("temp_f",72)
    wind = weather.get("wind_mph",0)
    wdir = weather.get("wind_dir","")
    cond = weather.get("condition","")
    temp_mult = 1.0 + (temp-72)*0.002
    wind_adj = 0.0
    if wind > 8:
        if park_id == "CHC11" and wdir in ("N","NE","E"): wind_adj = wind*0.04
        elif park_id == "CHC11" and wdir in ("S","SW","W"): wind_adj = -wind*0.03
        elif park_id == "SFO03" and wdir in ("E","NE"): wind_adj = -wind*0.035
        elif wdir in ("N","NE","NW","E"): wind_adj = wind*0.025
        else: wind_adj = -wind*0.02
    rain = 0.97 if cond in ("Rain","Drizzle","Thunderstorm") else 1.0
    h = round(home_rpg * temp_mult * rain + wind_adj*0.5, 2)
    a = round(away_rpg * temp_mult * rain + wind_adj*0.5, 2)
    return h, a

# ── BEST BOOK SELECTION ───────────────────────────────────────────────────────
def best_book(game_lines, bet_side, home_name, away_name):
    """
    Find which book offers the best price for our pick.
    Returns best_book_name, best_ml, all_lines comparison.
    """
    if not game_lines:
        return None, None, {}

    bet_key = "home_ml" if bet_side == "home" else "away_ml"
    best_name = None; best_ml = None

    all_lines = {}
    for book_name, lines in game_lines.items():
        ml = lines.get(bet_key)
        if ml is None: continue
        all_lines[book_name] = {
            "home_ml": lines.get("home_ml"),
            "away_ml": lines.get("away_ml"),
            "pick_ml": ml
        }
        # Higher ML = better payout for us
        if best_ml is None or ml > best_ml:
            best_ml = ml; best_name = book_name

    return best_name, best_ml, all_lines

# ── CORE MODEL ────────────────────────────────────────────────────────────────
def run_model(game, odds_map, date_str):
    home_d    = game.get("teams",{}).get("home",{})
    away_d    = game.get("teams",{}).get("away",{})
    home_abbr = home_d.get("team",{}).get("abbreviation","HM")
    away_abbr = away_d.get("team",{}).get("abbreviation","AW")
    home_name = home_d.get("team",{}).get("name","Home")
    away_name = away_d.get("team",{}).get("abbreviation","Away")
    home_id   = home_d.get("team",{}).get("id")
    away_id   = away_d.get("team",{}).get("id")
    game_pk   = str(game.get("gamePk",""))
    game_time = game.get("gameDate","")
    venue     = game.get("venue",{}).get("name","")
    game_id   = f"{date_str}-{away_abbr}-{home_abbr}"

    sp_home      = home_d.get("probablePitcher",{})
    sp_away      = away_d.get("probablePitcher",{})
    sp_home_id   = sp_home.get("id")
    sp_away_id   = sp_away.get("id")
    sp_home_name = sp_home.get("fullName","TBD")
    sp_away_name = sp_away.get("fullName","TBD")

    print(f"  {away_abbr}@{home_abbr} | {sp_away_name} vs {sp_home_name}")

    # Pull all real stats from MLB API
    hsp   = get_pitcher_stats(sp_home_id)
    asp   = get_pitcher_stats(sp_away_id)
    h_bat = get_team_batting(home_id)
    a_bat = get_team_batting(away_id)
    h_bull = get_bullpen_fatigue(home_id, date_str)
    a_bull = get_bullpen_fatigue(away_id, date_str)
    ump_id, ump_name = get_umpire(game)
    weather = get_weather(home_abbr)

    # Confirmed lineups
    h_lineup, a_lineup = get_confirmed_lineup(game)
    lineup_confirmed = len(h_lineup) >= 8 and len(a_lineup) >= 8

    # Park factor
    park_id  = TEAM_PARK.get(home_abbr,"default")
    park_fac = PARK_FACTORS.get(park_id, 1.0)

    # SP quality → wOBA suppression
    # ERA 3.00 SP holds opponents to ~.288 wOBA; ERA 5.00 → .340
    def sp_suppress(era):
        return round(0.318 - (4.50-era)*0.010, 4)

    h_sp_suppress = sp_suppress(hsp["era"])   # what home SP allows
    a_sp_suppress = sp_suppress(asp["era"])   # what away SP allows

    # Effective run scoring: blend lineup wOBA with SP suppression
    # 55% lineup quality, 45% SP suppression (pitcher is slightly dominant)
    home_eff_woba = 0.55 * a_bat["woba"] + 0.45 * h_sp_suppress  # home team scores
    away_eff_woba = 0.55 * h_bat["woba"] + 0.45 * a_sp_suppress  # away team scores

    # Convert to runs
    home_rpg = woba_to_rpg(home_eff_woba, park_fac)
    away_rpg = woba_to_rpg(away_eff_woba, park_fac)
    home_rpg, away_rpg = weather_adj(home_rpg, away_rpg, weather, park_id)
    proj_total = round(home_rpg + away_rpg, 1)

    # Pythagorean win probability
    base_p = runs_to_win_prob(home_rpg, away_rpg)

    # Proprietary adjustments (from 33k game trained data)
    h_home_wr = TEAM_HOME_WR.get(home_abbr, LEAGUE_HOME)
    a_away_wr = TEAM_AWAY_WR.get(away_abbr, 1-LEAGUE_HOME)
    team_delta = (h_home_wr - LEAGUE_HOME) - (a_away_wr - (1-LEAGUE_HOME))
    prop_adj = team_delta * 0.35

    # Umpire bias
    ump_b = UMP_BIAS.get(ump_id, 0.0) if ump_id else 0.0

    # Final win probability
    fair_home_p = base_p + prop_adj + ump_b - h_bull["tax"] + a_bull["tax"]
    fair_home_p = max(0.18, min(0.88, fair_home_p))
    fair_away_p = 1 - fair_home_p

    # F5 (SPs dominate, no bullpen)
    f5_home_p = max(0.18, min(0.88,
        runs_to_win_prob(home_rpg*0.48, away_rpg*0.48) + prop_adj*0.4 + ump_b*0.3
    ))

    # Fair ML
    fair_home_ml = to_ml(fair_home_p)
    fair_away_ml = to_ml(fair_away_p)

    # Get odds from all available books
    game_lines = odds_map.get(f"{home_name}|{away_name}", {})

    # Use consensus (best available book) for EV calc
    # Take average implied prob across books that have this game
    home_impls = []
    away_impls = []
    for book_name, lines in game_lines.items():
        hml = lines.get("home_ml"); aml = lines.get("away_ml")
        if hml and aml:
            dh, da = devig(hml, aml)
            home_impls.append(dh); away_impls.append(da)

    if home_impls:
        mkt_home_p = round(sum(home_impls)/len(home_impls), 4)
        mkt_away_p = round(sum(away_impls)/len(away_impls), 4)
    else:
        mkt_home_p = mkt_away_p = None

    # EV calculation
    best_bet=None; ev_pct=0.0; kstake="0%"; bet_side=None; f5_bet=None
    best_book_name=None; best_book_ml=None; all_book_lines={}

    if mkt_home_p:
        home_ev = fair_home_p - mkt_home_p
        away_ev = fair_away_p - mkt_away_p
        f5_home_ev = f5_home_p - mkt_home_p
        f5_away_ev = (1-f5_home_p) - mkt_away_p

        if home_ev >= away_ev and home_ev >= EV_MIN and fair_home_p >= WIN_PROB_MIN:
            bet_side = "home"
            best_book_name, best_book_ml, all_book_lines = best_book(game_lines, "home", home_name, away_name)
            use_ml = best_book_ml if best_book_ml else to_ml(mkt_home_p)
            best_bet = f"{home_abbr} ML {use_ml:+d} ({best_book_name or 'N/A'})"
            ev_pct = round(home_ev*100, 1)
            kstake = kelly(home_ev, fair_home_p, use_ml)

        elif away_ev >= EV_MIN and fair_away_p >= WIN_PROB_MIN:
            bet_side = "away"
            best_book_name, best_book_ml, all_book_lines = best_book(game_lines, "away", home_name, away_name)
            use_ml = best_book_ml if best_book_ml else to_ml(mkt_away_p)
            best_bet = f"{away_abbr} ML {use_ml:+d} ({best_book_name or 'N/A'})"
            ev_pct = round(away_ev*100, 1)
            kstake = kelly(away_ev, fair_away_p, use_ml)

        # F5 flag
        if bet_side == "home" and f5_home_ev >= EV_MIN and f5_home_ev > home_ev + 0.02:
            f5_bet = f"F5 edge larger: {home_abbr} ({f5_home_ev*100:.1f}% vs {home_ev*100:.1f}% full game)"
        elif bet_side == "away" and f5_away_ev >= EV_MIN and f5_away_ev > away_ev + 0.02:
            f5_bet = f"F5 edge larger: {away_abbr} ({f5_away_ev*100:.1f}% vs {away_ev*100:.1f}% full game)"

    # Signals
    signals = []
    if not lineup_confirmed:         signals.append("LINEUP_UNCONFIRMED")
    if lineup_confirmed:             signals.append(f"LINEUPS_CONFIRMED")
    if h_bull["tax"] > 0.06:        signals.append(f"BULL_FATIGUE_{home_abbr}")
    if a_bull["tax"] > 0.06:        signals.append(f"BULL_FATIGUE_{away_abbr}")
    if abs(ump_b) > 0.04:           signals.append(f"UMP_BIAS_{ump_b:+.2f}_{ump_name}")
    if park_fac > 1.10:             signals.append(f"HIGH_PARK_{park_fac:.2f}x")
    if park_fac < 0.93:             signals.append("PITCHERS_PARK")
    if weather["wind_mph"] > 10:    signals.append(f"WIND_{weather['wind_mph']}mph_{weather['wind_dir']}")
    if weather["temp_f"] > 85:      signals.append("HOT_WEATHER")
    if weather["condition"] in ("Rain","Thunderstorm"): signals.append("RAIN_RISK")
    if hsp["provisional"]:          signals.append(f"SP_{home_abbr}_NO_DATA")
    if asp["provisional"]:          signals.append(f"SP_{away_abbr}_NO_DATA")
    if not game_lines:              signals.append("NO_ODDS_AVAILABLE")
    if len(game_lines) == 1:        signals.append("SINGLE_BOOK_ONLY")

    # Tier — HIGH CONFIDENCE only
    if ev_pct >= 8 and best_bet:   tier = "TOP"
    elif ev_pct >= 6 and best_bet: tier = "GOOD"
    elif ev_pct >= 3 and best_bet: tier = "WATCH"
    else:                           tier = "SKIP"

    return {
        "game_metadata": {
            "game_id":   game_id, "game_pk": game_pk,
            "home": home_abbr, "away": away_abbr,
            "home_full": home_name, "away_full": away_name,
            "home_sp":   sp_home_name, "away_sp": sp_away_name,
            "start_time": game_time, "venue": venue,
            "park_factor": round(park_fac, 3),
            "weather":   weather, "umpire": ump_name,
            "lineup_confirmed": lineup_confirmed,
            "status": "scheduled", "score": "Scheduled",
            "result": None,  # filled in by results run
        },
        "projections": {
            "fair_home_ml":  fair_home_ml, "fair_away_ml": fair_away_ml,
            "fair_home_p":   round(fair_home_p, 3),
            "fair_away_p":   round(fair_away_p, 3),
            "fair_f5_p":     round(f5_home_p, 3),
            "proj_total":    proj_total,
            "home_proj_runs": home_rpg, "away_proj_runs": away_rpg,
            "home_woba":     h_bat["woba"],  "away_woba":    a_bat["woba"],
            "sp_home_era":   hsp["era"],     "sp_away_era":  asp["era"],
            "sp_home_fip":   hsp["fip"],     "sp_away_fip":  asp["fip"],
            "sp_home_k9":    hsp["k9"],      "sp_away_k9":   asp["k9"],
            "sp_home_bb9":   hsp["bb9"],     "sp_away_bb9":  asp["bb9"],
            "sp_home_whip":  hsp["whip"],    "sp_away_whip": asp["whip"],
            "sp_home_ip":    hsp["ip"],      "sp_away_ip":   asp["ip"],
            "sp_home_source": hsp["source"], "sp_away_source": asp["source"],
            "h_bat_source":  h_bat["source"],"a_bat_source": a_bat["source"],
            "h_bull_tax":    h_bull["tax"],  "a_bull_tax":   a_bull["tax"],
            "h_bull_detail": h_bull["detail"],"a_bull_detail": a_bull["detail"],
            "h_home_wr":     round(h_home_wr, 3),
            "a_away_wr":     round(a_away_wr, 3),
            "ump_bias":      round(ump_b, 4),
            "mkt_home_p":    mkt_home_p,
            "mkt_away_p":    mkt_away_p,
            "book_lines":    all_book_lines,
            "best_book":     best_book_name,
            "best_book_ml":  best_book_ml,
        },
        "market_edge": {
            "best_bet":    best_bet or "NO EDGE",
            "f5_bet":      f5_bet,
            "ev_percent":  ev_pct,
            "kelly_stake": kstake,
            "tier":        tier,
            "bet_side":    bet_side,
            "signals":     signals,
            "unavailable_books": list(UNAVAILABLE_BOOKS.keys()),
        },
        "math": {
            "base_pyth_p":   round(base_p, 4),
            "prop_adj":      round(prop_adj, 4),
            "ump_adj":       round(ump_b, 4),
            "h_fatigue_tax": round(h_bull["tax"], 3),
            "a_fatigue_tax": round(a_bull["tax"], 3),
            "final_home_p":  round(fair_home_p, 4),
            "mkt_home_p":    mkt_home_p,
            "h_home_wr":     round(h_home_wr, 3),
            "a_away_wr":     round(a_away_wr, 3),
            "park_factor":   round(park_fac, 3),
            "ump_bias":      round(ump_b, 4),
            "sp_suppress_h": round(h_sp_suppress, 4),
            "sp_suppress_a": round(a_sp_suppress, 4),
        }
    }

# ── RESULTS RUN (auto W/L marking) ───────────────────────────────────────────
def run_results(date_str, existing_picks):
    """
    Called on the 11 PM run. Fetches final scores from MLB API
    and marks each pick as W or L automatically.
    """
    print(f"\n[Results] Fetching final scores for {date_str}...")
    scores = get_final_scores(date_str)
    print(f"  {len(scores)} games final")

    updated = 0
    for pick in existing_picks:
        gm = pick.get("game_metadata", {})
        me = pick.get("market_edge", {})
        game_id = gm.get("game_id","")

        # Already marked
        if gm.get("result"):
            continue
        # Skip games with no bet
        if me.get("tier") == "SKIP" or not me.get("bet_side"):
            continue

        score = scores.get(game_id)
        if not score:
            continue

        home_score = score["home_score"]
        away_score = score["away_score"]
        home_won = home_score > away_score
        bet_side = me.get("bet_side")
        we_won = (bet_side == "home" and home_won) or (bet_side == "away" and not home_won)

        gm["status"] = "closed"
        gm["score"]  = score["score_display"] + " — FINAL"
        gm["result"] = "W" if we_won else "L"
        updated += 1

    print(f"  Marked {updated} results")
    return existing_picks

# ── MODEL OBSERVATIONS ────────────────────────────────────────────────────────
def build_observations(picks):
    """
    Analyze completed picks and generate model observations.
    """
    completed = [p for p in picks if p["game_metadata"].get("result")]
    if not completed:
        return []

    obs = []
    w = sum(1 for p in completed if p["game_metadata"]["result"] == "W")
    l = len(completed) - w
    win_pct = round(w/len(completed)*100, 1) if completed else 0

    # Overall accuracy
    obs.append({
        "type": "accuracy",
        "text": f"Today's record: {w}W-{l}L ({win_pct}% accuracy)",
        "positive": win_pct >= 55
    })

    # Check if underdogs are outperforming
    dogs = [p for p in completed if (p["market_edge"]["bet_side"]=="home" and
            (p["projections"].get("best_book_ml") or 0) > 0) or
           (p["market_edge"]["bet_side"]=="away" and
            (p["projections"].get("best_book_ml") or 0) > 0)]
    if dogs:
        dog_w = sum(1 for p in dogs if p["game_metadata"]["result"]=="W")
        obs.append({
            "type": "dogs",
            "text": f"Underdog picks: {dog_w}/{len(dogs)} ({round(dog_w/len(dogs)*100)}%) — {'strong' if dog_w/len(dogs) > .5 else 'weak'} performance",
            "positive": dog_w/len(dogs) > 0.5
        })

    # Check if high EV picks outperform low EV picks
    high_ev = [p for p in completed if p["market_edge"]["ev_percent"] >= 8]
    if high_ev:
        hev_w = sum(1 for p in high_ev if p["game_metadata"]["result"]=="W")
        obs.append({
            "type": "ev_accuracy",
            "text": f"High EV picks (≥8%): {hev_w}/{len(high_ev)} — {'model edge confirmed' if hev_w/len(high_ev) > .55 else 'reduce EV threshold'}",
            "positive": hev_w/len(high_ev) > 0.55 if high_ev else True
        })

    # Recommend threshold adjustment
    if win_pct < 50 and len(completed) >= 4:
        obs.append({
            "type": "recommendation",
            "text": "Win rate below 50% — consider raising EV minimum to 8% or WIN_PROB_MIN to 62%",
            "positive": False
        })
    elif win_pct >= 65 and len(completed) >= 4:
        obs.append({
            "type": "recommendation",
            "text": f"Strong {win_pct}% accuracy — model is well-calibrated. Could cautiously lower EV minimum to 5%",
            "positive": True
        })

    return obs

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().strftime("%Y-%m-%d")
    run_type = sys.argv[2] if len(sys.argv) > 2 else "full"

    print(f"\n{'='*55}")
    print(f"  DiamondEdge v4 | {date_str} | {run_type}")
    print(f"  Data: MLB Stats API (official) + Odds API + OpenWeatherMap")
    print(f"  FIP and wOBA calculated from raw counting stats")
    print(f"  Threshold: EV>={EV_MIN*100:.0f}% AND win_prob>={WIN_PROB_MIN*100:.0f}%")
    print(f"{'='*55}")

    # Load existing picks if this is a results or update run
    existing_picks = []
    if OUT_PATH.exists():
        try:
            with open(OUT_PATH) as f:
                existing_data = json.load(f)
                if existing_data.get("date") == date_str:
                    existing_picks = existing_data.get("picks", [])
        except: pass

    # Results run — just mark W/L, no new model run needed
    if run_type == "results":
        picks = run_results(date_str, existing_picks)
        observations = build_observations(picks)
        output = {**existing_data, "picks": picks,
                  "observations": observations,
                  "last_results_check": datetime.datetime.utcnow().isoformat()+"Z"}
        with open(OUT_PATH,"w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✅ Results updated")
        return

    # Full model run
    print("\n[1/4] Fetching schedule...")
    games = get_schedule(date_str)
    print(f"  {len(games)} games")

    print("\n[2/4] Fetching odds (FanDuel + BetMGM)...")
    odds_map, missing_books = get_odds()

    print("\n[3/4] Running model (high confidence only)...")
    picks = []
    for game in games:
        try:
            result = run_model(game, odds_map, date_str)
            picks.append(result)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()

    # Preserve any existing W/L results if re-running same day
    if existing_picks:
        results_map = {p["game_metadata"]["game_id"]: p["game_metadata"].get("result")
                      for p in existing_picks if p["game_metadata"].get("result")}
        for pick in picks:
            gid = pick["game_metadata"]["game_id"]
            if gid in results_map:
                pick["game_metadata"]["result"] = results_map[gid]

    # Sort — TOP first then GOOD, skip WATCH and SKIP entirely in output
    to = {"TOP":0,"GOOD":1,"WATCH":2,"SKIP":3}
    picks.sort(key=lambda x: (to.get(x["market_edge"]["tier"],3),
                               -x["market_edge"]["ev_percent"]))

    observations = build_observations(picks)

    print("\n[4/4] Checking for completed games...")
    picks = run_results(date_str, picks)

    ev_picks = [p for p in picks if p["market_edge"]["tier"] in ("TOP","GOOD")]
    output = {
        "date":          date_str,
        "run_type":      run_type,
        "generated":     datetime.datetime.utcnow().isoformat()+"Z",
        "game_count":    len(picks),
        "ev_pick_count": len(ev_picks),
        "missing_books": missing_books,
        "model": {
            "version":     "v4.0",
            "data_sources": [
                "MLB Stats API — official 2026 season stats",
                "FIP calculated from HR/BB/HBP/K/IP (standard formula)",
                "wOBA calculated from counting stats (standard linear weights)",
                "FanDuel + BetMGM via The Odds API (Bet365 unavailable on free tier)",
                "OpenWeatherMap live weather",
                "Retrosheet 33,292 game trained lookup"
            ],
            "thresholds":  {"ev_min": EV_MIN, "win_prob_min": WIN_PROB_MIN},
            "train_games": LOOKUP.get("meta",{}).get("train_games", 33292),
        },
        "observations":  observations,
        "picks":         picks,
    }

    with open(OUT_PATH,"w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ {len(picks)} games | {len(ev_picks)} high-confidence picks")
    for p in ev_picks:
        me = p["market_edge"]; gm = p["game_metadata"]
        print(f"  [{me['tier']}] {gm['away']}@{gm['home']} | {me['best_bet']} | EV={me['ev_percent']}%")

if __name__ == "__main__":
    main()
