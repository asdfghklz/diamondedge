"""
DiamondEdge MLB Engine v4
==========================
100% accurate data. No FanGraphs scraping. No Savant scraping.
All data from MLB Stats API (free, no key) + The Odds API + OpenWeatherMap.

FIP  = ((13*HR + 3*(BB+HBP) - 2*K) / IP) + 3.17
wOBA = standard linear weights from counting stats

Runs 4x daily via GitHub Actions:
  7:00 AM CT  - morning picks
  11:30 AM CT - lineup update
  5:30 PM CT  - evening update
  11:00 PM CT - auto W/L results
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

# Model thresholds (calibrated from 33k game dataset)
EV_MIN         = 0.080   # 8% minimum edge
WIN_PROB_MIN   = 0.580   # 58% minimum win probability
FATIGUE_PER_RP = 0.030
FATIGUE_MAX    = 0.120
FIP_CONST      = 3.17

# wOBA weights (2024-2026 run environment)
WOBA_BB=0.690; WOBA_HBP=0.720; WOBA_1B=0.890
WOBA_2B=1.270; WOBA_3B=1.620; WOBA_HR=2.100

# Ballpark GPS for weather
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
TEAM_FULL = {
    "ARI":"Arizona Diamondbacks","ATL":"Atlanta Braves","BAL":"Baltimore Orioles",
    "BOS":"Boston Red Sox","CHC":"Chicago Cubs","CWS":"Chicago White Sox",
    "CIN":"Cincinnati Reds","CLE":"Cleveland Guardians","COL":"Colorado Rockies",
    "DET":"Detroit Tigers","HOU":"Houston Astros","KC":"Kansas City Royals",
    "LAA":"Los Angeles Angels","LAD":"Los Angeles Dodgers","MIA":"Miami Marlins",
    "MIL":"Milwaukee Brewers","MIN":"Minnesota Twins","NYM":"New York Mets",
    "NYY":"New York Yankees","ATH":"Athletics","PHI":"Philadelphia Phillies",
    "PIT":"Pittsburgh Pirates","SD":"San Diego Padres","SF":"San Francisco Giants",
    "SEA":"Seattle Mariners","STL":"St. Louis Cardinals","TB":"Tampa Bay Rays",
    "TEX":"Texas Rangers","TOR":"Toronto Blue Jays","WSH":"Washington Nationals",
}

# ── LOAD LOOKUP ───────────────────────────────────────────────────────────────
print("Loading trained model lookup (33,292 games 2010-2024)...")
with open(LOOKUP_PATH) as f:
    LOOKUP = json.load(f)

def _extract(d, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v["wr"] if isinstance(v,dict) and "wr" in v else \
                   v["bias"] if isinstance(v,dict) and "bias" in v else \
                   v["factor"] if isinstance(v,dict) and "factor" in v else \
                   (v if not isinstance(v,dict) else None)
    return None

PARK_FACTORS = {k: float(v["factor"] if isinstance(v,dict) else v) for k,v in LOOKUP.get("park_factors",{}).items()}
TEAM_HOME_WR = {k: float(v["wr"] if isinstance(v,dict) else v) for k,v in LOOKUP.get("team_home_wr",{}).items()}
TEAM_AWAY_WR = {k: float(v["wr"] if isinstance(v,dict) else v) for k,v in LOOKUP.get("team_away_wr",{}).items()}
UMP_BIAS     = {k: float(v["bias"] if isinstance(v,dict) else v) for k,v in LOOKUP.get("ump_bias",{}).items()}
meta         = LOOKUP.get("meta",{})
LEAGUE_HOME  = meta.get("league_home_wr", 0.535)
LEAGUE_RUNS  = meta.get("league_avg_runs", 8.82)
print(f"  Loaded: {meta.get('train_games',0):,} games | {meta.get('train_seasons','')}")

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

# ── MATH HELPERS ──────────────────────────────────────────────────────────────
def calc_fip(hr, bb, hbp, k, ip):
    return round(((13*hr)+(3*(bb+hbp))-(2*k))/ip+FIP_CONST, 2) if ip>0 else 4.50

def calc_woba(bb, hbp, h, doubles, triples, hr, pa):
    if pa<=0: return 0.318
    singles = h - doubles - triples - hr
    return round((WOBA_BB*bb+WOBA_HBP*hbp+WOBA_1B*singles+WOBA_2B*doubles+WOBA_3B*triples+WOBA_HR*hr)/pa, 4)

def impl(ml):
    if not ml: return 0.5
    return abs(ml)/(abs(ml)+100) if ml<0 else 100/(ml+100)

def devig(hml, aml):
    rh=impl(hml); ra=impl(aml); tt=rh+ra
    return (rh/tt, ra/tt) if tt>0 else (0.5, 0.5)

def to_ml(p):
    p=max(0.01,min(0.99,p))
    return round(-(p/(1-p))*100) if p>=0.5 else round(((1-p)/p)*100)

def kelly(ev, wp, ml):
    if not ml or ev<=0: return "0%"
    b=100/abs(ml) if ml<0 else ml/100
    k=max(0,min((b*wp-(1-wp))/b,0.25))
    return f"{round(k*25,1)}%"

def woba_to_rpg(woba, park_fac=1.0):
    rpg = 4.41+(woba-0.318)*18.0
    return max(2.0,min(10.0,round(rpg*park_fac*1.02,2)))

def runs_to_win_prob(hr, ar):
    exp=1.83
    if hr<=0 or ar<=0: return 0.5
    return round((hr**exp)/(hr**exp+ar**exp),4)

def weather_adj(home_rpg, away_rpg, weather, park_id):
    temp=weather.get("temp_f",72); wind=weather.get("wind_mph",0); wdir=weather.get("wind_dir","")
    cond=weather.get("condition","")
    tm=1.0+(temp-72)*0.002
    wa=0.0
    if wind>8:
        if park_id=="CHC11" and wdir in("N","NE","E"): wa=wind*0.04
        elif park_id=="CHC11" and wdir in("S","SW","W"): wa=-wind*0.03
        elif park_id=="SFO03" and wdir in("E","NE"): wa=-wind*0.035
        elif wdir in("N","NE","NW","E"): wa=wind*0.025
        else: wa=-wind*0.02
    rm=0.97 if cond in("Rain","Drizzle","Thunderstorm") else 1.0
    return round(home_rpg*tm*rm+wa*0.5,2), round(away_rpg*tm*rm+wa*0.5,2)

# ── MLB API DATA FETCHERS ─────────────────────────────────────────────────────
def get_schedule(date_str):
    data=mlb("/schedule",{"sportId":1,"date":date_str,"hydrate":"probablePitcher,team,weather,officials,venue,lineups,linescore"})
    games=[]
    for db in data.get("dates",[]):
        for g in db.get("games",[]):
            if g.get("status",{}).get("abstractGameCode") not in("F","DR"):
                games.append(g)
    return games

def get_confirmed_lineup(game):
    lineups=game.get("lineups",{})
    home=[p.get("id") for p in lineups.get("homePlayers",[])]
    away=[p.get("id") for p in lineups.get("awayPlayers",[])]
    return home, away

def get_pitcher_stats(pitcher_id, days=30):
    if not pitcher_id:
        return {"era":4.50,"fip":4.50,"k9":8.0,"bb9":3.2,"whip":1.30,"ip":0,"gs":0,"provisional":True,"source":"no_id"}
    cutoff=(datetime.date.today()-datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    today=datetime.date.today().strftime("%Y-%m-%d")
    data=mlb(f"/people/{pitcher_id}/stats",{"stats":"gameLog","group":"pitching","season":2026,"startDate":cutoff,"endDate":today,"sportId":1})
    splits=data.get("stats",[{}])[0].get("splits",[])
    if len(splits)>=2:
        er=ip=k=bb=hbp=h=hr=gs=0
        for sp in splits:
            s=sp.get("stat",{})
            def si(key): return int(s.get(key,0) or 0)
            def sf(key): return float(s.get(key,0) or 0)
            er+=si("earnedRuns"); ip+=sf("inningsPitched"); k+=si("strikeOuts")
            bb+=si("baseOnBalls"); hbp+=si("hitBatsmen"); h+=si("hits")
            hr+=si("homeRuns"); gs+=si("gamesStarted")
        # Require 25 IP minimum to trust last-30d stats (small samples are too noisy)
        if ip>=25:
            return {"era":round((er/ip)*9,2),"fip":calc_fip(hr,bb,hbp,k,ip),
                    "k9":round((k/ip)*9,2),"bb9":round((bb/ip)*9,2),
                    "whip":round((h+bb)/ip,2),"ip":round(ip,1),"gs":gs,
                    "provisional":False,"source":f"last_{days}d"}
    # Fall back to full season
    data=mlb(f"/people/{pitcher_id}/stats",{"stats":"season","group":"pitching","season":2026,"sportId":1})
    splits=data.get("stats",[{}])[0].get("splits",[])
    if splits:
        s=splits[0].get("stat",{})
        def si(key): return int(s.get(key,0) or 0)
        def sf(key,d=0.0):
            try: return float(s.get(key,d) or d)
            except: return d
        ip=sf("inningsPitched")
        if ip>=5:
            hr=si("homeRuns"); bb=si("baseOnBalls"); hbp=si("hitBatsmen"); k=si("strikeOuts"); h=si("hits")
            return {"era":sf("era",4.50),"fip":calc_fip(hr,bb,hbp,k,ip),
                    "k9":sf("strikeoutsPer9Inn",8.0),"bb9":sf("walksPer9Inn",3.2),
                    "whip":sf("whip",1.30),"ip":round(ip,1),"gs":si("gamesStarted"),
                    "provisional":False,"source":"season_2026"}
    return {"era":4.50,"fip":4.50,"k9":8.0,"bb9":3.2,"whip":1.30,"ip":0,"gs":0,"provisional":True,"source":"league_avg"}

def get_team_batting(team_id, days=30):
    if not team_id:
        return {"woba":0.318,"ops":0.720,"provisional":True,"source":"no_id"}
    cutoff=(datetime.date.today()-datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    today=datetime.date.today().strftime("%Y-%m-%d")
    data=mlb(f"/teams/{team_id}/stats",{"stats":"byDateRange","group":"hitting","season":2026,"startDate":cutoff,"endDate":today,"sportId":1})
    splits=data.get("stats",[{}])[0].get("splits",[])
    if splits:
        s=splits[0].get("stat",{})
        def si(key): return int(s.get(key,0) or 0)
        def sf(key,d=0.0):
            try: return float(s.get(key,d) or d)
            except: return d
        pa=si("plateAppearances")
        if pa>=50:
            woba=calc_woba(si("baseOnBalls"),si("hitByPitch"),si("hits"),si("doubles"),si("triples"),si("homeRuns"),pa)
            return {"woba":woba,"ops":round(sf("obp",0.320)+sf("slg",0.400),3),"pa":pa,"provisional":False,"source":f"last_{days}d"}
    data=mlb(f"/teams/{team_id}/stats",{"stats":"season","group":"hitting","season":2026,"sportId":1})
    splits=data.get("stats",[{}])[0].get("splits",[])
    if splits:
        s=splits[0].get("stat",{})
        def si(key): return int(s.get(key,0) or 0)
        def sf(key,d=0.0):
            try: return float(s.get(key,d) or d)
            except: return d
        pa=si("plateAppearances")
        if pa>=20:
            woba=calc_woba(si("baseOnBalls"),si("hitByPitch"),si("hits"),si("doubles"),si("triples"),si("homeRuns"),pa)
            return {"woba":woba,"ops":round(sf("obp",0.320)+sf("slg",0.400),3),"pa":pa,"provisional":False,"source":"season_2026"}
    return {"woba":0.318,"ops":0.720,"provisional":True,"source":"league_avg"}

def get_bullpen_fatigue(team_id, date_str):
    if not team_id: return {"fatigued":0,"tax":0.0,"detail":[]}
    game_date=datetime.date.fromisoformat(date_str)
    fatigued=0; detail=[]
    for days_back in [1,2]:
        check=(game_date-datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
        data=mlb("/schedule",{"sportId":1,"teamId":team_id,"date":check,"hydrate":"linescore,boxscore"})
        for db in data.get("dates",[]):
            for g in db.get("games",[]):
                if g.get("status",{}).get("abstractGameState")!="Final": continue
                box=g.get("boxscore",{})
                for side in ["home","away"]:
                    td=box.get("teams",{}).get(side,{})
                    if td.get("team",{}).get("id")==team_id:
                        pitchers=td.get("pitchers",[])
                        relievers=len(pitchers)-1 if len(pitchers)>1 else 0
                        if relievers>0:
                            fatigued+=min(relievers,2)
                            detail.append(f"{relievers} RP used {days_back}d ago")
    return {"fatigued":fatigued,"tax":min(fatigued*FATIGUE_PER_RP,FATIGUE_MAX),"detail":detail}

def get_umpire(game):
    for o in game.get("officials",[]):
        if o.get("officialType")=="Home Plate":
            uid=str(o.get("official",{}).get("id",""))
            name=o.get("official",{}).get("fullName","Unknown")
            return uid,name
    return None,"TBD"

def get_final_scores(date_str):
    data=mlb("/schedule",{"sportId":1,"date":date_str,"hydrate":"linescore,team"})
    results={}
    for db in data.get("dates",[]):
        for g in db.get("games",[]):
            if g.get("status",{}).get("abstractGameState")!="Final": continue
            home=g.get("teams",{}).get("home",{}); away=g.get("teams",{}).get("away",{})
            home_abbr=home.get("team",{}).get("abbreviation","")
            away_abbr=away.get("team",{}).get("abbreviation","")
            ls=g.get("linescore",{})
            home_score=ls.get("teams",{}).get("home",{}).get("runs")
            away_score=ls.get("teams",{}).get("away",{}).get("runs")
            if home_score is None or away_score is None: continue
            game_id=f"{date_str}-{away_abbr}-{home_abbr}"
            innings=ls.get("innings",[])
            f1_home=f1_away=None
            if innings:
                f1_home=innings[0].get("home",{}).get("runs")
                f1_away=innings[0].get("away",{}).get("runs")
            results[game_id]={"home_score":int(home_score),"away_score":int(away_score),
                              "score_display":f"{away_abbr} {away_score} · {home_abbr} {home_score}",
                              "status":"Final","f1_home":f1_home,"f1_away":f1_away}
    return results

def get_odds():
    if not ODDS_KEY:
        print("  [INFO] No ODDS_API_KEY"); return {}, []
    r=get(ODDS_API,{"apiKey":ODDS_KEY,"regions":"us","markets":"h2h","oddsFormat":"american","bookmakers":"fanduel,betmgm"})
    if not r: return {}, ["Odds API failed"]
    result={}
    for game in r.json():
        home=game.get("home_team",""); away=game.get("away_team","")
        game_lines={}
        for bk in game.get("bookmakers",[]):
            bk_name={"fanduel":"FanDuel","betmgm":"BetMGM"}.get(bk.get("key",""),bk.get("key",""))
            for mkt in bk.get("markets",[]):
                if mkt["key"]=="h2h":
                    oc={o["name"]:o["price"] for o in mkt["outcomes"]}
                    game_lines[bk_name]={"home_ml":oc.get(home),"away_ml":oc.get(away)}
        if game_lines: result[f"{home}|{away}"]=game_lines
    print(f"  Odds: {len(result)} games | FanDuel + BetMGM")
    return result, []

def get_weather(team_abbr):
    coords=PARK_COORDS.get(team_abbr)
    if not coords or not WEATHER_KEY:
        return {"temp_f":72,"wind_mph":0,"wind_dir":"","condition":"unknown","provisional":True}
    r=get(WEATHER_API,{"lat":coords[0],"lon":coords[1],"appid":WEATHER_KEY,"units":"imperial"})
    if not r: return {"temp_f":72,"wind_mph":0,"wind_dir":"","condition":"unknown","provisional":True}
    data=r.json(); wind=data.get("wind",{})
    wdeg=wind.get("deg",0); dirs=["N","NE","E","SE","S","SW","W","NW"]
    return {"temp_f":round(float(data.get("main",{}).get("temp",72)),1),
            "wind_mph":round(float(wind.get("speed",0)),1),
            "wind_dir":dirs[round(wdeg/45)%8],
            "condition":data.get("weather",[{}])[0].get("main","Clear"),
            "provisional":False}

def best_book_for_side(game_lines, bet_side, home_name, away_name):
    if not game_lines: return None,None,{}
    bet_key="home_ml" if bet_side=="home" else "away_ml"
    best_name=None; best_ml=None; all_lines={}
    for book_name,lines in game_lines.items():
        ml=lines.get(bet_key)
        if ml is None: continue
        all_lines[book_name]={"home_ml":lines.get("home_ml"),"away_ml":lines.get("away_ml"),"pick_ml":ml}
        if best_ml is None or ml>best_ml: best_ml=ml; best_name=book_name
    return best_name,best_ml,all_lines

# ── MAIN GAME MODEL ───────────────────────────────────────────────────────────
def run_model(game, odds_map, date_str):
    home_d=game.get("teams",{}).get("home",{}); away_d=game.get("teams",{}).get("away",{})
    home_abbr=home_d.get("team",{}).get("abbreviation","HM"); away_abbr=away_d.get("team",{}).get("abbreviation","AW")
    home_name=home_d.get("team",{}).get("name","Home"); away_name=away_d.get("team",{}).get("name","Away")
    home_id=home_d.get("team",{}).get("id"); away_id=away_d.get("team",{}).get("id")
    game_pk=str(game.get("gamePk","")); game_time=game.get("gameDate","")
    venue=game.get("venue",{}).get("name","")
    sp_home=home_d.get("probablePitcher",{}); sp_away=away_d.get("probablePitcher",{})
    sp_home_id=sp_home.get("id"); sp_away_id=sp_away.get("id")
    sp_home_name=sp_home.get("fullName","TBD"); sp_away_name=sp_away.get("fullName","TBD")
    game_id=f"{date_str}-{away_abbr}-{home_abbr}"
    print(f"  {away_abbr}@{home_abbr} | {sp_away_name} vs {sp_home_name}")

    hsp=get_pitcher_stats(sp_home_id); asp=get_pitcher_stats(sp_away_id)
    h_bat=get_team_batting(home_id); a_bat=get_team_batting(away_id)
    h_bull=get_bullpen_fatigue(home_id,date_str); a_bull=get_bullpen_fatigue(away_id,date_str)
    ump_id,ump_name=get_umpire(game); weather=get_weather(home_abbr)
    h_lineup,a_lineup=get_confirmed_lineup(game)
    lineup_confirmed=len(h_lineup)>=8 and len(a_lineup)>=8
    park_id=TEAM_PARK.get(home_abbr,"default"); park_fac=PARK_FACTORS.get(park_id,1.0)

    def sp_suppress(era): return round(0.318-(4.50-era)*0.010,4)
    h_sup=sp_suppress(hsp["era"]); a_sup=sp_suppress(asp["era"])
    home_eff=0.55*a_bat["woba"]+0.45*h_sup; away_eff=0.55*h_bat["woba"]+0.45*a_sup
    home_rpg=woba_to_rpg(home_eff,park_fac); away_rpg=woba_to_rpg(away_eff,park_fac)
    home_rpg,away_rpg=weather_adj(home_rpg,away_rpg,weather,park_id)
    proj_total=round(home_rpg+away_rpg,1)

    base_p=runs_to_win_prob(home_rpg,away_rpg)
    h_home_wr=TEAM_HOME_WR.get(home_abbr,LEAGUE_HOME); a_away_wr=TEAM_AWAY_WR.get(away_abbr,1-LEAGUE_HOME)
    team_delta=(h_home_wr-LEAGUE_HOME)-(a_away_wr-(1-LEAGUE_HOME))
    prop_adj=team_delta*0.50  # calibrated from 33k game data
    ump_b=UMP_BIAS.get(ump_id,0.0) if ump_id else 0.0

    fair_home_p=max(0.18,min(0.88,base_p+prop_adj+ump_b-h_bull["tax"]+a_bull["tax"]))
    fair_away_p=1-fair_home_p
    f5_home_p=max(0.18,min(0.88,runs_to_win_prob(home_rpg*0.48,away_rpg*0.48)+prop_adj*0.4+ump_b*0.3))
    fair_home_ml=to_ml(fair_home_p); fair_away_ml=to_ml(fair_away_p)

    home_full=TEAM_FULL.get(home_abbr,home_name); away_full=TEAM_FULL.get(away_abbr,away_name)
    key=f"{home_full}|{away_full}"
    game_lines=(odds_map.get(key) or odds_map.get(f"{home_name}|{away_name}") or {})
    home_impls=[]; away_impls=[]
    for bk,lines in game_lines.items():
        hml=lines.get("home_ml"); aml=lines.get("away_ml")
        if hml and aml:
            dh,da=devig(hml,aml); home_impls.append(dh); away_impls.append(da)
    mkt_home_p=round(sum(home_impls)/len(home_impls),4) if home_impls else None
    mkt_away_p=round(sum(away_impls)/len(away_impls),4) if away_impls else None

    best_bet=None; ev_pct=0.0; kstake="0%"; bet_side=None; f5_bet=None
    best_book_name=None; best_book_ml=None; all_book_lines={}
    if mkt_home_p:
        home_ev=fair_home_p-mkt_home_p; away_ev=fair_away_p-mkt_away_p
        f5_hev=f5_home_p-mkt_home_p; f5_aev=(1-f5_home_p)-mkt_away_p
        if home_ev>=away_ev and home_ev>=EV_MIN and fair_home_p>=WIN_PROB_MIN:
            bet_side="home"; best_book_name,best_book_ml,all_book_lines=best_book_for_side(game_lines,"home",home_full,away_full)
            use_ml=best_book_ml or to_ml(mkt_home_p)
            best_bet=f"{home_abbr} ML {use_ml:+d} ({best_book_name or 'N/A'})"
            ev_pct=round(home_ev*100,1); kstake=kelly(home_ev,fair_home_p,use_ml)
        elif away_ev>=EV_MIN and fair_away_p>=WIN_PROB_MIN:
            bet_side="away"; best_book_name,best_book_ml,all_book_lines=best_book_for_side(game_lines,"away",home_full,away_full)
            use_ml=best_book_ml or to_ml(mkt_away_p)
            best_bet=f"{away_abbr} ML {use_ml:+d} ({best_book_name or 'N/A'})"
            ev_pct=round(away_ev*100,1); kstake=kelly(away_ev,fair_away_p,use_ml)
        if bet_side=="home" and f5_hev>=EV_MIN and f5_hev>home_ev+0.02:
            f5_bet=f"F5: {home_abbr} — edge stronger first 5"
        elif bet_side=="away" and f5_aev>=EV_MIN and f5_aev>away_ev+0.02:
            f5_bet=f"F5: {away_abbr} — edge stronger first 5"

    signals=[]
    if not lineup_confirmed: signals.append("LINEUP_UNCONFIRMED")
    if lineup_confirmed: signals.append("LINEUPS_CONFIRMED")
    if h_bull["tax"]>0.06: signals.append(f"BULL_FATIGUE_{home_abbr}")
    if a_bull["tax"]>0.06: signals.append(f"BULL_FATIGUE_{away_abbr}")
    if abs(ump_b)>0.04: signals.append(f"UMP_BIAS_{ump_b:+.2f}_{ump_name}")
    if park_fac>1.10: signals.append(f"HIGH_PARK_{park_fac:.2f}x")
    if park_fac<0.93: signals.append("PITCHERS_PARK")
    if weather["wind_mph"]>10: signals.append(f"WIND_{weather['wind_mph']}mph_{weather['wind_dir']}")
    if weather["temp_f"]>85: signals.append("HOT_WEATHER")
    if weather["condition"] in("Rain","Thunderstorm"): signals.append("RAIN_RISK")
    if hsp["provisional"]: signals.append(f"SP_{home_abbr}_NO_DATA")
    if asp["provisional"]: signals.append(f"SP_{away_abbr}_NO_DATA")
    if not game_lines: signals.append("NO_ODDS_AVAILABLE")

    if ev_pct>=8 and best_bet: tier="TOP"
    elif ev_pct>=6 and best_bet: tier="GOOD"
    elif ev_pct>=3 and best_bet: tier="WATCH"
    else: tier="SKIP"

    return {
        "game_metadata":{"game_id":game_id,"game_pk":game_pk,"home":home_abbr,"away":away_abbr,
            "home_full":home_name,"away_full":away_name,"home_sp":sp_home_name,"away_sp":sp_away_name,
            "start_time":game_time,"venue":venue,"park_factor":round(park_fac,3),
            "weather":weather,"umpire":ump_name,"lineup_confirmed":lineup_confirmed,
            "status":"scheduled","score":"Scheduled","result":None},
        "projections":{"fair_home_ml":fair_home_ml,"fair_away_ml":fair_away_ml,
            "fair_home_p":round(fair_home_p,3),"fair_away_p":round(fair_away_p,3),
            "fair_f5_p":round(f5_home_p,3),"proj_total":proj_total,
            "home_proj_runs":home_rpg,"away_proj_runs":away_rpg,
            "home_woba":h_bat["woba"],"away_woba":a_bat["woba"],
            "sp_home_era":hsp["era"],"sp_away_era":asp["era"],
            "sp_home_fip":hsp["fip"],"sp_away_fip":asp["fip"],
            "sp_home_k9":hsp["k9"],"sp_away_k9":asp["k9"],
            "sp_home_bb9":hsp["bb9"],"sp_away_bb9":asp["bb9"],
            "sp_home_whip":hsp["whip"],"sp_away_whip":asp["whip"],
            "sp_home_ip":hsp["ip"],"sp_away_ip":asp["ip"],
            "sp_home_source":hsp["source"],"sp_away_source":asp["source"],
            "h_bat_source":h_bat["source"],"a_bat_source":a_bat["source"],
            "h_bull_tax":h_bull["tax"],"a_bull_tax":a_bull["tax"],
            "h_bull_detail":h_bull["detail"],"a_bull_detail":a_bull["detail"],
            "h_home_wr":round(h_home_wr,3),"a_away_wr":round(a_away_wr,3),
            "ump_bias":round(ump_b,4),"mkt_home_p":mkt_home_p,"mkt_away_p":mkt_away_p,
            "book_lines":all_book_lines,"best_book":best_book_name,"best_book_ml":best_book_ml},
        "market_edge":{"best_bet":best_bet or "NO EDGE","f5_bet":f5_bet,
            "ev_percent":ev_pct,"kelly_stake":kstake,"tier":tier,"bet_side":bet_side,"signals":signals},
        "math":{"base_pyth_p":round(base_p,4),"prop_adj":round(prop_adj,4),"ump_adj":round(ump_b,4),
            "h_fatigue_tax":round(h_bull["tax"],3),"a_fatigue_tax":round(a_bull["tax"],3),
            "final_home_p":round(fair_home_p,4),"mkt_home_p":mkt_home_p,
            "h_home_wr":round(h_home_wr,3),"a_away_wr":round(a_away_wr,3),
            "park_factor":round(park_fac,3),"ump_bias":round(ump_b,4),
            "sp_suppress_h":round(h_sup,4),"sp_suppress_a":round(a_sup,4)}
    }

# ── FIRST INNING MODULE ───────────────────────────────────────────────────────
def get_team_f1_stats(team_id, n_games=20):
    if not team_id:
        return {"scored_pct":0.40,"allowed_pct":0.40,"recent_scored":[],"recent_allowed":[],"provisional":True}
    today=datetime.date.today()
    start=(today-datetime.timedelta(days=45)).strftime("%Y-%m-%d")
    data=mlb("/schedule",{"sportId":1,"teamId":team_id,"startDate":start,"endDate":today.strftime("%Y-%m-%d"),"hydrate":"linescore,team"})
    scored_f1=[]; allowed_f1=[]
    for db in data.get("dates",[]):
        for g in db.get("games",[]):
            if g.get("status",{}).get("abstractGameState")!="Final": continue
            innings=g.get("linescore",{}).get("innings",[])
            if not innings: continue
            inn1=innings[0]; home_id=g.get("teams",{}).get("home",{}).get("team",{}).get("id")
            home_r1=inn1.get("home",{}).get("runs"); away_r1=inn1.get("away",{}).get("runs")
            if home_r1 is None or away_r1 is None: continue
            if home_id==team_id:
                scored_f1.append(1 if int(home_r1)>0 else 0)
                allowed_f1.append(1 if int(away_r1)>0 else 0)
            else:
                scored_f1.append(1 if int(away_r1)>0 else 0)
                allowed_f1.append(1 if int(home_r1)>0 else 0)
    if len(scored_f1)<5:
        return {"scored_pct":0.40,"allowed_pct":0.40,"recent_scored":[],"recent_allowed":[],"provisional":True}
    s=scored_f1[-n_games:]; a=allowed_f1[-n_games:]
    return {"scored_pct":round(sum(s)/len(s),3),"allowed_pct":round(sum(a)/len(a),3),
            "recent_scored":scored_f1[-5:],"recent_allowed":allowed_f1[-5:],"games":len(s),"provisional":False}

def get_sp_f1_rate(pitcher_id):
    if not pitcher_id:
        return {"f1_ra_pct":0.40,"starts":0,"provisional":True}
    data=mlb(f"/people/{pitcher_id}/stats",{"stats":"season","group":"pitching","season":2026,"sportId":1})
    splits=data.get("stats",[{}])[0].get("splits",[])
    if not splits: return {"f1_ra_pct":0.40,"starts":0,"provisional":True}
    s=splits[0].get("stat",{})
    try:
        era=float(s.get("era",4.50) or 4.50); gs=int(s.get("gamesStarted",0) or 0)
    except: era,gs=4.50,0
    if gs<3: return {"f1_ra_pct":0.40,"starts":gs,"provisional":True}
    # Calibrated: ERA 1.50→18%, ERA 3.00→29%, ERA 5.00→43%, ERA 7.00→58%
    f1_rate=max(0.10,min(0.70,0.18+(era-1.50)*0.070))
    return {"f1_ra_pct":round(f1_rate,3),"era":round(era,2),"starts":gs,"provisional":False}

def get_top_lineup_ops(lineup_ids, n=3):
    if not lineup_ids or len(lineup_ids)<2:
        return {"avg_ops":0.720,"avg_obp":0.320,"provisional":True}
    ops_list=[]; obp_list=[]
    for pid in lineup_ids[:n]:
        data=mlb(f"/people/{pid}/stats",{"stats":"season","group":"hitting","season":2026,"sportId":1})
        splits=data.get("stats",[{}])[0].get("splits",[])
        if splits:
            s=splits[0].get("stat",{})
            try:
                ops=float(s.get("ops",0) or 0); obp=float(s.get("obp",0) or 0)
                if ops>0: ops_list.append(ops)
                if obp>0: obp_list.append(obp)
            except: pass
    if not ops_list: return {"avg_ops":0.720,"avg_obp":0.320,"provisional":True}
    return {"avg_ops":round(sum(ops_list)/len(ops_list),3),"avg_obp":round(sum(obp_list)/len(obp_list),3) if obp_list else 0.320,"batters":len(ops_list),"provisional":False}

def get_kalshi_f1_line(away_abbr, home_abbr):
    try:
        r=get("https://api.elections.kalshi.com/trade-api/v2/markets",{"limit":200,"status":"open"},timeout=8)
        if not r: return None,None,"Kalshi unavailable"
        markets=r.json().get("markets",[])
        for m in markets:
            ticker=m.get("ticker","").upper(); title=m.get("title","").upper()
            if (away_abbr in ticker or home_abbr in ticker or away_abbr in title or home_abbr in title):
                if "NRFI" in ticker or "NRFI" in title or "FIRST INNING" in title:
                    yes_p=m.get("yes_ask") or m.get("last_price"); no_p=m.get("no_ask")
                    return yes_p,no_p,m.get("title","Kalshi market")
        return None,None,"No Kalshi market found"
    except Exception as e:
        return None,None,f"Kalshi error: {str(e)[:40]}"

def predict_first_inning(game, date_str):
    home_d=game.get("teams",{}).get("home",{}); away_d=game.get("teams",{}).get("away",{})
    home_abbr=home_d.get("team",{}).get("abbreviation","HM"); away_abbr=away_d.get("team",{}).get("abbreviation","AW")
    home_name=home_d.get("team",{}).get("name","Home"); away_name=away_d.get("team",{}).get("name","Away")
    home_id=home_d.get("team",{}).get("id"); away_id=away_d.get("team",{}).get("id")
    game_pk=str(game.get("gamePk","")); game_time=game.get("gameDate","")
    sp_home=home_d.get("probablePitcher",{}); sp_away=away_d.get("probablePitcher",{})
    sp_home_id=sp_home.get("id"); sp_away_id=sp_away.get("id")
    sp_home_name=sp_home.get("fullName","TBD"); sp_away_name=sp_away.get("fullName","TBD")
    print(f"  F1: {away_abbr}@{home_abbr}")

    h_f1=get_team_f1_stats(home_id); a_f1=get_team_f1_stats(away_id)
    hsp=get_sp_f1_rate(sp_home_id); asp=get_sp_f1_rate(sp_away_id)
    h_lineup,a_lineup=get_confirmed_lineup(game)
    h_hit=get_top_lineup_ops(h_lineup); a_hit=get_top_lineup_ops(a_lineup)

    def recency(arr):
        if not arr: return 0.40
        w=[1,1.5,2,2.5,3]; wt=sum(w[-len(arr):]); ws=sum(w[-len(arr):][i]*v for i,v in enumerate(arr))
        return ws/wt if wt>0 else 0.40

    a_rec=recency(a_f1["recent_scored"]); h_rec=recency(h_f1["recent_scored"])
    a_lb=(a_hit["avg_ops"]-0.720)*0.12; h_lb=(h_hit["avg_ops"]-0.720)*0.12

    # Away scores top 1st (vs home SP)
    away_p=0.40*a_f1["scored_pct"]+0.35*hsp["f1_ra_pct"]+0.15*a_rec+0.10*max(0,min(1,0.40+a_lb))
    # Home scores bottom 1st (vs away SP)
    home_p=0.40*h_f1["scored_pct"]+0.35*asp["f1_ra_pct"]+0.15*h_rec+0.10*max(0,min(1,0.40+h_lb))

    yrfi_p=round(max(0.15,min(0.90,1-(1-away_p)*(1-home_p))),3)
    nrfi_p=round(1-yrfi_p,3)

    # NRFI guard: if picking NRFI, both pitchers must have <50% F1 RA rate
    nrfi_disq=nrfi_p>0.60 and (hsp["f1_ra_pct"]>0.50 or asp["f1_ra_pct"]>0.50)

    prediction=None; confidence=None; pick_label=None
    if nrfi_p>=0.62 and not nrfi_disq:  # 0.62 required — 0.68 was too strict, even elite SPs couldn't reach it
        prediction="NRFI"; confidence=round(nrfi_p*100,1); pick_label=f"NRFI — {away_abbr}@{home_abbr}"
    elif yrfi_p>=0.68:  # YRFI threshold unchanged
        prediction="YRFI"; confidence=round(yrfi_p*100,1); pick_label=f"YRFI — {away_abbr}@{home_abbr}"

    kalshi_yes,kalshi_no,kalshi_note=get_kalshi_f1_line(away_abbr,home_abbr)

    signals=[]
    if a_f1["provisional"]: signals.append(f"{away_abbr}_LIMITED_F1_DATA")
    if h_f1["provisional"]: signals.append(f"{home_abbr}_LIMITED_F1_DATA")
    if hsp.get("provisional"): signals.append(f"{sp_home_name.split()[-1]}_ERA_ESTIMATED")
    if asp.get("provisional"): signals.append(f"{sp_away_name.split()[-1]}_ERA_ESTIMATED")
    if nrfi_disq: signals.append("NRFI_BLOCKED_WEAK_SP")
    if len(h_lineup)<8: signals.append("LINEUP_UNCONFIRMED")
    if hsp["f1_ra_pct"]<0.22: signals.append(f"{sp_home_name.split()[-1]}_DOMINATES_F1")
    if asp["f1_ra_pct"]<0.22: signals.append(f"{sp_away_name.split()[-1]}_DOMINATES_F1")
    if hsp["f1_ra_pct"]>0.58: signals.append(f"{sp_home_name.split()[-1]}_STRUGGLES_F1")
    if asp["f1_ra_pct"]>0.58: signals.append(f"{sp_away_name.split()[-1]}_STRUGGLES_F1")

    return {"game_id":f"{date_str}-F1-{away_abbr}-{home_abbr}","game_pk":game_pk,
            "home":home_abbr,"away":away_abbr,"home_full":home_name,"away_full":away_name,
            "home_sp":sp_home_name,"away_sp":sp_away_name,"start_time":game_time,
            "status":"scheduled","score":"Scheduled","result":None,
            "prediction":prediction,"pick_label":pick_label,"confidence":confidence,
            "yrfi_p":yrfi_p,"nrfi_p":nrfi_p,"away_score_p":round(away_p,3),"home_score_p":round(home_p,3),
            "kalshi_yes":kalshi_yes,"kalshi_no":kalshi_no,"kalshi_note":kalshi_note,
            "data":{"away_f1_score_pct":a_f1["scored_pct"],"away_f1_allow_pct":a_f1["allowed_pct"],
                    "away_recent_f1":a_f1["recent_scored"],"home_f1_score_pct":h_f1["scored_pct"],
                    "home_f1_allow_pct":h_f1["allowed_pct"],"home_recent_f1":h_f1["recent_scored"],
                    "hsp_f1_ra_pct":hsp["f1_ra_pct"],"hsp_era":hsp.get("era","N/A"),
                    "asp_f1_ra_pct":asp["f1_ra_pct"],"asp_era":asp.get("era","N/A"),
                    "away_top3_ops":a_hit["avg_ops"],"home_top3_ops":h_hit["avg_ops"]},
            "signals":signals,"nrfi_disqualified":nrfi_disq}

def run_f1_predictions(games, date_str):
    print("\n[F1] Running first inning predictions...")
    results=[]
    for game in games:
        try:
            r=predict_first_inning(game,date_str)
            if r: results.append(r)
        except Exception as e:
            print(f"  [F1 ERROR] {e}")
    confident=[p for p in results if p["prediction"]]
    print(f"  {len(results)} games | {len(confident)} confident F1 picks")
    return results

# ── RESULTS RUN ───────────────────────────────────────────────────────────────
def run_results(date_str, existing_data):
    print(f"\n[Results] Fetching final scores for {date_str}...")
    scores=get_final_scores(date_str)
    print(f"  {len(scores)} games final")
    picks=existing_data.get("picks",[])
    updated=0
    for pick in picks:
        gm=pick.get("game_metadata",{}); me=pick.get("market_edge",{})
        game_id=gm.get("game_id","")
        if gm.get("result"): continue
        if me.get("tier")=="SKIP" or not me.get("bet_side"): continue
        score=scores.get(game_id)
        if not score: continue
        home_won=score["home_score"]>score["away_score"]
        we_won=(me["bet_side"]=="home" and home_won) or (me["bet_side"]=="away" and not home_won)
        gm["status"]="closed"; gm["score"]=score["score_display"]+" — FINAL"; gm["result"]="W" if we_won else "L"
        updated+=1
    # Mark F1 results
    f1_picks=existing_data.get("f1_picks",[])
    for fp in f1_picks:
        if fp.get("result") or not fp.get("prediction"): continue
        parts=fp["game_id"].split("-")
        if len(parts)<5: continue
        away_a=parts[3]; home_a=parts[4]
        score_key=f"{date_str}-{away_a}-{home_a}"
        score=scores.get(score_key)
        if not score: continue
        f1h=score.get("f1_home"); f1a=score.get("f1_away")
        fp["score"]=score["score_display"]+" — FINAL"; fp["status"]="closed"
        if f1h is not None and f1a is not None:
            actual_yrfi=(int(f1h)+int(f1a))>0
            fp["f1_score"]=f"F1: {away_a} {f1a} · {home_a} {f1h}"
            correct=(fp["prediction"]=="YRFI")==actual_yrfi
            fp["result"]="W" if correct else "L"
    print(f"  Marked {updated} game results")
    return picks, f1_picks

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    date_str=sys.argv[1] if len(sys.argv)>1 else datetime.date.today().strftime("%Y-%m-%d")
    run_type=sys.argv[2] if len(sys.argv)>2 else "full"
    print(f"\n{'='*55}")
    print(f"  DiamondEdge Engine v4 | {date_str} | {run_type}")
    print(f"  Data: MLB Stats API (official) + Odds API + OpenWeatherMap")
    print(f"  FIP and wOBA calculated from raw counting stats (no scraping)")
    print(f"  Threshold: EV>={EV_MIN*100:.0f}% AND win_prob>={WIN_PROB_MIN*100:.0f}%")
    print(f"{'='*55}")

    existing_data={}
    if OUT_PATH.exists():
        try:
            with open(OUT_PATH) as f:
                existing_data=json.load(f)
                if existing_data.get("date")!=date_str: existing_data={}
        except: pass

    if run_type=="results":
        if not existing_data: print("No existing data to update"); return
        picks,f1_picks=run_results(date_str,existing_data)
        existing_data["picks"]=picks; existing_data["f1_picks"]=f1_picks
        existing_data["last_results_check"]=datetime.datetime.utcnow().isoformat()+"Z"
        with open(OUT_PATH,"w") as f: json.dump(existing_data,f,indent=2)
        print("✅ Results updated"); return

    print("\n[1/4] Fetching schedule...")
    games=get_schedule(date_str); print(f"  {len(games)} games")

    print("\n[2/4] Fetching odds...")
    odds_map,missing_books=get_odds()

    print("\n[3/4] Running game model...")
    picks=[]
    for game in games:
        try: picks.append(run_model(game,odds_map,date_str))
        except Exception as e: import traceback; print(f"  [ERROR] {e}"); traceback.print_exc()

    # Preserve existing results
    if existing_data.get("picks"):
        rmap={p["game_metadata"]["game_id"]:p["game_metadata"].get("result") for p in existing_data["picks"] if p["game_metadata"].get("result")}
        for pick in picks:
            gid=pick["game_metadata"]["game_id"]
            if gid in rmap: pick["game_metadata"]["result"]=rmap[gid]

    to={"TOP":0,"GOOD":1,"WATCH":2,"SKIP":3}
    picks.sort(key=lambda x:(to.get(x["market_edge"]["tier"],3),-x["market_edge"]["ev_percent"]))

    print("\n[4/4] Running first inning predictions...")
    f1_picks=run_f1_predictions(games,date_str)

    # Preserve existing F1 results
    if existing_data.get("f1_picks"):
        f1rmap={fp["game_id"]:fp.get("result") for fp in existing_data["f1_picks"] if fp.get("result")}
        for fp in f1_picks:
            if fp["game_id"] in f1rmap: fp["result"]=f1rmap[fp["game_id"]]

    # Check for any already completed games
    picks_final,f1_final=run_results(date_str,{"picks":picks,"f1_picks":f1_picks})

    ev_picks=[p for p in picks_final if p["market_edge"]["tier"] in("TOP","GOOD")]
    output={"date":date_str,"run_type":run_type,
            "generated":datetime.datetime.utcnow().isoformat()+"Z",
            "game_count":len(picks_final),"ev_pick_count":len(ev_picks),
            "f1_confident_count":len([p for p in f1_final if p["prediction"]]),
            "missing_books":missing_books,
            "model":{"version":"v4.0",
                "data_sources":["MLB Stats API — official 2026 season stats",
                    "FIP calculated from HR/BB/HBP/K/IP","wOBA from counting stats",
                    "FanDuel + BetMGM via The Odds API","OpenWeatherMap","Retrosheet 33k game lookup"],
                "thresholds":{"ev_min":EV_MIN,"win_prob_min":WIN_PROB_MIN},
                "train_games":LOOKUP.get("meta",{}).get("train_games",33292)},
            "picks":picks_final,"f1_picks":f1_final}

    with open(OUT_PATH,"w") as f: json.dump(output,f,indent=2)
    print(f"\n✅ {len(picks_final)} games | {len(ev_picks)} value picks | {output['f1_confident_count']} F1 picks")
    for p in ev_picks:
        me=p["market_edge"]; gm=p["game_metadata"]
        print(f"  [{me['tier']}] {gm['away']}@{gm['home']} | {me['best_bet']} | EV={me['ev_percent']}%")

if __name__=="__main__":
    main()
