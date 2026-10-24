from datetime import datetime, timezone
from uuid import uuid4
# pyrefly: ignore[missing-import]
from uagents import Agent, Context, Protocol
# pyrefly: ignore[missing-import]
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

agent = Agent()
protocol = Protocol(spec=chat_protocol_spec)

import math
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

# ── Geocode cache: avoid re-querying the same city in one request ─────────────
_GEO_CACHE = {}

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
WIKI_GEO_URL = "https://en.wikipedia.org/w/api.php"   # GeoSearch for reliable attractions
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
USER_AGENT = "SmartTripPlannerAgent/2.0 (student-project; shaurya-planner@agentverse.ai)"

WMO = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

VERB_WORDS = {
    "eat", "stay", "visit", "travel", "go", "reach", "get", "see",
    "plan", "find", "book", "do", "help", "explore", "know", "check",
    "sleep", "hire", "drive", "fly", "walk",
}

NON_PLACE_WORDS = {
    "hi", "hello", "hey", "help", "trip", "weekend", "holiday", "vacation",
    "weather", "forecast", "beach", "beaches", "fort", "forts", "place",
    "places", "hotel", "hotels", "food", "restaurants", "restaurant",
    "budget", "day", "days", "me", "you", "my", "us", "it", "here", "there",
    "things", "attractions", "attraction", "sights", "sightseeing", "spots",
}


# ---------- Geometry & Math helpers ----------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         (math.sin(dlon / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(max(0.0, a)), math.sqrt(max(0.0, 1.0 - a)))
    return r * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def bearing_to_compass(brng):
    points = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((brng + 11.25) / 22.5) % 16
    return points[idx]


def bearing_to_direction_name(brng):
    names = [
        "north", "north-east", "north-east", "east", "east", "south-east",
        "south-east", "south", "south", "south-west", "south-west", "west",
        "west", "north-west", "north-west", "north"
    ]
    idx = int((brng + 11.25) / 22.5) % 16
    return names[idx]


# ---------- Parsing functions ----------
def _clean_candidate(cand):
    if not cand:
        return None
    s = cand.strip(" \t\n\r,.:;?!\"'()")
    s = re.sub(r"\s+", " ", s)
    # Repeatedly strip leading action verbs, prepositions & articles
    while True:
        m_verb = re.match(
            r"^(?:reach|visit|visiting|get\s+to|go\s+to|travel\s+to|trip\s+to|plan\s+to|explore|exploring|tour|touring|see|enjoy|experience|get|for|in|at|around|near|to)\s+(.+)$",
            s,
            re.I,
        )
        if m_verb:
            s = m_verb.group(1).strip(" \t\n\r,.:;?!\"'()")
            continue
        m_art = re.match(r"^(?:the|a|an)\s+(.+)$", s, re.I)
        if m_art:
            s = m_art.group(1).strip(" \t\n\r,.:;?!\"'()")
            continue
        break

    if not s or len(s) < 2:
        return None
    low = s.lower()
    if low in NON_PLACE_WORDS or low in VERB_WORDS:
        return None
    # Reject if >=4 words — real place names are ≤3 words; more means we caught filler
    if len(s.split()) >= 4:
        return None
    return s


def find_places(text):
    t = text.strip()
    if not t:
        return None, None
    if t.lower() in NON_PLACE_WORDS:
        return None, None

    # Stop lookahead: these words immediately after a place name indicate the name ended
    stop = (
        r"(?=\s+(?:"
        r"under|for|with|within|budget|at|on|next|this|and|"
        r"suggest|multiple|transportation|availability|available|"
        r"tell|want|need|show|give|please|help|by|via|using|"
        r"how|what|where|when|which|who|can|will|should|is|are|does|"
        r"transport|mode|option|options|route|routes|way|ways|flight|train|bus"
        r")\b|"
        r"[,.?!:;]|\s*(?:₹|rs\.?\b|inr\b)|$)"
    )

    # 1. "from {origin} to {dest}"
    m = re.search(r"\bfrom\s+([A-Za-z][A-Za-z\s.'-]*?)\s+to\s+([A-Za-z][A-Za-z\s.'-]*?)" + stop, t, re.I)
    if m:
        orig = _clean_candidate(m.group(1))
        dest = _clean_candidate(m.group(2))
        if orig and dest:
            return orig, dest

    # 2. "to {dest} from {origin}" or "reach {dest} from {origin}"
    m = re.search(r"\b(?:how\s+to\s+reach|how\s+to\s+get\s+to|to|reach|visiting|visit|trip\s+to)\s+([A-Za-z][A-Za-z\s.'-]*?)\s+from\s+([A-Za-z][A-Za-z\s.'-]*?)" + stop, t, re.I)
    if m:
        dest = _clean_candidate(m.group(1))
        orig = _clean_candidate(m.group(2))
        if orig and dest:
            return orig, dest

    # 2b. "I want to go to {dest} from {origin} ..." — natural language with filler before
    m = re.search(r"\bgo\s+to\s+([A-Za-z][A-Za-z\s.'-]*?)\s+from\s+([A-Za-z][A-Za-z\s.'-]*?)" + stop, t, re.I)
    if m:
        dest = _clean_candidate(m.group(1))
        orig = _clean_candidate(m.group(2))
        if orig and dest:
            return orig, dest

    # 3. "{origin} to {dest} transport"
    m = re.search(r"\b([A-Za-z][A-Za-z\s.'-]*?)\s+to\s+([A-Za-z][A-Za-z\s.'-]*?)\s+(?:transport|route|flight|train|bus)\b", t, re.I)
    if m:
        orig = _clean_candidate(m.group(1))
        dest = _clean_candidate(m.group(2))
        if orig and dest:
            return orig, dest

    # 4. Single destination with "in", "at", "around", "near" (e.g. "what to eat in Goa", "Best beaches in Goa India")
    m = re.search(r"\b(?:in|at|around|near)\s+([A-Za-z][A-Za-z\s.'-]*?)" + stop, t, re.I)
    if m:
        dest = _clean_candidate(m.group(1))
        if dest:
            return None, dest

    # 5. Single destination with "explore", "visit", "trip to", "for", "to"
    m = re.search(r"\b(?:trip\s+to|travel\s+to|going\s+to|go\s+to|visit|visiting|explore|exploring|tour|touring|for|to)\s+([A-Za-z][A-Za-z\s.'-]*?)" + stop, t, re.I)
    if m:
        dest = _clean_candidate(m.group(1))
        if dest:
            return None, dest

    return None, None


def parse_days_budget(text):
    t = text.strip()
    days = 2
    m = re.search(r"(\d+)\s*[- ]?\s*days?", t, re.I)
    if m:
        try:
            days = int(m.group(1))
        except ValueError:
            days = 2
    elif re.search(r"\bweekend\b", t, re.I):
        days = 2
    days = max(1, min(days, 7))

    budget = None
    m = re.search(
        r"(?:under|below|within|budget(?:\s+of)?|upto|up to|max)?\s*"
        r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(k)?",
        t, re.I,
    )
    if not m:
        m = re.search(
            r"(?:under|below|within|budget(?:\s+of)?)\s*([\d,]+(?:\.\d+)?)\s*(k)?",
            t, re.I,
        )
    if m:
        val_str = m.group(1).replace(",", "")
        try:
            val = float(val_str)
            if m.group(2):
                val *= 1000.0
            budget = int(val)
        except ValueError:
            budget = None

    return days, budget


# ---------- Network & Geocoding helpers ----------
def _http_get_json(url, params=None, headers=None, timeout=10, retries=2):
    last_err = None
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=hdrs, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(str(last_err))


def _http_post_json(url, data=None, headers=None, timeout=25, retries=2):
    last_err = None
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    for attempt in range(retries):
        try:
            r = requests.post(url, data=data, headers=hdrs, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(str(last_err))


def geocode_place(query, prefer_india=False):
    """Geocode a place name to coordinates (cached per process).

    Strategy (no hardcoded city lists):
    1. Search globally (count=10), pick highest-population result.
    2. ALSO search India (count=5), pick highest-population India result.
    3. Compare both by population — whichever has more people wins.
       This automatically resolves ambiguous names:
         "Delhi" → India (11M) beats Ohio (500)   → India wins
         "Paris" → France (2M) beats Texas (25k)  → France wins
         "Goa"   → India state beats any namesake → India wins
    4. Nominatim fallback if open-meteo returns nothing.
    """
    q = query.strip()
    if not q:
        return None
    cache_key = (q.lower(), prefer_india)
    if cache_key in _GEO_CACHE:
        return _GEO_CACHE[cache_key]

    def _search(name, **extra):
        """Call open-meteo geocoding API and return raw results list."""
        params = {"name": name, "count": 10, "language": "en"}
        params.update(extra)
        try:
            res = _http_get_json(GEO_URL, params=params, timeout=8, retries=2)
            return res.get("results") or []
        except Exception:
            return []

    def _best(results, name_q):
        """From a result list, pick the highest-population entry whose name matches."""
        if not results:
            return None
        # Case-insensitive exact name match preferred; fall back to all results
        qn = name_q.lower().strip()
        exact = [r for r in results if r.get("name", "").strip().lower() == qn]
        pool = exact if exact else results
        # Always pick the most populous — largest city wins ties automatically
        return max(pool, key=lambda r: r.get("population") or 0)

    def _make_result(top, fallback_name):
        return {
            "name": top.get("name") or fallback_name.title(),
            "latitude": float(top["latitude"]),
            "longitude": float(top["longitude"]),
            "admin1": top.get("admin1") or "",
            "country": top.get("country") or "",
            "country_code": (top.get("country_code") or "").upper(),
            "population": top.get("population") or 0,
        }

    # ── Steps 1+2: Global + India search IN PARALLEL ─────────────────────────
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_global = ex.submit(_search, q)
        f_india  = ex.submit(_search, q, countryCode="IN")
    global_top = _best(f_global.result(), q)
    india_top  = _best(f_india.result(),  q)

    # ── Step 3: Pick winner by population ─────────────────────────────────────
    g_pop = (global_top.get("population") or 0) if global_top else -1
    i_pop = (india_top.get("population") or 0) if india_top else -1

    if prefer_india and india_top:
        # Explicit India preference (e.g. from context): India wins unless global
        # has massively higher population (> 10x), meaning it's clearly not India.
        top = india_top if i_pop * 10 >= g_pop else global_top
    elif india_top and global_top:
        # No explicit preference: highest population wins automatically
        top = india_top if i_pop > g_pop else global_top
    elif india_top:
        top = india_top
    elif global_top:
        top = global_top
    else:
        top = None

    if top:
        result = _make_result(top, q)
        _GEO_CACHE[cache_key] = result
        return result

    # ── Step 4: Nominatim fallback ────────────────────────────────────────────
    try:
        nom_params = {"q": q, "format": "jsonv2", "addressdetails": 1, "limit": 1}
        nom_data = _http_get_json(NOMINATIM_URL, params=nom_params, timeout=8, retries=1)
        if nom_data:
            item = nom_data[0]
            addr = item.get("address") or {}
            place_name = (
                item.get("name") or addr.get("city") or addr.get("town") or
                addr.get("state") or q.title()
            )
            return {
                "name": place_name,
                "latitude": float(item["lat"]),
                "longitude": float(item["lon"]),
                "admin1": addr.get("state") or addr.get("region") or addr.get("county") or "",
                "country": addr.get("country") or "",
                "country_code": (addr.get("country_code") or "").upper(),
                "population": 0,
            }
    except Exception:
        pass

    return None
# ---------- Transport computation ----------
def get_transport_info(origin, dest):
    lat1, lon1 = origin["latitude"], origin["longitude"]
    lat2, lon2 = dest["latitude"], dest["longitude"]
    straight_km = haversine_km(lat1, lon1, lat2, lon2)

    osrm_url = f"{OSRM_URL}/{lon1},{lat1};{lon2},{lat2}?overview=false"
    road_km = None
    drive_sec = None
    is_osrm = False

    try:
        data = _http_get_json(osrm_url, timeout=10, retries=2)
        routes = data.get("routes") or []
        if routes:
            road_km = (routes[0].get("distance") or 0.0) / 1000.0
            drive_sec = routes[0].get("duration") or 0.0
            if road_km > 0:
                is_osrm = True
    except Exception:
        pass

    if road_km is None or road_km <= 0:
        road_km = straight_km * 1.3
        drive_sec = (road_km / 65.0) * 3600.0
        is_osrm = False

    drive_hrs = drive_sec / 3600.0
    drive_h = int(drive_hrs)
    drive_m = int((drive_hrs - drive_h) * 60)
    drive_time_str = f"{drive_h} hr {drive_m} min" if drive_h > 0 else f"{drive_m} min"

    both_in_india = (origin.get("country_code") == "IN" and dest.get("country_code") == "IN")
    diff_countries = (origin.get("country_code") and dest.get("country_code") and
                      origin["country_code"] != dest["country_code"])

    options = []
    # 1. Flight
    if straight_km >= 180:
        flight_air_hrs = (straight_km / 650.0) + 2.5
        fh = int(flight_air_hrs)
        fm = int((flight_air_hrs - fh) * 60)
        ftime_str = f"{fh} hr {fm} min (incl. airport buffer)"
        if both_in_india:
            if straight_km < 600:
                f_cost = (2800, 5000)
            elif straight_km < 1200:
                f_cost = (3800, 7200)
            else:
                f_cost = (4800, 9500)
        else:
            f_cost = None
        options.append({
            "mode": "Flight",
            "time": ftime_str,
            "cost_range": f_cost,
            "is_public": True,
            "sort_time": flight_air_hrs,
            "min_cost": f_cost[0] if f_cost else 999999,
        })

    # 2. Train
    train_km = straight_km * 1.15
    train_hrs = (train_km / 55.0) + 0.5
    th = int(train_hrs)
    tm = int((train_hrs - th) * 60)
    ttime_str = f"{th} hr {tm} min"
    if both_in_india:
        t_min = max(350, int(train_km * 0.55))
        t_max = max(750, int(train_km * 1.65))
        t_cost = (t_min, t_max)
    else:
        t_cost = None
    options.append({
        "mode": "Train",
        "time": ttime_str,
        "cost_range": t_cost,
        "is_public": True,
        "sort_time": train_hrs,
        "min_cost": t_cost[0] if t_cost else 999999,
    })

    # 3. Bus
    bus_hrs = (road_km / 45.0) + 1.0
    bh = int(bus_hrs)
    bm = int((bus_hrs - bh) * 60)
    btime_str = f"{bh} hr {bm} min"
    if both_in_india:
        b_min = max(400, int(road_km * 0.85))
        b_max = max(800, int(road_km * 1.65))
        b_cost = (b_min, b_max)
    else:
        b_cost = None
    options.append({
        "mode": "Bus",
        "time": btime_str,
        "cost_range": b_cost,
        "is_public": True,
        "sort_time": bus_hrs,
        "min_cost": b_cost[0] if b_cost else 999999,
    })

    # 4. Self-drive
    if both_in_india:
        d_min = int(road_km * 7.5)
        d_max = int(road_km * 10.5)
        d_cost = (d_min, d_max)
    else:
        d_cost = None
    options.append({
        "mode": "Self-drive",
        "time": drive_time_str,
        "cost_range": d_cost,
        "is_public": False,
        "sort_time": drive_hrs,
        "min_cost": d_cost[0] if d_cost else 999999,
    })

    fastest = min(options, key=lambda o: o["sort_time"])
    public_opts = [o for o in options if o["is_public"]]
    if both_in_india and public_opts:
        cheapest_public = min(public_opts, key=lambda o: o["min_cost"])
        round_trip_travel = cheapest_public["min_cost"] * 2
    else:
        cheapest_public = min(public_opts, key=lambda o: o["sort_time"]) if public_opts else None
        round_trip_travel = 0

    return {
        "straight_km": straight_km,
        "road_km": road_km,
        "drive_time_str": drive_time_str,
        "is_osrm": is_osrm,
        "options": options,
        "both_in_india": both_in_india,
        "diff_countries": diff_countries,
        "fastest": fastest["mode"],
        "cheapest_public": cheapest_public["mode"] if cheapest_public else "Train",
        "round_trip_travel": round_trip_travel,
    }


# ---------- Weather computation ----------
def get_weather_forecast(place, days):
    lat, lon = place["latitude"], place["longitude"]
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
        "timezone": "auto",
        "forecast_days": days,
    }
    data = _http_get_json(FORECAST_URL, params=params, timeout=10, retries=2)
    daily = data.get("daily") or {}
    times = daily.get("time") or []
    tmaxs = daily.get("temperature_2m_max") or []
    tmins = daily.get("temperature_2m_min") or []
    rains = daily.get("precipitation_probability_max") or []
    codes = daily.get("weathercode") or []

    forecast = []
    for i in range(len(times)):
        wcode = codes[i] if i < len(codes) else 0
        forecast.append({
            "date": times[i],
            "desc": WMO.get(wcode, "Mixed conditions"),
            "tmin": tmins[i] if i < len(tmins) else 20.0,
            "tmax": tmaxs[i] if i < len(tmaxs) else 30.0,
            "rain": rains[i] if (i < len(rains) and rains[i] is not None) else 0,
        })
    return forecast


def compute_packing_tips(forecast):
    if not forecast:
        return []
    max_rain = max(f["rain"] for f in forecast)
    max_tmax = max(f["tmax"] for f in forecast)
    min_tmin = min(f["tmin"] for f in forecast)

    tips = []
    if max_rain >= 60:
        tips.append(f"Umbrella or rain gear (rain chance reaches {max_rain}%)")
    if max_tmax >= 30.0:
        tips.append(f"Light clothing (temperatures up to {max_tmax:.1f}°C)")
    if min_tmin <= 12.0:
        tips.append(f"Warm layer (temperatures down to {min_tmin:.1f}°C)")
    return tips


# ---------- Places, Eat & Stay via Overpass ----------
def _run_overpass_query(query, timeout=10):
    """Run an Overpass query, trying each mirror once with a hard timeout."""
    last_err = None
    for url in OVERPASS_URLS:
        try:
            res = _http_post_json(url, data={"data": query}, timeout=timeout, retries=1)
            elements = res.get("elements") or []
            return elements
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Overpass mirrors failed: {last_err}")


def classify_osm_spot(tags, name):
    low = name.lower()
    tourism = tags.get("tourism") or ""
    historic = tags.get("historic") or ""
    natural = tags.get("natural") or ""
    amenity = tags.get("amenity") or ""
    leisure = tags.get("leisure") or ""
    shop = tags.get("shop") or ""

    if natural == "beach" or tourism == "beach" or "beach" in low:
        return "Beaches"
    if (historic in ("fort", "castle", "palace", "monument", "memorial", "archaeological_site") or
            "fort" in low or "palace" in low or "monument" in low):
        return "Forts & heritage"
    if (tourism in ("viewpoint", "zoo") or leisure in ("park", "nature_reserve") or
            natural in ("waterfall", "peak") or "viewpoint" in low or "waterfall" in low):
        return "Nature & viewpoints"
    if (tourism in ("museum", "gallery", "arts_centre") or "museum" in low or "gallery" in low):
        return "Museums & galleries"
    if (amenity == "place_of_worship" or "temple" in low or "church" in low or
            "mosque" in low or "cathedral" in low or "ashram" in low):
        return "Temples & places of worship"
    if (amenity == "marketplace" or shop == "mall" or "market" in low or "bazaar" in low):
        return "Markets"
    return "Attractions"


def _get_spots_wikipedia(lat, lon, c_name, radius_m=10000, limit=50):
    """
    Fetch notable tourist spots via Wikipedia GeoSearch API.
    Fast, reliable, no rate limits.
    Filters out irrelevant articles (elections, schools, admin boundaries).
    """
    # Keywords that indicate a SKIP (non-tourist article)
    SKIP_WORDS = {
        "election", "constituency", "assembly", "lok sabha", "rajya sabha",
        "municipality", "municipal", "corporation", "ward", "panchayat",
        "railway station", "bus stand", "bus stop", "airport", "metro station", "metro",
        "school", "college", "university", "institute", "academy",
        "hospital", "clinic", "police", "court", "jail",
        "road", "street", "highway", "bridge", "flyover",
        "village", "hamlet", "tehsil", "taluk", "mandal",
        "district", "state", "country", "region", "area",
        "election result", "by-election", "constituency",
        "organization", "association", "society", "trust",
        "dam", "reservoir", "canal", "pipeline",
        "post office", "bank", "market area",
    }
    # Keywords that CONFIRM tourist interest
    TOURIST_WORDS = {
        "fort", "palace", "mahal", "haveli", "gate", "castle", "wada",
        "temple", "mosque", "church", "shrine", "gurudwara", "dargah", "mandir", "ganapati",
        "museum", "gallery", "monument", "memorial", "statue",
        "park", "garden", "lake", "waterfall", "peak", "hill", "valley",
        "forest", "wildlife", "sanctuary", "national park", "reserve",
        "beach", "coast", "bay", "island", "ghat", "stepwell", "baori",
        "cave", "falls", "trekking", "viewpoint", "promenade",
        "market", "bazaar", "chowk",
        "ashram", "math", "stupa", "pagoda",
        "zoo", "botanical", "aquarium",
        "heritage", "archaeological", "ruins", "ancient",
        "resort", "hill station",
    }

    def _is_tourist(title):
        low = title.lower()
        # Hard skip
        if any(w in low for w in SKIP_WORDS):
            return False
        # Strong keep
        if any(w in low for w in TOURIST_WORDS):
            return True
        # If title is short (1-3 words) and none of the skip words match, it might be a place
        word_count = len(title.split())
        if word_count <= 3 and not any(w in low for w in SKIP_WORDS):
            return True
        return False

    def _search(radius):
        params = {
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": min(radius, 10000),
            "gslimit": limit,
            "format": "json",
        }
        try:
            data = _http_get_json(WIKI_GEO_URL, params=params, timeout=8, retries=2)
            return (data.get("query") or {}).get("geosearch") or []
        except Exception:
            return []

    items = _search(min(radius_m, 10000))

    spots = []
    seen = set()
    for item in items:
        name = (item.get("title") or "").strip()
        if not name or name.lower() in seen:
            continue
        if any(x in name for x in ["(disambiguation)", "List of"]):
            continue
        if not _is_tourist(name):
            continue
        seen.add(name.lower())

        s_lat = item.get("lat") or lat
        s_lon = item.get("lon") or lon
        dist_m = item.get("dist") or (haversine_km(lat, lon, s_lat, s_lon) * 1000)
        dist_km = dist_m / 1000.0
        brng = calculate_bearing(lat, lon, s_lat, s_lon)
        comp = bearing_to_compass(brng)
        dname = bearing_to_direction_name(brng)

        low = name.lower()
        if any(w in low for w in ["beach", "coast", "shore", "bay", "island"]):
            cat = "Beaches"
        elif any(w in low for w in ["fort", "palace", "castle", "mahal", "gate", "haveli", "heritage", "ruins", "ancient"]):
            cat = "Forts & heritage"
        elif any(w in low for w in ["temple", "mosque", "church", "shrine", "dargah", "mandir", "masjid", "gurudwara", "ashram", "stupa"]):
            cat = "Temples & places of worship"
        elif any(w in low for w in ["museum", "gallery", "exhibit", "archaeological"]):
            cat = "Museums & galleries"
        elif any(w in low for w in ["lake", "waterfall", "peak", "hill", "valley", "forest", "park", "garden", "reserve", "sanctuary", "cave", "falls"]):
            cat = "Nature & viewpoints"
        elif any(w in low for w in ["market", "bazaar", "chowk"]):
            cat = "Markets"
        else:
            cat = "Attractions"

        spots.append({
            "name": name,
            "lat": s_lat,
            "lon": s_lon,
            "category": cat,
            "score": 3,
            "dist": dist_km,
            "bearing": brng,
            "compass": comp,
            "direction_name": dname,
            "centre": c_name,
            "map_url": f"https://www.google.com/maps/search/?api=1&query={s_lat:.6f},{s_lon:.6f}",
        })

    return spots



def get_destination_spots(place, max_time_cap=18.0):
    """
    Fetch tourist spots — Wikipedia GeoSearch first (fast, reliable),
    then Overpass as supplement if Wikipedia returns too few results.
    """
    lat, lon = place["latitude"], place["longitude"]
    c_name = place["name"]
    seen = set()
    ranked_spots = []

    # ── PRIMARY: Wikipedia GeoSearch (1-2s, always works) ────────────────────
    wiki_spots = _get_spots_wikipedia(lat, lon, c_name, radius_m=10000, limit=30)
    for s in wiki_spots:
        norm = s["name"].strip().lower()
        if norm not in seen:
            seen.add(norm)
            ranked_spots.append(s)

    # ── SUPPLEMENT: Overpass (if Wikipedia gave <8 results) ──────────────────
    if len(ranked_spots) < 8:
        radius = 20000
        query = f"""
        [out:json][timeout:15];
        (
          nwr["tourism"~"attraction|museum|viewpoint|zoo|gallery|theme_park"](around:{radius},{lat},{lon});
          nwr["historic"~"monument|castle|fort|memorial|palace|archaeological_site"](around:{radius},{lat},{lon});
          nwr["natural"~"beach|waterfall|peak"](around:{radius},{lat},{lon});
          nwr["amenity"~"place_of_worship|marketplace"](around:{radius},{lat},{lon});
          nwr["leisure"~"park|nature_reserve"](around:{radius},{lat},{lon});
        );
        out center 40;
        """
        try:
            elements = _run_overpass_query(query, timeout=15)
        except Exception:
            elements = []

        for el in elements:
            tags = el.get("tags") or {}
            name = tags.get("name:en") or tags.get("name")
            if not name:
                continue
            norm = name.strip().lower()
            if norm in seen or len(norm) < 2:
                continue
            seen.add(norm)

            spot_lat = el.get("lat") or (el.get("center", {}).get("lat"))
            spot_lon = el.get("lon") or (el.get("center", {}).get("lon"))
            if spot_lat is None or spot_lon is None:
                continue

            score = 0
            if "wikidata" in tags or "wikipedia" in tags:
                score += 3
            if tags.get("tourism") == "attraction":
                score += 2
            if tags.get("historic"):
                score += 1
            if "website" in tags or "contact:website" in tags:
                score += 1

            cat = classify_osm_spot(tags, name)
            dist = haversine_km(lat, lon, spot_lat, spot_lon)
            brng = calculate_bearing(lat, lon, spot_lat, spot_lon)
            comp = bearing_to_compass(brng)
            dname = bearing_to_direction_name(brng)

            ranked_spots.append({
                "name": name,
                "lat": spot_lat,
                "lon": spot_lon,
                "category": cat,
                "score": score,
                "dist": dist,
                "bearing": brng,
                "compass": comp,
                "direction_name": dname,
                "centre": c_name,
                "map_url": f"https://www.google.com/maps/search/?api=1&query={spot_lat:.6f},{spot_lon:.6f}",
            })

    ranked_spots.sort(key=lambda s: -s["score"])
    return ranked_spots



def get_eat_places(place, limit=6):
    lat, lon = place["latitude"], place["longitude"]
    c_name = place["name"]
    results = []
    seen = set()

    # 1. Fast Nominatim search (0.2s - 0.5s, highly reliable)
    try:
        nom_res = _http_get_json(
            NOMINATIM_URL,
            params={"q": f"restaurants in {c_name}", "format": "json", "limit": limit + 4},
            timeout=4,
            retries=1,
        )
        if isinstance(nom_res, list):
            for item in nom_res:
                raw_name = item.get("display_name", "").split(",")[0].strip()
                low_name = raw_name.lower()
                if any(w in low_name for w in ["road", "highway", "marg", "expressway", "lane"]):
                    continue
                if raw_name and low_name not in seen:
                    seen.add(low_name)
                    item_lat = float(item["lat"])
                    item_lon = float(item["lon"])
                    dist = haversine_km(lat, lon, item_lat, item_lon)
                    brng = calculate_bearing(lat, lon, item_lat, item_lon)
                    results.append({
                        "name": raw_name,
                        "score": 1,
                        "dist": dist,
                        "compass": bearing_to_compass(brng),
                        "detail": "",
                        "centre": c_name,
                        "map_url": f"https://www.google.com/maps/search/?api=1&query={item_lat:.6f},{item_lon:.6f}",
                    })
                    if len(results) >= limit:
                        break
    except Exception:
        pass

    if results:
        results.sort(key=lambda x: x["dist"])
        return results[:limit]

    # 2. Fast Overpass fallback (small radius, quick timeout)
    query = f"""
    [out:json][timeout:5];
    (
      node["amenity"~"restaurant|cafe"](around:5000,{lat},{lon});
    );
    out 12;
    """
    try:
        elements = _run_overpass_query(query, timeout=5)
        for el in elements:
            tags = el.get("tags") or {}
            name = tags.get("name:en") or tags.get("name")
            if not name or name.strip().lower() in seen:
                continue
            seen.add(name.strip().lower())
            spot_lat = el.get("lat")
            spot_lon = el.get("lon")
            if spot_lat is None or spot_lon is None:
                continue
            dist = haversine_km(lat, lon, spot_lat, spot_lon)
            brng = calculate_bearing(lat, lon, spot_lat, spot_lon)
            cuisine = tags.get("cuisine")
            cuisine_label = f" (Cuisine: {cuisine.replace(';', ', ')})" if cuisine else ""
            results.append({
                "name": name,
                "score": 2 if cuisine else 1,
                "dist": dist,
                "compass": bearing_to_compass(brng),
                "detail": cuisine_label,
                "centre": c_name,
                "map_url": f"https://www.google.com/maps/search/?api=1&query={spot_lat:.6f},{spot_lon:.6f}",
            })
    except Exception:
        pass

    results.sort(key=lambda x: x["dist"])
    return results[:limit]


def get_stay_places(place, limit=6):
    lat, lon = place["latitude"], place["longitude"]
    c_name = place["name"]
    results = []
    seen = set()

    # 1. Fast Nominatim search (0.2s - 0.5s)
    try:
        nom_res = _http_get_json(
            NOMINATIM_URL,
            params={"q": f"hotels in {c_name}", "format": "json", "limit": limit + 4},
            timeout=4,
            retries=1,
        )
        if isinstance(nom_res, list):
            for item in nom_res:
                raw_name = item.get("display_name", "").split(",")[0].strip()
                low_name = raw_name.lower()
                if any(w in low_name for w in ["road", "highway", "marg", "expressway", "lane"]):
                    continue
                if raw_name and low_name not in seen:
                    seen.add(low_name)
                    item_lat = float(item["lat"])
                    item_lon = float(item["lon"])
                    dist = haversine_km(lat, lon, item_lat, item_lon)
                    brng = calculate_bearing(lat, lon, item_lat, item_lon)
                    results.append({
                        "name": raw_name,
                        "score": 1,
                        "dist": dist,
                        "compass": bearing_to_compass(brng),
                        "detail": " (Hotel)",
                        "centre": c_name,
                        "map_url": f"https://www.google.com/maps/search/?api=1&query={item_lat:.6f},{item_lon:.6f}",
                    })
                    if len(results) >= limit:
                        break
    except Exception:
        pass

    if results:
        results.sort(key=lambda x: x["dist"])
        return results[:limit]

    # 2. Fast Overpass fallback (small radius, quick timeout)
    query = f"""
    [out:json][timeout:5];
    (
      node["tourism"~"hotel|hostel|guest_house|resort"](around:5000,{lat},{lon});
    );
    out 12;
    """
    try:
        elements = _run_overpass_query(query, timeout=5)
        for el in elements:
            tags = el.get("tags") or {}
            name = tags.get("name:en") or tags.get("name")
            if not name or name.strip().lower() in seen:
                continue
            seen.add(name.strip().lower())
            spot_lat = el.get("lat")
            spot_lon = el.get("lon")
            if spot_lat is None or spot_lon is None:
                continue
            dist = haversine_km(lat, lon, spot_lat, spot_lon)
            brng = calculate_bearing(lat, lon, spot_lat, spot_lon)
            kind = tags.get("tourism", "hotel").replace("_", " ").title()
            stars = tags.get("stars")
            stars_label = f", {stars}★" if stars else ""
            results.append({
                "name": name,
                "score": 2 if stars else 1,
                "dist": dist,
                "compass": bearing_to_compass(brng),
                "detail": f" ({kind}{stars_label})",
                "centre": c_name,
                "map_url": f"https://www.google.com/maps/search/?api=1&query={spot_lat:.6f},{spot_lon:.6f}",
            })
    except Exception:
        pass

    results.sort(key=lambda x: x["dist"])
    return results[:limit]


# ---------- Itinerary grouped by area ----------
def build_area_itinerary(spots, weather, days, centre_name):
    if not spots:
        return []

    # Round robin pick spots across categories
    cat_buckets = {}
    for s in spots:
        c = s["category"]
        if c not in cat_buckets:
            cat_buckets[c] = []
        cat_buckets[c].append(s)

    cats = list(cat_buckets.keys())
    needed = min(len(spots), max(days * 3, days))
    selected = []
    idx = 0
    while len(selected) < needed:
        added_any = False
        for c in cats:
            if cat_buckets[c]:
                selected.append(cat_buckets[c].pop(0))
                added_any = True
                if len(selected) >= needed:
                    break
        if not added_any:
            break

    # Sort selected spots by compass bearing from centre
    selected.sort(key=lambda s: s["bearing"])

    # Split into days
    per_day = max(1, -(-len(selected) // days))
    day_itinerary = []
    slots = ["Morning", "Afternoon", "Evening"]

    for d in range(days):
        chunk = selected[d * per_day: (d + 1) * per_day]
        rain = weather[d]["rain"] if d < len(weather) else 0

        if chunk:
            avg_bearing = sum(s["bearing"] for s in chunk) / len(chunk)
            dominant_dir = bearing_to_direction_name(avg_bearing)
        else:
            dominant_dir = "around"

        day_itinerary.append({
            "day": d + 1,
            "dominant_dir": dominant_dir,
            "rain": rain,
            "slots": [
                {
                    "time": slots[i % len(slots)],
                    "name": s["name"],
                    "category": s["category"],
                    "dist": s["dist"],
                    "compass": s["compass"],
                    "map_url": s["map_url"],
                }
                for i, s in enumerate(chunk)
            ],
        })
    return day_itinerary


# ---------- Budget computation ----------
def compute_budget_breakdown(days, user_budget, round_trip_travel):
    nights = max(days - 1, 0)
    stay_cost = 1500 * nights
    food_cost = 700 * days
    local_cost = 400 * days
    extras_cost = 300 * days
    total = stay_cost + food_cost + local_cost + extras_cost + round_trip_travel

    items = [
        {"label": "Stay", "detail": f"₹1,500 × {nights} night(s)", "amount": stay_cost},
        {"label": "Food", "detail": f"₹700 × {days} day(s)", "amount": food_cost},
        {"label": "Local transport", "detail": f"₹400 × {days} day(s)", "amount": local_cost},
        {"label": "Entry fees & extras", "detail": f"₹300 × {days} day(s)", "amount": extras_cost},
    ]
    if round_trip_travel > 0:
        items.append({
            "label": "Round-trip travel",
            "detail": "Estimated cheapest public transport (round-trip)",
            "amount": round_trip_travel,
        })

    diff = (user_budget - total) if (user_budget is not None) else None
    return {
        "items": items,
        "total": total,
        "user_budget": user_budget,
        "diff": diff,
    }


# ---------- Public plan_trip function ----------
def plan_trip(text):
    t = text.strip()
    if not t:
        return (
            "Tell me where you want to go and I will help. Try:\n"
            "- \"Plan a 3-day trip from Delhi to Goa under ₹15,000\"\n"
            "- \"How do I get from Delhi to Goa?\"\n"
            "- \"Best beaches in Goa India\"\n"
            "- \"Where to eat in Jaipur\"\n"
            "- \"Hotels in Udaipur\""
        )

    # 1. Parse places, days, budget
    origin_name, dest_name = find_places(t)
    if not dest_name:
        return (
            "Tell me where you want to go and I will help. Try:\n"
            "- \"Plan a 3-day trip from Delhi to Goa under ₹15,000\"\n"
            "- \"How do I get from Delhi to Goa?\"\n"
            "- \"Best beaches in Goa India\"\n"
            "- \"Where to eat in Jaipur\"\n"
            "- \"Hotels in Udaipur\""
        )

    days, budget = parse_days_budget(t)
    prefer_india = bool(re.search(r"₹|\brs\.?\b|\binr\b", t, re.I))

    # 2. Geocode destination
    dest = geocode_place(dest_name, prefer_india=prefer_india)
    if not dest:
        return f"I couldn't find '{dest_name}'. Please check the spelling."

    # 3. Geocode origin if provided
    origin = None
    if origin_name:
        orig_india = prefer_india or (dest.get("country_code") == "IN")
        origin = geocode_place(origin_name, prefer_india=orig_india)
        if not origin:
            return f"I couldn't find '{origin_name}'. Please check the spelling."

    # 4. Intent detection
    low = t.lower()
    is_full_plan = bool(re.search(r"\b(?:plan|itinerary|trip|weekend|holiday|vacation)\b", low))
    wants_transport = bool(
        origin or re.search(r"\b(?:how\s+to\s+reach|how\s+do\s+i\s+get|transport|flight|train|bus|route|drive|reach)\b", low)
    )
    wants_weather = bool(re.search(r"\b(?:weather|forecast|temperature|rain|climate|packing)\b", low))
    wants_eat = bool(re.search(r"\b(?:eat|food|restaurant|restaurants|cafe|cafes|dining|dishes|cuisine)\b", low))
    wants_stay = bool(re.search(r"\b(?:stay|hotel|hotels|hostel|hostels|resort|resorts|accommodation|lodging)\b", low))

    # Category-specific spot intents
    wants_beaches = bool(re.search(r"\b(?:beach|beaches|coast|shore)\b", low))
    wants_forts = bool(re.search(r"\b(?:fort|forts|palace|palaces|castle|monument|monuments|heritage)\b", low))
    wants_nature = bool(re.search(r"\b(?:nature|viewpoint|viewpoints|waterfall|waterfalls|peak|park|parks)\b", low))
    wants_museums = bool(re.search(r"\b(?:museum|museums|gallery|galleries)\b", low))
    wants_temples = bool(re.search(r"\b(?:temple|temples|church|churches|mosque|mosques|worship)\b", low))
    wants_markets = bool(re.search(r"\b(?:market|markets|bazaar|shopping|mall)\b", low))
    wants_attractions = bool(re.search(r"\b(?:attraction|attractions|sights|sightseeing|spots|places\s+to\s+visit|things\s+to\s+do)\b", low))

    has_specific_category = (
        wants_beaches or wants_forts or wants_nature or wants_museums or
        wants_temples or wants_markets or wants_attractions
    )

    # Disambiguate: if it's a specific question and not explicitly "plan a trip"
    is_pure_question = False
    if not re.search(r"\bplan\b", low):
        if (wants_transport and not is_full_plan) or wants_eat or wants_stay or wants_weather or has_specific_category:
            is_pure_question = True

    # 5. Fetch ALL external data in PARALLEL for speed
    notes = []
    transport_data = None
    weather = []
    spots = []
    eat_places = []
    stay_places = []

    need_spots   = is_full_plan or has_specific_category
    need_eat     = is_full_plan or wants_eat
    need_stay    = is_full_plan or wants_stay
    need_weather = is_full_plan or wants_weather
    need_transport = origin and (is_full_plan or wants_transport)

    futures = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        if need_transport:
            futures["transport"] = ex.submit(get_transport_info, origin, dest)
        if need_weather:
            futures["weather"]   = ex.submit(get_weather_forecast, dest, days)
        if need_spots:
            futures["spots"]     = ex.submit(get_destination_spots, dest)
        if need_eat:
            futures["eat"]       = ex.submit(get_eat_places, dest)
        if need_stay:
            futures["stay"]      = ex.submit(get_stay_places, dest)

    for key, fut in futures.items():
        try:
            val = fut.result(timeout=20)
            if key == "transport":  transport_data = val
            elif key == "weather":  weather        = val
            elif key == "spots":    spots          = val
            elif key == "eat":      eat_places     = val
            elif key == "stay":     stay_places    = val
        except Exception:
            if key == "transport": notes.append("Routing service was unavailable right now.")
            elif key == "weather": notes.append("Weather forecast was unavailable right now.")
            elif key == "spots":   notes.append("Attraction data was unavailable right now.")
            elif key == "eat":     notes.append("Dining recommendations were unavailable right now.")
            elif key == "stay":    notes.append("Accommodation recommendations were unavailable right now.")

    # 6. Render Markdown sections
    dest_label_parts = [dest["name"], dest.get("admin1"), dest.get("country")]
    dest_label = ", ".join([p for p in dest_label_parts if p])
    out = []

    # Title header
    if is_pure_question:
        out.append(f"# Travel Guide: {dest_label}")
    else:
        out.append(f"# Your {days}-Day Trip Plan: {dest_label}")
    out.append(f"_(If this is not the intended place, add the country or region, e.g. \"{dest['name']} India\")_")
    out.append("")

    # Transport section
    if transport_data and (is_full_plan or wants_transport):
        orig_label_parts = [origin["name"], origin.get("admin1"), origin.get("country")]
        orig_label = ", ".join([p for p in orig_label_parts if p])

        out.append(f"## Getting There: {orig_label} to {dest_label}")
        out.append(f"- **Straight-line distance (haversine):** {transport_data['straight_km']:.0f} km")
        osrm_tag = "public driving route" if transport_data["is_osrm"] else "fallback estimated at 1.3× straight-line"
        out.append(
            f"- **Road distance & drive time ({osrm_tag}):** "
            f"{transport_data['road_km']:.0f} km (~{transport_data['drive_time_str']})"
        )
        out.append("")
        out.append("### Transport Options:")
        for opt in transport_data["options"]:
            cost_str = ""
            if opt["cost_range"]:
                cost_str = f", estimated cost ₹{opt['cost_range'][0]:,} – ₹{opt['cost_range'][1]:,}"
            elif not transport_data["both_in_india"]:
                cost_str = ", fares vary by booking & season (check live operators)"
            out.append(f"- **{opt['mode']}:** ~{opt['time']}{cost_str}")

        out.append("")
        out.append(f"- **Fastest option:** {transport_data['fastest']}")
        out.append(f"- **Cheapest public option:** {transport_data['cheapest_public']}")
        if transport_data["diff_countries"]:
            out.append(f"- **Border note:** Check passport, visa and entry rules for travel between {origin['country']} and {dest['country']}.")
        out.append("_Note: Travel times and costs are rough estimates, not live tickets or fares._")
        out.append("")

    # If pure transport was requested and done, skip itinerary/budget
    if is_pure_question and wants_transport and not (wants_weather or wants_eat or wants_stay or has_specific_category):
        pass  # skip to footer

    # Weather & Packing
    if is_full_plan or wants_weather:
        out.append(f"## Weather Forecast ({dest['name']})")
        if weather:
            for w in weather:
                out.append(f"- **{w['date']}:** {w['desc']}, {w['tmin']:.0f}°C – {w['tmax']:.0f}°C, rain chance {w['rain']}%")
            tips = compute_packing_tips(weather)
            if tips:
                out.append("")
                out.append("**Packing tips based on forecast:**")
                for tip in tips:
                    out.append(f"- {tip}")
        else:
            out.append("- Forecast data was temporarily unavailable.")
        out.append("")

    # Specific category queries (e.g. "Best beaches in Goa India")
    if is_pure_question and has_specific_category:
        cat_map = {
            "Beaches": wants_beaches,
            "Forts & heritage": wants_forts,
            "Nature & viewpoints": wants_nature,
            "Museums & galleries": wants_museums,
            "Temples & places of worship": wants_temples,
            "Markets": wants_markets,
            "Attractions": wants_attractions,
        }
        for cat_name, wanted in cat_map.items():
            if not wanted:
                continue
            cat_spots = [s for s in spots if s["category"] == cat_name]
            out.append(f"## {cat_name} in {dest['name']}")
            if cat_spots:
                for s in cat_spots[:10]:
                    out.append(
                        f"- **{s['name']}** — {s['dist']:.1f} km {s['compass']} of {dest['name']} "
                        f"([View on Google Maps]({s['map_url']}))"
                    )
            else:
                out.append(f"- No specific {cat_name.lower()} found in OpenStreetMap around {dest['name']}.")
            out.append("")

    # Itinerary grouped by area (full plan only)
    if is_full_plan:
        out.append(f"## Day-by-Day Itinerary (Grouped by Area)")
        itinerary = build_area_itinerary(spots, weather, days, dest["name"])
        if itinerary:
            for day in itinerary:
                dir_label = day["dominant_dir"]
                out.append(f"### Day {day['day']}: Exploring {dir_label} of {dest['name']}")
                if day["rain"] >= 60:
                    out.append(f"_High rain chance ({day['rain']}%): carry an umbrella and keep indoor options handy._")
                for slot in day["slots"]:
                    out.append(
                        f"- **{slot['time']}:** {slot['name']} ({slot['category']}, "
                        f"{slot['dist']:.1f} km {slot['compass']} of {dest['name']}) "
                        f"— [Google Maps]({slot['map_url']})"
                    )
                out.append("")
        else:
            out.append("- Attractions were unavailable or sparse in this area. Try adding nearby major sights.")
            out.append("")

    # Where to eat
    if is_full_plan or wants_eat:
        out.append(f"## Where to Eat in {dest['name']}")
        if eat_places:
            for ep in eat_places:
                out.append(
                    f"- **{ep['name']}**{ep['detail']} — {ep['dist']:.1f} km {ep['compass']} of {dest['name']} "
                    f"([Google Maps]({ep['map_url']}))"
                )
        else:
            c_query = dest["name"].replace(" ", "+")
            out.append(f"- Try local culinary hubs and top street food areas around central {dest['name']}.")
            out.append(f"- **[Search Best Restaurants & Cafes in {dest['name']} on Google Maps](https://www.google.com/maps/search/?api=1&query=best+restaurants+in+{c_query})**")
        out.append("_(Note: No live prices or availability. Check opening hours locally.)_")
        out.append("")

    # Where to stay
    if is_full_plan or wants_stay:
        out.append(f"## Where to Stay in {dest['name']}")
        if stay_places:
            for sp in stay_places:
                out.append(
                    f"- **{sp['name']}**{sp['detail']} — {sp['dist']:.1f} km {sp['compass']} of {dest['name']} "
                    f"([Google Maps]({sp['map_url']}))"
                )
        else:
            c_query = dest["name"].replace(" ", "+")
            out.append(f"- Recommended: Choose hotels or boutique homestays in central {dest['name']} for easy access to top sights.")
            out.append(f"- **[Browse Top Hotels & Hostels in {dest['name']} on Google Maps](https://www.google.com/maps/search/?api=1&query=hotels+in+{c_query})**")
        out.append("_(Note: No live prices or availability. Book through verified portals.)_")
        out.append("")

    # Budget (full plan only)
    if is_full_plan:
        round_trip_val = transport_data["round_trip_travel"] if transport_data else 0
        b_info = compute_budget_breakdown(days, budget, round_trip_val)
        out.append("## Estimated Budget (Rough Assumptions)")
        for item in b_info["items"]:
            out.append(f"- **{item['label']}:** {item['detail']} = ₹{item['amount']:,}")
        out.append(f"- **Estimated Total: ₹{b_info['total']:,}**")

        if b_info["user_budget"] is not None:
            if b_info["diff"] >= 0:
                out.append(
                    f"Your budget of ₹{b_info['user_budget']:,} leaves about ₹{b_info['diff']:,} "
                    f"for shopping, extras, or emergencies."
                )
            else:
                out.append(
                    f"This is about ₹{-b_info['diff']:,} over your ₹{b_info['user_budget']:,} budget. "
                    f"Suggestion: choose a budget hostel/dorm, focus on free sights, or reduce by one night."
                )
        out.append("_Note: Budget figures are standard rough assumptions, NOT live prices._")
        out.append("")

    # Notes if any
    if notes:
        out.append("### Service Notes:")
        for note in notes:
            out.append(f"- {note}")
        out.append("")

    # Ask me next
    c_city = dest["name"]
    out.append("**Ask me next:**")
    out.append(f"- \"Where to eat in {c_city}\"")
    out.append(f"- \"Hotels in {c_city}\"")
    out.append(f"- \"Weather and packing tips for {c_city}\"")
    out.append(f"- \"Plan a 3-day trip to {c_city}\"")
    out.append("")

    # Data sources
    out.append(
        "_Data sources: Open-Meteo (weather & geocoding), OpenStreetMap & Overpass "
        "(places & attractions), OSRM (public routing service). Costs and times are rough estimates._"
    )

    return "\n".join(out)


# ---------- ACP handlers ----------
@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id,
        ),
    )
    for item in msg.content:
        if isinstance(item, StartSessionContent):
            continue
        if not isinstance(item, TextContent):
            continue
        try:
            reply = plan_trip(item.text)
        except Exception as exc:
            ctx.logger.error(f"Planner error: {exc}")
            reply = "Something went wrong while planning. Please try again."
        await ctx.send(
            sender,
            ChatMessage(
                timestamp=datetime.now(timezone.utc),
                msg_id=uuid4(),
                content=[
                    TextContent(type="text", text=reply),
                    EndSessionContent(type="end-session"),
                ],
            ),
        )


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)
