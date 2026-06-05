from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from geopy.geocoders import Nominatim
import requests
from dotenv import load_dotenv
import os, json
from shapely.geometry import shape,Point
import pandas as pd
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from jinja2 import TemplateNotFound
import logging
from flask import send_from_directory
import re
from copy import deepcopy

from hazard_engine import build_hazard_results, merge_structured_result
from location_service import location_from_session
from rag_service import retrieve_chunks
from source_registry import load_hazard_registry, load_jurisdictions, load_local_plans, load_resident_guidance_chunks, source_records_payload

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# --- Load env vars and setup ---
load_dotenv("secret.env")
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LAT = 37.75
DEFAULT_LON = -122.2
COVERAGE_BOUNDS = {
    "min_lat": 37.0,
    "max_lat": 38.5,
    "min_lon": -123.0,
    "max_lon": -121.0,
}
RISK_DATA_WARNING = "Risk data is temporarily unavailable, but general preparedness guidance is still shown."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = None
logger.warning("AI hazard responses are disabled in production-hardening mode; deterministic fallback guidance will be used.")

DATA_SOURCES = [
    {
        "key": "flood",
        "label": "Flood data",
        "name": "FEMA",
        "full_name": "Federal Emergency Management Agency",
        "icon": "fa-house-flood-water",
    },
    {
        "key": "earthquake",
        "label": "Earthquake data",
        "name": "USGS",
        "full_name": "United States Geological Survey",
        "icon": "fa-mountain",
    },
    {
        "key": "wildfire",
        "label": "Wildfire data",
        "name": "CAL FIRE",
        "full_name": "California Department of Forestry and Fire Protection",
        "icon": "fa-fire",
    },
]

HAZARD_DATA_PATH = os.path.join(BASE_DIR, "oakland.json")
HAZARD_PROFILE_SESSION_KEY = "hazard_user_profile"
LOCATION_MODE_SESSION_KEY = "location_mode"
HAZARD_LOCATION_OPTIONS = [
    {"value": "flatlands", "label": "Flatlands"},
    {"value": "hills", "label": "Hills"},
    {"value": "shoreline", "label": "Shoreline"},
    {"value": "other", "label": "Other"},
]
OAKLAND_HAZARD_GUIDANCE_ZIPS = {
    "94601", "94602", "94603", "94605", "94606", "94607",
    "94609", "94610", "94611", "94612", "94613", "94618",
    "94619", "94621",
}
CORE_DEMO_HAZARDS = {"flood", "wildfire", "earthquake"}
ADDITIONAL_HAZARD_LABELS = {
    "landslide": "Landslide",
    "sea_level_rise": "Sea level rise",
    "severe_weather": "Severe weather",
    "tsunami": "Tsunami",
    "dam_failure": "Dam failure",
    "drought": "Drought",
    "extreme_heat": "Extreme heat",
    "poor_air_quality": "Poor air quality",
    "utility_disruption": "Utility disruption",
    "high_wind": "High wind",
}
ADDITIONAL_HAZARD_CONTEXT = {
    "landslide": "Local plans may identify hillside, creek-bank, post-fire, or rain-triggered slope hazards. This is plan context until an official landslide GIS layer is checked.",
    "sea_level_rise": "Local plans may identify shoreline, groundwater, storm-surge, or coastal flooding concerns. This is long-term local context, not an address flood-zone match.",
    "severe_weather": "Local plans may identify heavy rain, storms, wind, drainage, and service-disruption issues that can affect recovery even outside mapped hazard zones.",
    "tsunami": "Local plans may identify shoreline or marina evacuation zones. This should be checked with official tsunami evacuation maps before making address-level claims.",
    "dam_failure": "Local plans may identify dam-inundation scenarios. This is important for evacuation planning but needs official inundation mapping for address-level precision.",
    "drought": "Local plans may identify water-supply, vegetation, fire-weather, or agricultural impacts. This is generally regional context rather than address-point exposure.",
    "extreme_heat": "Local plans may identify heat exposure, nighttime heat, public-health risk, and power/cooling needs. Household vulnerability can matter more than property location.",
    "poor_air_quality": "Local plans may identify wildfire smoke and air-quality impacts. This is often regional and health-specific rather than parcel-specific.",
    "utility_disruption": "Local plans may identify power, water, communication, fuel, or lifeline disruptions that shape preparedness and recovery after major hazards.",
    "high_wind": "Local plans may identify wind impacts, treefall, power outages, and wildfire spread conditions. This is generally weather/context guidance.",
}
ADDITIONAL_HAZARD_PRIORITY = {
    "Berkeley": ["landslide", "poor_air_quality", "extreme_heat", "utility_disruption", "tsunami", "sea_level_rise", "high_wind", "drought"],
    "Alameda": ["sea_level_rise", "tsunami", "dam_failure", "poor_air_quality", "extreme_heat"],
    "Oakland": ["landslide", "sea_level_rise", "severe_weather", "tsunami", "dam_failure", "drought"],
    "Fremont": ["landslide", "sea_level_rise", "tsunami", "severe_weather"],
    "Newark": ["sea_level_rise", "tsunami", "landslide", "severe_weather"],
    "Union City": ["landslide", "sea_level_rise", "tsunami", "severe_weather"],
    "Hayward": ["landslide", "sea_level_rise", "severe_weather"],
    "San Leandro": ["sea_level_rise", "landslide", "severe_weather"],
    "Dublin": ["landslide", "severe_weather", "dam_failure", "drought"],
    "Livermore": ["severe_weather", "landslide", "dam_failure", "drought"],
    "Pleasanton": ["severe_weather", "landslide", "dam_failure", "drought"],
}


def slugify_hazard_name(name):
    cleaned = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower())
    return cleaned.strip("-")


def coerce_priority_score(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def dedupe_steps(steps):
    ordered = []
    seen = set()
    for step in steps or []:
        normalized = (step or "").strip()
        if not normalized:
            continue
        step_key = normalized.lower()
        if step_key in seen:
            continue
        seen.add(step_key)
        ordered.append(normalized)
    return ordered


def load_hazard_dataset():
    # Normalize the dataset once at startup so routes and templates share the same structure.
    empty_dataset = {
        "hazards": [],
        "system_insights": {
            "how_oakland_prepares": [],
            "major_weaknesses": [],
            "key_recommendations": [],
        },
    }

    if not os.path.exists(HAZARD_DATA_PATH):
        logger.warning("Hazard dataset not found at %s", HAZARD_DATA_PATH)
        return empty_dataset

    try:
        with open(HAZARD_DATA_PATH, "r", encoding="utf-8") as source:
            raw_dataset = json.load(source)
    except Exception:
        logger.exception("Hazard dataset failed to load from %s", HAZARD_DATA_PATH)
        return empty_dataset

    normalized_hazards = []
    for hazard in raw_dataset.get("hazards", []):
        normalized = dict(hazard)
        normalized["priority_score"] = coerce_priority_score(hazard.get("priority_score"))
        normalized["slug"] = slugify_hazard_name(hazard.get("name"))
        normalized["action_steps"] = dedupe_steps(hazard.get("action_steps"))
        normalized["top_risks"] = dedupe_steps(hazard.get("top_risks"))
        normalized["locations"] = dedupe_steps(hazard.get("locations"))
        normalized["at_risk_groups"] = dedupe_steps(hazard.get("at_risk_groups"))
        normalized_hazards.append(normalized)

    system_insights = raw_dataset.get("system_insights") or empty_dataset["system_insights"]
    return {
        "hazards": normalized_hazards,
        "system_insights": {
            "how_oakland_prepares": dedupe_steps(system_insights.get("how_oakland_prepares")),
            "major_weaknesses": dedupe_steps(system_insights.get("major_weaknesses")),
            "key_recommendations": dedupe_steps(system_insights.get("key_recommendations")),
        },
    }


def get_system_insights():
    return deepcopy(hazard_dataset.get("system_insights", {}))


def normalize_user_profile(payload=None):
    payload = payload or {}
    housing = (payload.get("housing") or "").strip().lower()
    if housing not in {"renter", "homeowner"}:
        housing = ""

    location = (payload.get("location") or "").strip().lower()
    valid_locations = {item["value"] for item in HAZARD_LOCATION_OPTIONS}
    if location not in valid_locations:
        location = "other" if location else ""

    try:
        age = int(str(payload.get("age", "")).strip())
        if age < 0:
            age = None
    except (TypeError, ValueError):
        age = None

    return {
        "housing": housing,
        "age": age,
        "location": location,
    }


def is_oakland_hazard_context(zip_code=None, address=None):
    normalized_zip = str(zip_code or "").strip()
    normalized_address = (address or "").strip().lower()

    if normalized_address:
        if "oakland" in normalized_address:
            return True

        other_city_markers = (
            "berkeley", "alameda", "fremont", "hayward", "san leandro",
            "union city", "newark", "dublin", "pleasanton", "livermore",
            "castro valley", "san lorenzo", "emeryville", "piedmont",
        )
        if any(marker in normalized_address for marker in other_city_markers):
            return False

    return normalized_zip in OAKLAND_HAZARD_GUIDANCE_ZIPS


def get_oakland_guidance_error_context():
    return {
        "title": "Oakland Guidance Only",
        "error_heading": "Oakland-specific hazard guidance is limited to Oakland addresses",
        "error_message": "This hazard intelligence layer was built from Oakland planning data, so it only appears for Oakland locations.",
        "helper_message": "You can still use the countywide map, ZIP risk scoring, and resources for other Alameda County cities.",
    }


def get_saved_hazard_profile():
    return normalize_user_profile(session.get(HAZARD_PROFILE_SESSION_KEY, {}))


def get_session_location_context():
    zip_code = session.get("zip_code")
    address = session.get("address")
    location_mode = session.get(LOCATION_MODE_SESSION_KEY, "zip" if zip_code else "")
    location_result = location_from_session({
        "input_address": session.get("input_address"),
        "address": address,
        "zip_code": zip_code,
        "lat": session.get("lat"),
        "lon": session.get("lon"),
        "city": session.get("city"),
        "county": session.get("county"),
        "neighborhood": session.get("neighborhood"),
        "census_tract": session.get("census_tract"),
        "location_mode": location_mode,
    })
    has_precise_location = (
        location_mode == "address" and
        address and
        is_valid_coordinate(session.get("lat"), session.get("lon"))
    )
    return {
        "zip_code": zip_code,
        "address": address,
        "display_name": address or (f"ZIP {zip_code}" if zip_code else "No location selected"),
        "is_oakland": is_oakland_hazard_context(zip_code, address),
        "location_mode": location_mode,
        "has_precise_location": bool(has_precise_location),
        "location_result": location_result.model_dump(),
        "city": location_result.city,
        "county": location_result.county,
        "precision_label": "Address point" if has_precise_location else "ZIP estimate" if zip_code else "County fallback",
        "gis_status": "Address point ready for GIS checks" if has_precise_location else "No address-level GIS check available from ZIP-only input",
    }


def get_zip_risk_snapshot(zip_code):
    if not zip_code:
        return {}

    data = zip_risk_data.get(str(zip_code), {})
    if not data:
        return {}

    snapshot = {}
    csv_mappings = {
        "earthquake": ("Earthquake_Risk_Score", "Earthquake_Risk_Explanation"),
        "flood": ("Flood_Risk_Score", "Flood_Risk_Explanation"),
        "wildfire": ("Wildfire_Risk_Score", "Wildfire_Risk_Explanation"),
    }

    for slug, (score_key, explanation_key) in csv_mappings.items():
        try:
            score = float(data.get(score_key, 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        snapshot[slug] = {
            "score": score,
            "level": get_risk_level(score),
            "explanation": data.get(explanation_key, RISK_DATA_WARNING),
        }

    return snapshot


def step_priority(step, keywords):
    text = (step or "").lower()
    return 0 if any(keyword in text for keyword in keywords) else 1


def personalize_hazard(hazard, user, location_context=None):
    # Apply lightweight profile rules without mutating the canonical JSON dataset in memory.
    personalized = deepcopy(hazard)
    personalized["base_priority_score"] = coerce_priority_score(hazard.get("priority_score"))
    personalized["priority_score"] = personalized["base_priority_score"]
    personalized["personalization_notes"] = []
    personalized["action_steps"] = dedupe_steps(personalized.get("action_steps"))

    user = normalize_user_profile(user)
    location_context = location_context or {}
    zip_risk_snapshot = location_context.get("zip_risk_snapshot") or {}
    local_risk = zip_risk_snapshot.get(personalized.get("slug"))

    if local_risk:
        personalized["local_risk_score"] = local_risk["score"]
        personalized["local_risk_level"] = local_risk["level"]
        personalized["local_risk_explanation"] = local_risk["explanation"]

        if local_risk["score"] >= 7:
            personalized["priority_score"] = min(personalized["priority_score"] + 2, 10)
            personalized["personalization_notes"].append(
                f"Your current Oakland location has a high local {personalized['name'].lower()} score ({local_risk['score']:.1f}/10), so this hazard is ranked higher for you."
            )
        elif local_risk["score"] >= 4:
            personalized["priority_score"] = min(personalized["priority_score"] + 1, 10)
            personalized["personalization_notes"].append(
                f"Your current Oakland location has a moderate local {personalized['name'].lower()} score ({local_risk['score']:.1f}/10), which raises its priority."
            )

    if not any(user.values()):
        personalized["personalized_what_this_means_for_you"] = personalized.get("what_this_means_for_you", "")
        return personalized

    if user.get("housing") == "renter":
        personalized["action_steps"] = sorted(
            personalized["action_steps"],
            key=lambda step: step_priority(step, ("evacuation", "route", "leave", "bag", "shelter"))
        )
        personalized["action_steps"] = dedupe_steps([
            "Know your building exits, meeting point, and what you can take with you quickly.",
            *personalized["action_steps"],
        ])
        personalized["personalization_notes"].append(
            "Because you selected renter, evacuation and fast-leave steps are moved to the top."
        )

    if user.get("age") is not None and user["age"] >= 65:
        personalized["action_steps"] = dedupe_steps([
            *personalized["action_steps"],
            "Pack extra medications, a written medication list, and backup medical supplies.",
            "Plan transportation and mobility support in case you need help leaving or reaching a safe site."
        ])
        personalized["personalization_notes"].append(
            "Because you selected age 65+, medical and mobility planning steps were added."
        )

    if user.get("location") == "hills" and personalized.get("slug") == "wildfire":
        personalized["priority_score"] = min(personalized["priority_score"] + 2, 10)
        personalized["personalization_notes"].append(
            "Wildfire priority was raised because you selected a hills location."
        )

    personalized["personalized_what_this_means_for_you"] = personalized.get("what_this_means_for_you", "")
    if personalized["personalization_notes"]:
        personalized["personalized_what_this_means_for_you"] = (
            f"{personalized['personalized_what_this_means_for_you']} "
            f"{' '.join(personalized['personalization_notes'])}"
        ).strip()

    return personalized


def get_all_hazards(user=None, location_context=None):
    user = normalize_user_profile(user)
    location_context = location_context or get_session_location_context()
    if "zip_risk_snapshot" not in location_context:
        location_context["zip_risk_snapshot"] = get_zip_risk_snapshot(location_context.get("zip_code"))

    hazards = [personalize_hazard(hazard, user, location_context) for hazard in hazard_dataset.get("hazards", [])]
    location_result = location_from_session({
        **(location_context.get("location_result") or {}),
        "address": location_context.get("address"),
        "zip_code": location_context.get("zip_code"),
        "location_mode": location_context.get("location_mode"),
    })
    user_context = {
        "household": session.get("household"),
        "preparedness": session.get("preparedness"),
        "special_needs": session.get("special_needs"),
    }
    structured_results = build_hazard_results(
        hazards,
        location_result,
        location_context.get("zip_risk_snapshot") or {},
        user_context=user_context,
    )
    hazards_by_slug = {hazard.get("slug"): hazard for hazard in hazards}
    merged_hazards = []
    used_slugs = set()

    for result in structured_results:
        hazard = hazards_by_slug.get(result.hazard_id)
        if not hazard:
            continue
        merged_hazards.append(merge_structured_result(hazard, result))
        used_slugs.add(result.hazard_id)

    for hazard in sorted(hazards, key=lambda item: item.get("priority_score", 0), reverse=True):
        if hazard.get("slug") not in used_slugs:
            merged_hazards.append(hazard)

    return merged_hazards


def get_hazard_by_name(name, user=None, location_context=None):
    lookup_value = slugify_hazard_name(name)
    for hazard in get_all_hazards(user, location_context):
        if hazard.get("slug") == lookup_value or (hazard.get("name", "").strip().lower() == (name or "").strip().lower()):
            return hazard
    return None


def get_top_hazards_sorted_by_priority(user=None, location_context=None):
    return get_all_hazards(user, location_context)[:3]


def normalize_plan_name(value):
    return (value or "").strip().lower().replace(" ", "_").replace("-", "_")


def get_local_plan_for_context(location_context):
    city = (location_context or {}).get("city") or ""
    city_key = normalize_plan_name(city)
    if not city_key:
        return {}
    for plan in load_local_plans():
        if normalize_plan_name(plan.get("name")) == city_key:
            return plan
    return {}


def get_additional_local_hazards(location_context, shown_hazards=None, limit=6):
    plan = get_local_plan_for_context(location_context)
    if not plan or plan.get("review_status") not in {"reviewed", "draft_reviewed"}:
        return []

    shown_hazards = {
        slugify_hazard_name(item.get("slug") or item.get("hazard_id") or item.get("name"))
        for item in shown_hazards or []
    }
    excluded = CORE_DEMO_HAZARDS | shown_hazards
    city = plan.get("name") or (location_context or {}).get("city") or "this jurisdiction"
    priority_order = ADDITIONAL_HAZARD_PRIORITY.get(city, [])
    priority_index = {hazard: index for index, hazard in enumerate(priority_order)}

    candidates = []
    for hazard_id in plan.get("hazards", []):
        normalized = slugify_hazard_name(hazard_id).replace("-", "_")
        if normalized in excluded or normalized == "all":
            continue
        label = ADDITIONAL_HAZARD_LABELS.get(normalized, hazard_id.replace("_", " ").title())
        candidates.append({
            "hazard_id": normalized,
            "label": label,
            "scope_label": "Local plan context",
            "data_status_label": "Not checked by GIS",
            "review_status": plan.get("review_status", "draft"),
            "plan_name": plan.get("plan_name"),
            "plan_url": plan.get("url"),
            "why_shown": f"{label} is included because {city}'s reviewed local plan lists it as a hazard or planning concern.",
            "limitations": "This is not one of the current address-level GIS checks. It should be treated as local source context until an official layer or reviewed location rule is added.",
            "next_step": ADDITIONAL_HAZARD_CONTEXT.get(normalized, "Review the local plan source and official guidance before making address-level claims."),
            "priority_rank": priority_index.get(normalized, 999),
        })

    candidates.sort(key=lambda item: (item["priority_rank"], item["label"]))
    return candidates[:limit]


def get_data_sources_context():
    """Centralized source metadata so it can become dynamic later."""
    return {
        "last_updated": "March 2026",
        "data_sources": DATA_SOURCES,
    }


@app.context_processor
def inject_data_sources_context():
    return get_data_sources_context()

def is_valid_coordinate(lat, lon):
    return (
        lat is not None and
        lon is not None and
        COVERAGE_BOUNDS["min_lat"] <= lat <= COVERAGE_BOUNDS["max_lat"] and
        COVERAGE_BOUNDS["min_lon"] <= lon <= COVERAGE_BOUNDS["max_lon"]
    )


def load_risk_data():
    try:
        risk_df = pd.read_csv(os.path.join(BASE_DIR, "static", "zip_risk_scores.csv"), dtype={"ZIP": str})
        return risk_df.set_index("ZIP").to_dict(orient="index")
    except Exception:
        logger.exception("Risk CSV failed to load.")
        return {}


def safe_render(template_name, **context):
    base_context = get_data_sources_context()
    base_context.update(context)
    try:
        return render_template(template_name, **base_context)
    except TemplateNotFound:
        logger.exception("Template %s not found.", template_name)
        return render_template(
            "error.html",
            title="Page Unavailable",
            error_heading="This page is temporarily unavailable",
            error_message="Something went wrong, but your saved location and profile are still available.",
            helper_message="This page is temporarily unavailable. You can keep using the rest of the app.",
            **base_context
        ), 200


def get_default_hazards():
    return [
        ("Earthquake", 0, RISK_DATA_WARNING),
        ("Flood", 0, RISK_DATA_WARNING),
        ("Wildfire", 0, RISK_DATA_WARNING),
    ]


def get_default_recommended_actions():
    return {
        "hazard_name": "General preparedness",
        "risk_level": "General",
        "score": 0.0,
        "explanation": RISK_DATA_WARNING,
        "steps": get_action_items(GENERAL_CHECKLIST_IDS[:4]),
    }


zip_risk_data = load_risk_data()
hazard_dataset = load_hazard_dataset()

def get_zip_from_coordinates(lat, lon, zip_geojson_file="static/zipbound.geojson"):
    """
    Return the correct ZIP code based on coordinates using your ZIP boundaries.
    """
    if not is_valid_coordinate(lat, lon):
        return None

    point = Point(lon, lat)

    try:
        with open(os.path.join(BASE_DIR, zip_geojson_file), "r", encoding="utf-8") as f:
            zip_data = json.load(f)

        for feature in zip_data.get("features", []):
            geom = feature.get("geometry")
            props = feature.get("properties", {})
            if not geom:
                continue
            polygon = shape(geom)
            if polygon.contains(point):
                for field in ["ZCTA5CE10", "ZIP", "ZIPCODE", "zip_code", "ZIP_CODE"]:
                    zip_code = props.get(field)
                    if zip_code:
                        return str(zip_code)
    except Exception:
        logger.exception("ZIP boundary lookup failed for coordinates (%s, %s).", lat, lon)
    return None
# --- Geocoding fallback ---
def geocode_zip(zip_code):
    geolocator = Nominatim(user_agent="disaster_app")
    try:
        location = geolocator.geocode({"postalcode": zip_code, "country": "US"})
        if location and is_valid_coordinate(location.latitude, location.longitude):
            return (location.latitude, location.longitude)
    except Exception:
        logger.exception("ZIP geocoding failed for %s.", zip_code)
    return (DEFAULT_LAT, DEFAULT_LON)
def geocode_address(address_query):
    """
    Convert address to coordinates and ZIP code
    Returns: (lat, lon, zip_code, formatted_address) or None
    """
    geolocator = Nominatim(user_agent="disaster_app", timeout=10)
    
    try:
        # Check if input looks like coordinates (lat, lon)
        coord_pattern = r'^(-?\d+\.?\d*),\s*(-?\d+\.?\d*)(?:,.*)?$'
        coord_match = re.match(coord_pattern, address_query.strip())
        
        if coord_match:
            # Handle coordinate input with reverse geocoding
            lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
            
            # Verify coordinates are in reasonable range for Bay Area
            if not is_valid_coordinate(lat, lon):
                logger.info("Coordinates outside coverage bounds: %s, %s", lat, lon)
                return None
            
            # Reverse geocode to get address and ZIP
            location = geolocator.reverse((lat, lon), exactly_one=True)
            if location:
                zip_code = extract_zip_from_address(location.address)
                return (lat, lon, zip_code, location.address)
            else:
                return None
        
        # Handle regular address input
        clean_address = clean_address_input(address_query)
        
        # Try geocoding with different variations
        location = None
        
        # First try: exact input with Alameda County
        location = geolocator.geocode(clean_address + ", Alameda County, CA, USA")
        
        # Second try: without county specification
        if not location:
            location = geolocator.geocode(clean_address + ", CA, USA")
        
        # Third try: with Oakland area fallback
        if not location and "oakland" not in clean_address.lower():
            location = geolocator.geocode(clean_address + ", Oakland, CA, USA")
        
        # Fourth try: just the address as-is
        if not location:
            location = geolocator.geocode(clean_address)
            
        if location:
            lat, lon = location.latitude, location.longitude
            if not is_valid_coordinate(lat, lon):
                logger.info("Address geocoded outside coverage bounds: %s, %s", lat, lon)
                return None

            # Get correct ZIP from your GeoJSON boundaries
            zip_code = get_zip_from_coordinates(lat, lon)

            # If not found, fall back to Nominatim
            if not zip_code:
                zip_code = extract_zip_from_address(location.address)

            return (
                lat,
                lon,
                zip_code,
                location.address
            )
    
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.exception("Geocoding service error.")
    except ValueError as e:
        logger.exception("Coordinate parsing error.")
    except Exception:
        logger.exception("Unexpected address geocoding error.")
        
    return None

def clean_address_input(address):
    """Clean and standardize address input"""
    # Remove extra spaces and normalize
    address = re.sub(r'\s+', ' ', address.strip())
    
    # Expand common abbreviations
    address = re.sub(r'\bst\b', 'street', address, flags=re.IGNORECASE)
    address = re.sub(r'\bave\b', 'avenue', address, flags=re.IGNORECASE)
    address = re.sub(r'\bblvd\b', 'boulevard', address, flags=re.IGNORECASE)
    address = re.sub(r'\bdr\b', 'drive', address, flags=re.IGNORECASE)
    address = re.sub(r'\brd\b', 'road', address, flags=re.IGNORECASE)
    address = re.sub(r'\bct\b', 'court', address, flags=re.IGNORECASE)
    address = re.sub(r'\bpl\b', 'place', address, flags=re.IGNORECASE)
    
    return address

def extract_zip_from_address(full_address):
    """Extract ZIP code from geocoded address"""
    if not full_address:
        return None
    # Look for 5-digit ZIP code pattern
    zip_match = re.search(r'\b(\d{5})\b', full_address)
    if zip_match:
        return zip_match.group(1)
    return None

@app.route("/form", methods=["POST"])
def process_form():
    try:
        zip_code = request.form.get("zip_code", "").strip()
        address = request.form.get("address", "").strip()

        final_zip = None
        lat = lon = None
        formatted_address = None
        location_mode = ""

        if zip_code:
            if not zip_code.isdigit() or len(zip_code) != 5:
                session["form_error"] = "Invalid ZIP code. Please enter a 5-digit number."
                session["form_data"] = request.form.to_dict()
                return redirect(url_for("home"))
            if zip_risk_data and zip_code not in zip_risk_data:
                session["form_error"] = f"ZIP code {zip_code} is outside our coverage area."
                session["form_data"] = request.form.to_dict()
                return redirect(url_for("home"))
            final_zip = zip_code
            lat, lon = geocode_zip(zip_code)
            formatted_address = None
            location_mode = "zip"

        elif address:
            if len(address) < 5 or re.search(r'[^a-zA-Z0-9\s,.-]', address):
                session["form_error"] = "Invalid address. Please enter a proper street address."
                session["form_data"] = request.form.to_dict()
                return redirect(url_for("home"))

            result = geocode_address(address)
            if not result:
                session["form_error"] = "Could not find a valid address in the supported service area."
                session["form_data"] = request.form.to_dict()
                return redirect(url_for("home"))

            lat, lon, zip_from_address, formatted_address = result
            zip_from_address = (zip_from_address or "").split("-")[0]

            if not zip_from_address:
                session["form_error"] = "We found the location, but could not verify a supported ZIP code."
                session["form_data"] = request.form.to_dict()
                return redirect(url_for("home"))

            if zip_risk_data and zip_from_address not in zip_risk_data:
                session["form_error"] = f"The address ZIP ({zip_from_address}) is not covered."
                session["form_data"] = request.form.to_dict()
                return redirect(url_for("home"))

            final_zip = zip_from_address
            location_mode = "address"
        else:
            session["form_error"] = "Please enter a ZIP code or address."
            session["form_data"] = request.form.to_dict()
            return redirect(url_for("home"))

        household = request.form.get("household")
        preparedness = request.form.get("preparedness")
        special_needs = request.form.get("special_needs", "")

        if not household or not preparedness:
            return safe_render(
                "home.html",
                error="Please fill in all required fields.",
                form_data=request.form.to_dict()
            )

        if not is_valid_coordinate(lat, lon):
            lat, lon = geocode_zip(final_zip) if final_zip else (DEFAULT_LAT, DEFAULT_LON)

        session["zip_code"] = final_zip
        session["lat"] = lat
        session["lon"] = lon
        session["address"] = formatted_address
        session["input_address"] = address or zip_code
        session["county"] = "Alameda County" if final_zip else ""
        session["city"] = ""
        session[LOCATION_MODE_SESSION_KEY] = location_mode
        session["household"] = household
        session["special_needs"] = special_needs
        session["preparedness"] = preparedness

        for hazard in ["wildfire", "flood", "earthquake"]:
            session.pop(f"chat_{hazard}", None)
            session.pop(f"meta_{hazard}", None)

        return redirect(url_for("risk_summary"))
    except Exception:
        logger.exception("Form processing failed.")
        session["form_error"] = "We could not process that request, but your saved data is still intact. Please try again."
        session["form_data"] = request.form.to_dict()
        return redirect(url_for("home"))


# Remove the enhanced_form route since we're consolidating into one route

# AP
# Error handling route for better user experience
@app.errorhandler(404)
def not_found_error(error):
    logger.warning("404: %s", request.path)
    return safe_render(
        "error.html",
        title="Page Not Found",
        error_heading="Page not found",
        error_message="That page does not exist, but your saved location and profile are still available.",
        helper_message="This page does not exist, but you can return home and continue using the app."
    ), 404

@app.errorhandler(500)
def internal_error(error):
    logger.exception("500 error on %s", request.path)
    return safe_render(
        "error.html",
        title="Temporary Error",
        error_heading="Something went wrong",
        error_message="Something went wrong, but your saved location and profile are still available.",
        helper_message="You can return home or continue to your risk summary."
    ), 500


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.exception("Unhandled exception on %s", request.path)
    return safe_render(
        "error.html",
        title="Temporary Error",
        error_heading="Something went wrong",
        error_message="Something went wrong, but your saved location and profile are still available.",
        helper_message="The app has fallen back to a safe page so you can keep going."
    ), 500


@app.route("/search-address", methods=["POST"])
def search_address():
    """Handle address search and convert to ZIP code"""
    try:
        address_query = request.form.get("address", "").strip()

        if not address_query:
            return jsonify({"error": "Address is required"}), 400

        result = geocode_address(address_query)

        if not result:
            return jsonify({
                "error": "Address not found or outside service area",
                "suggestion": "Please try a more specific address or use ZIP code search"
            }), 404

        lat, lon, zip_code, formatted_address = result

        if zip_code and zip_risk_data and zip_code not in zip_risk_data:
            return jsonify({
                "error": f"ZIP code {zip_code} is outside our coverage area",
                "found_address": formatted_address,
                "suggestion": "This tool covers Alameda County ZIP codes"
            }), 404

        return jsonify({
            "success": True,
            "zip_code": zip_code,
            "coordinates": [lat, lon],
            "formatted_address": formatted_address,
            "message": f"Found address in ZIP {zip_code}"
        })
    except Exception:
        logger.exception("Address search failed.")
        return jsonify({
            "error": "Address lookup is temporarily unavailable",
            "suggestion": "Please try ZIP code search instead"
        }), 503


# API endpoint for address suggestions (optional autocomplete)
@app.route("/api/address-suggestions")
def address_suggestions():
    """Simple address validation/suggestions"""
    query = request.args.get("q", "").strip()
    
    if len(query) < 3:
        return jsonify([])
    
    # Basic suggestions based on common Alameda County cities
    alameda_cities = [
        "Oakland", "Berkeley", "Fremont", "Hayward", "Alameda", 
        "San Leandro", "Union City", "Newark", "Dublin", "Pleasanton",
        "Livermore", "Castro Valley", "San Lorenzo", "Emeryville"
    ]
    
    suggestions = []
    for city in alameda_cities:
        if query.lower() in city.lower():
            suggestions.append(f"{query}, {city}, CA")
    
    return jsonify(suggestions[:5])
def get_risk_level(score):
    """Convert numeric risk score to text level"""
    if score >= 7:
        return "High"
    elif score >= 4:
        return "Moderate"
    else:
        return "Low"


ACTION_ITEM_LIBRARY = {
    "bag": {"id": "bag", "label": "Put water, a flashlight, and a charger in a bag."},
    "alerts": {"id": "alerts", "label": "Turn on local emergency alerts for your area."},
    "contacts": {"id": "contacts", "label": "Save emergency contacts on every phone."},
    "water": {"id": "water", "label": "Store at least 3 days of water for your home."},
    "evacuation-routes": {"id": "evacuation-routes", "label": "Plan 2 evacuation routes from your area."},
    "defensible-space": {"id": "defensible-space", "label": "Clear dry leaves and brush near your home."},
    "high-ground": {"id": "high-ground", "label": "Pick the fastest route to higher ground."},
    "move-valuables": {"id": "move-valuables", "label": "Move valuables and documents above floor level."},
    "avoid-floodwater": {"id": "avoid-floodwater", "label": "Stay out of floodwater and never drive through it."},
    "secure-furniture": {"id": "secure-furniture", "label": "Secure heavy furniture and TVs to the wall."},
    "safe-spots": {"id": "safe-spots", "label": "Choose a safe spot in each room away from windows."},
    "drill": {"id": "drill", "label": "Practice drop, cover, and hold on once this week."},
}

GENERAL_CHECKLIST_IDS = ["water", "bag", "evacuation-routes", "alerts", "contacts"]

HAZARD_ACTION_IDS = {
    "wildfire": {
        "high": ["bag", "evacuation-routes", "alerts", "defensible-space"],
        "moderate": ["bag", "alerts", "evacuation-routes", "defensible-space"],
        "low": ["bag", "alerts", "contacts", "water"],
    },
    "flood": {
        "high": ["move-valuables", "high-ground", "water", "avoid-floodwater"],
        "moderate": ["move-valuables", "high-ground", "water", "avoid-floodwater"],
        "low": ["water", "contacts", "alerts", "high-ground"],
    },
    "earthquake": {
        "high": ["secure-furniture", "safe-spots", "bag", "drill"],
        "moderate": ["secure-furniture", "safe-spots", "bag", "drill"],
        "low": ["bag", "safe-spots", "contacts", "drill"],
    },
}


def get_action_items(item_ids):
    return [ACTION_ITEM_LIBRARY[item_id] for item_id in item_ids if item_id in ACTION_ITEM_LIBRARY]


def get_action_steps(hazard_type, risk_level):
    """Return deterministic preparedness steps for a hazard and risk level."""
    hazard_key = (hazard_type or "").strip().lower()
    normalized_level = (risk_level or "").strip().lower()
    if normalized_level == "very high":
        normalized_level = "high"
    hazard_steps = HAZARD_ACTION_IDS.get(hazard_key, {})
    action_ids = hazard_steps.get(normalized_level) or GENERAL_CHECKLIST_IDS[:4]
    return get_action_items(action_ids)


def build_hazard_fallback_response(hazard, zip_code, score, explanation):
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = 0
    risk_level = get_risk_level(numeric_score)
    steps = get_action_steps(hazard, risk_level)
    return (
        f"Your {hazard.title()} risk for ZIP {zip_code} is {numeric_score:.1f}/10. "
        f"{explanation if explanation else RISK_DATA_WARNING}\n\n"
        "Recommended next steps:\n- " + "\n- ".join(step["label"] for step in steps)
    )

def load_geojson_file(filename):
    """Helper function to load geojson files safely"""
    filepath = os.path.join(BASE_DIR, "static", filename)
    logger.info("Loading GeoJSON from %s", filepath)

    if not os.path.exists(filepath):
        logger.warning("GeoJSON file does not exist: %s", filepath)
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.exception("Error decoding GeoJSON file %s", filepath)
        return None
    except Exception:
        logger.exception("Unexpected error loading GeoJSON file %s", filepath)
        return None


def geojson_feature_bounds(feature):
    try:
        return shape(feature.get("geometry", {})).bounds
    except Exception:
        return None


def bounds_intersect(a, b):
    if not a or not b:
        return False
    minx, miny, maxx, maxy = a
    other_minx, other_miny, other_maxx, other_maxy = b
    return not (maxx < other_minx or minx > other_maxx or maxy < other_miny or miny > other_maxy)


def get_geojson_bounds(data):
    bounds = None
    for feature in (data or {}).get("features", []):
        feature_bounds = geojson_feature_bounds(feature)
        if not feature_bounds:
            continue
        if bounds is None:
            bounds = feature_bounds
        else:
            bounds = (
                min(bounds[0], feature_bounds[0]),
                min(bounds[1], feature_bounds[1]),
                max(bounds[2], feature_bounds[2]),
                max(bounds[3], feature_bounds[3]),
            )
    return bounds


def expand_bounds(bounds, padding=0.025):
    if not bounds:
        return None
    return (
        bounds[0] - padding,
        bounds[1] - padding,
        bounds[2] + padding,
        bounds[3] + padding,
    )


def build_filter_bounds(zip_code=None, lat=None, lon=None):
    bounds = None
    if zip_code:
        zip_boundary = get_zip_boundary(str(zip_code))
        bounds = get_geojson_bounds(zip_boundary)

    try:
        if lat is not None and lon is not None:
            lat_value = float(lat)
            lon_value = float(lon)
            point_bounds = (lon_value - 0.045, lat_value - 0.045, lon_value + 0.045, lat_value + 0.045)
            bounds = point_bounds if bounds is None else (
                min(bounds[0], point_bounds[0]),
                min(bounds[1], point_bounds[1]),
                max(bounds[2], point_bounds[2]),
                max(bounds[3], point_bounds[3]),
            )
    except (TypeError, ValueError):
        pass

    return expand_bounds(bounds, padding=0.02)


def filter_geojson_by_bounds(data, filter_bounds, max_features=400):
    if not data:
        return {"type": "FeatureCollection", "features": []}
    if not filter_bounds:
        return {
            **data,
            "features": data.get("features", [])[:max_features],
        }
    filtered_features = [
        feature for feature in data.get("features", [])
        if bounds_intersect(geojson_feature_bounds(feature), filter_bounds)
    ]
    return {
        "type": "FeatureCollection",
        "features": filtered_features[:max_features],
    }

def get_zip_boundary(zip_code):
    """Get ZIP boundary from zipbound.geojson"""
    try:
        zipbound_data = load_geojson_file("zipbound.geojson")
        if zipbound_data:
            for feature in zipbound_data.get('features', []):
                props = feature.get('properties', {})
                zip_fields = ['ZCTA5CE10', 'ZIP', 'ZIPCODE', 'zip_code', 'ZIP_CODE']
                for field in zip_fields:
                    if props.get(field) == zip_code:
                        return {
                            "type": "FeatureCollection",
                            "features": [feature]
                        }
    except Exception:
        logger.exception("ZIP boundary lookup failed for %s", zip_code)

    lat, lon = DEFAULT_LAT, DEFAULT_LON
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"ZIP": zip_code, "fallback": True},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon - 0.02, lat - 0.02],
                        [lon + 0.02, lat - 0.02],
                        [lon + 0.02, lat + 0.02],
                        [lon - 0.02, lat + 0.02],
                        [lon - 0.02, lat - 0.02]
                    ]]
                }
            }
        ]
    }

# --- Home Page ---
@app.route("/")
def home():
    error = session.pop("form_error", None)
    form_data = session.pop("form_data", None)
    location_context = get_session_location_context()
    return safe_render("home.html", error=error, form_data=form_data, location_context=location_context)


@app.route("/hazards/profile", methods=["POST"])
def save_hazard_profile():
    profile = normalize_user_profile(request.form)
    session[HAZARD_PROFILE_SESSION_KEY] = profile

    next_url = request.form.get("next_url", "").strip()
    if next_url:
        return redirect(next_url)
    return redirect(url_for("hazards_dashboard"))


@app.route("/hazards")
def hazards_dashboard():
    user_profile = get_saved_hazard_profile()
    location_context = get_session_location_context()
    location_context["zip_risk_snapshot"] = get_zip_risk_snapshot(location_context.get("zip_code"))
    hazards = get_all_hazards(user_profile, location_context)
    top_hazards = get_top_hazards_sorted_by_priority(user_profile, location_context)

    return safe_render(
        "hazards_dashboard.html",
        hazards=hazards,
        top_hazards=top_hazards,
        system_insights=get_system_insights(),
        user_profile=user_profile,
        location_options=HAZARD_LOCATION_OPTIONS,
        location_context=location_context,
    )


@app.route("/hazards/<name>")
def hazard_detail(name):
    user_profile = get_saved_hazard_profile()
    location_context = get_session_location_context()
    location_context["zip_risk_snapshot"] = get_zip_risk_snapshot(location_context.get("zip_code"))
    hazard = get_hazard_by_name(name, user_profile, location_context)
    if not hazard:
        return safe_render(
            "error.html",
            title="Hazard Not Found",
            error_heading="Hazard not found",
            error_message="We could not find that hazard profile, but the rest of the app is still available.",
            helper_message="Return to the hazard dashboard to browse the ranked hazard list."
        ), 404

    return safe_render(
        "hazard_detail.html",
        hazard=hazard,
        top_hazards=get_top_hazards_sorted_by_priority(user_profile, location_context),
        user_profile=user_profile,
        location_options=HAZARD_LOCATION_OPTIONS,
        location_context=location_context,
    )



# --- Optional: Redirect old form GET requests ---
@app.route("/form", methods=["GET"])
def redirect_form():
    """Redirect old form page requests to home"""
    return redirect(url_for("home"))




# --- Risk Summary Page ---
@app.route("/risk_summary")
def risk_summary():
    zip_code = session.get("zip_code")
    if not zip_code:
        return redirect(url_for("home"))
    warning_message = None
    location_context = get_session_location_context()
    location_context["zip_risk_snapshot"] = get_zip_risk_snapshot(location_context.get("zip_code"))
    all_structured_hazards = get_all_hazards(get_saved_hazard_profile(), location_context)
    core_hazards = [
        hazard for hazard in all_structured_hazards
        if hazard.get("slug") in {"flood", "wildfire", "earthquake"}
    ]
    structured_hazards = core_hazards[:3] if core_hazards else all_structured_hazards[:3]
    additional_local_hazards = get_additional_local_hazards(location_context, structured_hazards)

    try:
        data = zip_risk_data.get(zip_code, {})
        if data:
            hazards = [
                ("Earthquake", data.get("Earthquake_Risk_Score", 0), data.get("Earthquake_Risk_Explanation", "")),
                ("Flood", data.get("Flood_Risk_Score", 0), data.get("Flood_Risk_Explanation", "")),
                ("Wildfire", data.get("Wildfire_Risk_Score", 0), data.get("Wildfire_Risk_Explanation", ""))
            ]
            hazards_sorted = sorted(hazards, key=lambda x: -float(x[1]))
            highest_hazard_name, highest_hazard_score, highest_hazard_explanation = hazards_sorted[0]
            highest_risk_level = get_risk_level(float(highest_hazard_score))
            recommended_actions = {
                "hazard_name": highest_hazard_name,
                "risk_level": highest_risk_level,
                "score": float(highest_hazard_score),
                "explanation": highest_hazard_explanation,
                "steps": get_action_steps(highest_hazard_name, highest_risk_level),
            }
            if structured_hazards:
                primary_structured = structured_hazards[0]
                recommended_actions = {
                    "hazard_name": primary_structured.get("name", primary_structured.get("label", highest_hazard_name)),
                    "risk_level": primary_structured.get("exposure_level", highest_risk_level).title(),
                    "score": primary_structured.get("structured_result", {}).get("legacy_score") or float(highest_hazard_score),
                    "explanation": primary_structured.get("why_shown", highest_hazard_explanation),
                    "steps": [
                        {"id": item.get("id"), "label": item.get("label")}
                        for item in primary_structured.get("recommended_actions", [])
                    ] or get_action_steps(highest_hazard_name, highest_risk_level),
                }
        else:
            hazards_sorted = get_default_hazards()
            recommended_actions = get_default_recommended_actions()
            warning_message = RISK_DATA_WARNING
    except Exception:
        logger.exception("Risk summary failed for ZIP %s", zip_code)
        hazards_sorted = get_default_hazards()
        recommended_actions = get_default_recommended_actions()
        warning_message = RISK_DATA_WARNING

    start_here_steps = recommended_actions["steps"][:4]
    all_checklist_items = get_action_items(
        list(dict.fromkeys(GENERAL_CHECKLIST_IDS + [item["id"] for item in recommended_actions["steps"]]))
    )
    start_here_ids = {item["id"] for item in start_here_steps}
    checklist_items = [item for item in all_checklist_items if item["id"] not in start_here_ids]

    return safe_render(
        "risk_summary.html",
        zip_code=zip_code,
        location_context=location_context,
        structured_hazards=structured_hazards,
        additional_local_hazards=additional_local_hazards,
        hazards=hazards_sorted,
        recommended_actions=recommended_actions,
        start_here_steps=start_here_steps,
        checklist_items=checklist_items,
        checklist_total_count=len(all_checklist_items),
        warning_message=warning_message,
        oakland_hazard_ready=is_oakland_hazard_context(session.get("zip_code"), session.get("address")),
    )

# --- Unified Hazard Map ---
@app.route("/map", methods=["GET", "POST"])
def map():
    """Unified hazard map showing all risks with toggleable layers"""
    zip_code = session.get("zip_code", "94601")
    address = session.get("address")
    location_context = get_session_location_context()
    location_context["zip_risk_snapshot"] = get_zip_risk_snapshot(location_context.get("zip_code"))
    map_hazards = [
        hazard for hazard in get_all_hazards(get_saved_hazard_profile(), location_context)
        if hazard.get("slug") in {"wildfire", "flood", "earthquake"}
    ]
    map_notice = None

    if request.method == "POST":
        return jsonify({
            "ok": True,
            "message": "Interactive assistant is temporarily limited. You can still use the map and risk summary safely."
        })

    try:
        lat = session.get("lat")
        lon = session.get("lon")
        if not is_valid_coordinate(lat, lon):
            lat, lon = DEFAULT_LAT, DEFAULT_LON

        data = zip_risk_data.get(zip_code, {})
        risk_scores = {
            'wildfire': {
                'score': float(data.get("Wildfire_Risk_Score", 0) or 0),
                'explanation': data.get("Wildfire_Risk_Explanation", RISK_DATA_WARNING)
            },
            'earthquake': {
                'score': float(data.get("Earthquake_Risk_Score", 0) or 0),
                'explanation': data.get("Earthquake_Risk_Explanation", RISK_DATA_WARNING)
            },
            'flood': {
                'score': float(data.get("Flood_Risk_Score", 0) or 0),
                'explanation': data.get("Flood_Risk_Explanation", RISK_DATA_WARNING)
            }
        }
        if not data:
            map_notice = RISK_DATA_WARNING
    except Exception:
        logger.exception("Map route failed for ZIP %s", zip_code)
        lat, lon = DEFAULT_LAT, DEFAULT_LON
        risk_scores = {
            'wildfire': {'score': 0.0, 'explanation': RISK_DATA_WARNING},
            'earthquake': {'score': 0.0, 'explanation': RISK_DATA_WARNING},
            'flood': {'score': 0.0, 'explanation': RISK_DATA_WARNING}
        }
        map_notice = "Map data temporarily unavailable. You can still view your risk and preparedness steps below."

    return safe_render(
        "map.html",
        zip_code=zip_code,
        risk_scores=risk_scores,
        user_lat=lat,
        user_lon=lon,
        user_address=address or f"ZIP {zip_code}",
        map_notice=map_notice,
        has_precise_location=location_context.get("has_precise_location", False),
        location_mode=location_context.get("location_mode", "zip"),
        location_context=location_context,
        structured_hazards=map_hazards,
        oakland_hazard_ready=is_oakland_hazard_context(zip_code, address),
    )
# ---  Page ---
@app.route("/about")
def about():
    return safe_render("about.html")


@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory(BASE_DIR, 'sitemap.xml')


@app.route("/robots.txt")
def robots():
    return send_from_directory(BASE_DIR, "robots.txt")


# --- Resources Page ---
@app.route("/resources")
def resources():
    return safe_render("resources.html")


@app.route("/sources")
def sources():
    return safe_render(
        "sources.html",
        sources=source_records_payload(),
        hazards=load_hazard_registry(),
        jurisdictions=load_jurisdictions(),
        local_plans=load_local_plans(),
    )

# --- API Endpoints ---


@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "app": "StayReady",
        "hazard_engine": "structured",
        "required_gis_slice": "flood_address_point",
        "source_count": len(source_records_payload()),
        "local_plan_count": len(load_local_plans()),
        "resident_guidance_count": len(load_resident_guidance_chunks()),
    })


@app.route("/api/sources")
def api_sources():
    return jsonify({
        "sources": source_records_payload(),
        "hazards": load_hazard_registry(),
        "jurisdictions": load_jurisdictions(),
        "local_plans": load_local_plans(),
        "resident_guidance_count": len(load_resident_guidance_chunks()),
    })


@app.route("/api/local-plans")
def api_local_plans():
    return jsonify({
        "plans": load_local_plans(),
        "count": len(load_local_plans()),
        "review_note": "City-specific hazard claims are only used when the plan/chunk has reviewed or draft-reviewed status.",
    })


@app.route("/api/hazards")
def api_hazards():
    location_context = get_session_location_context()
    location_context["zip_risk_snapshot"] = get_zip_risk_snapshot(location_context.get("zip_code"))
    hazards = get_all_hazards(get_saved_hazard_profile(), location_context)
    return jsonify({
        "location": location_context.get("location_result"),
        "hazards": [hazard.get("structured_result", hazard) for hazard in hazards],
        "additional_local_hazards": get_additional_local_hazards(location_context, hazards),
    })


@app.route("/api/hazards/<name>")
def api_hazard(name):
    location_context = get_session_location_context()
    location_context["zip_risk_snapshot"] = get_zip_risk_snapshot(location_context.get("zip_code"))
    hazard = get_hazard_by_name(name, get_saved_hazard_profile(), location_context)
    if not hazard:
        return jsonify({"error": "Hazard not found"}), 404
    return jsonify(hazard.get("structured_result", hazard))


@app.route("/api/top-risks")
def api_top_risks():
    location_context = get_session_location_context()
    location_context["zip_risk_snapshot"] = get_zip_risk_snapshot(location_context.get("zip_code"))
    hazards = get_top_hazards_sorted_by_priority(get_saved_hazard_profile(), location_context)
    return jsonify({
        "location": location_context.get("location_result"),
        "hazards": [hazard.get("structured_result", hazard) for hazard in hazards],
        "additional_local_hazards": get_additional_local_hazards(location_context, hazards),
    })


@app.route("/api/explain-hazard", methods=["POST"])
def api_explain_hazard():
    payload = request.get_json(silent=True) or {}
    hazard_name = (payload.get("hazard_type") or payload.get("hazard") or "").strip().lower()
    if not hazard_name:
        return jsonify({"error": "hazard_type is required"}), 400

    location_context = get_session_location_context()
    location_context["zip_risk_snapshot"] = get_zip_risk_snapshot(location_context.get("zip_code"))
    hazard = get_hazard_by_name(hazard_name, get_saved_hazard_profile(), location_context)
    if not hazard:
        return jsonify({"error": "Hazard not found"}), 404

    result = hazard.get("structured_result", {})
    chunks = retrieve_chunks(
        result.get("hazard_type", hazard_name),
        jurisdiction=location_context.get("county") or "Alameda County",
        scope=result.get("scope", ""),
    )
    return jsonify({
        "hazard": result,
        "explanation": {
            "plain_english_explanation": result.get("why_shown", ""),
            "why_shown": result.get("why_shown", ""),
            "recommended_actions": result.get("recommended_actions", []),
            "recovery_questions": result.get("recovery_questions", []),
            "limitations": result.get("limitations", []),
            "citations": result.get("sources", []),
            "confidence": result.get("confidence", "needs_review"),
            "review_status": result.get("review_status", "draft"),
        },
        "supporting_chunks": [chunk.model_dump() for chunk in chunks],
        "decision_note": "Explanation only. This endpoint does not change hazard scope, basis, GIS status, or exposure level.",
    })

# Live Earthquake API
@app.route("/api/live-earthquakes")
def api_live_earthquakes():
    """API endpoint for recent earthquake data with normal empty states."""
    scope = request.args.get("scope", "bay_area").strip().lower()
    days = request.args.get("days", "7").strip()
    feed = "all_week" if days == "7" else "all_day"
    url = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{feed}.geojson"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.exception("Unable to load live earthquake data.")
        return jsonify({
            "type": "FeatureCollection",
            "features": [],
            "source": "USGS Earthquake Hazards Program",
            "feature_count": 0,
            "filtered": True,
            "data_status": "unavailable",
            "message": "Recent earthquake feed is temporarily unavailable. You can still use the preparedness guidance below."
        })

    bounds_by_scope = {
        "alameda": {"min_lat": 37.4, "max_lat": 37.9, "min_lon": -122.4, "max_lon": -121.4},
        "bay_area": {"min_lat": 36.7, "max_lat": 38.7, "min_lon": -123.4, "max_lon": -120.8},
    }
    bounds = bounds_by_scope.get(scope, bounds_by_scope["bay_area"])
    
    features = []
    for feature in data.get("features", []):
        try:
            coords = feature["geometry"]["coordinates"]
            lon, lat = coords[0], coords[1]
            if (bounds["min_lat"] <= lat <= bounds["max_lat"] and
                bounds["min_lon"] <= lon <= bounds["max_lon"]):
                props = feature["properties"]
                props["depth"] = coords[2] if len(coords) > 2 else 0
                features.append(feature)
        except Exception:
            logger.exception("Skipping malformed earthquake feature.")

    message = (
        f"No recent earthquakes found in the selected {scope.replace('_', ' ')} window."
        if not features else
        f"Showing {len(features)} recent earthquakes from the USGS {feed.replace('_', ' ')} feed."
    )
    return jsonify({
        "type": "FeatureCollection",
        "features": features,
        "source": "USGS Earthquake Hazards Program",
        "feature_count": len(features),
        "filtered": True,
        "data_status": "no_recent_events" if not features else "checked",
        "message": message,
        "scope": scope,
        "days": 7 if feed == "all_week" else 1,
    })

# Wildfire zones API
@app.route("/api/wildfire-zones")
def api_wildfire_zones():
    """API endpoint for wildfire hazard zones"""
    data = load_geojson_file("FireHaz.geojson")
    if data:
        return jsonify(data)
    return jsonify({
        "type": "FeatureCollection",
        "features": [],
        "message": "Map data temporarily unavailable. You can still view your risk and preparedness steps below."
    })

# Flood zones API
@app.route("/api/flood-zones")
def api_flood_zones():
    """API endpoint for filtered flood hazard zones."""
    data = load_geojson_file("FldHaz.geojson")
    if data:
        zip_code = request.args.get("zip") or session.get("zip_code")
        lat = request.args.get("lat") or session.get("lat")
        lon = request.args.get("lon") or session.get("lon")
        filter_bounds = build_filter_bounds(zip_code=zip_code, lat=lat, lon=lon)
        filtered = filter_geojson_by_bounds(data, filter_bounds)
        feature_count = len(filtered.get("features", []))
        filtered.update({
            "source": "FEMA National Flood Hazard Layer",
            "feature_count": feature_count,
            "filtered": bool(filter_bounds),
            "data_status": "checked" if feature_count else "not_in_layer",
            "message": (
                f"Showing {feature_count} nearby flood polygons from the FEMA flood layer."
                if feature_count else
                "No FEMA flood polygons were returned for the selected map area. This does not prove there is no flood risk."
            ),
        })
        return jsonify(filtered)
    return jsonify({
        "type": "FeatureCollection",
        "features": [],
        "source": "FEMA National Flood Hazard Layer",
        "feature_count": 0,
        "filtered": True,
        "data_status": "unavailable",
        "message": "Flood map data is temporarily unavailable. You can still view your risk and preparedness steps below."
    })

# Fault lines API
@app.route("/api/fault-lines")
def api_fault_lines():
    """API endpoint for earthquake fault lines"""
    data = load_geojson_file("Fault_lines.Geojson")
    if data:
        return jsonify(data)
    return jsonify({
        "type": "FeatureCollection",
        "features": [],
        "message": "Map data temporarily unavailable. You can still view your risk and preparedness steps below."
    })

# ZIP boundary API
@app.route("/api/zip-boundary/<zip_code>")
def api_zip_boundary(zip_code):
    """API endpoint for ZIP code boundary"""
    boundary_data = get_zip_boundary(zip_code)
    if boundary_data:
        return jsonify(boundary_data)
    return jsonify({
        "type": "FeatureCollection",
        "features": [],
        "message": "Map data temporarily unavailable. You can still view your risk and preparedness steps below."
    })

# County boundary API
@app.route("/api/county-boundary")
def api_county_boundary():
    """API endpoint for Alameda County boundary"""
    data = load_geojson_file("countbound.geojson")
    if data:
        return jsonify(data)
    return jsonify({
        "type": "FeatureCollection",
        "features": [],
        "message": "Map data temporarily unavailable. You can still view your risk and preparedness steps below."
    })

# Risk assessment API
@app.route("/api/risk-assessment/<zip_code>")
def api_risk_assessment(zip_code):
    """API endpoint for comprehensive risk assessment"""
    data = zip_risk_data.get(zip_code)
    if not data:
        return jsonify({
            "zip_code": zip_code,
            "risks": {
                "wildfire": {"score": 0, "level": "Unknown", "explanation": RISK_DATA_WARNING},
                "earthquake": {"score": 0, "level": "Unknown", "explanation": RISK_DATA_WARNING},
                "flood": {"score": 0, "level": "Unknown", "explanation": RISK_DATA_WARNING}
            },
            "overall_risk": 0,
            "message": RISK_DATA_WARNING
        })
    try:
        assessment = {
            "zip_code": zip_code,
            "risks": {
                "wildfire": {
                    "score": float(data.get("Wildfire_Risk_Score", 0)),
                    "level": get_risk_level(float(data.get("Wildfire_Risk_Score", 0))),
                    "explanation": data.get("Wildfire_Risk_Explanation", ""),
                    "hazard_level": data.get("Wildfire_Hazard_Level", "Unknown")
                },
                "earthquake": {
                    "score": float(data.get("Earthquake_Risk_Score", 0)),
                    "level": get_risk_level(float(data.get("Earthquake_Risk_Score", 0))),
                    "explanation": data.get("Earthquake_Risk_Explanation", "")
                },
                "flood": {
                    "score": float(data.get("Flood_Risk_Score", 0)),
                    "level": get_risk_level(float(data.get("Flood_Risk_Score", 0))),
                    "explanation": data.get("Flood_Risk_Explanation", ""),
                    "control_district": data.get("Flood_Control_District", "Unknown")
                }
            },
            "overall_risk": max(
                float(data.get("Wildfire_Risk_Score", 0)),
                float(data.get("Earthquake_Risk_Score", 0)),
                float(data.get("Flood_Risk_Score", 0))
            )
        }
        return jsonify(assessment)
    except Exception:
        logger.exception("Risk assessment API failed for ZIP %s", zip_code)
        return jsonify({
            "zip_code": zip_code,
            "risks": {
                "wildfire": {"score": 0, "level": "Unknown", "explanation": RISK_DATA_WARNING},
                "earthquake": {"score": 0, "level": "Unknown", "explanation": RISK_DATA_WARNING},
                "flood": {"score": 0, "level": "Unknown", "explanation": RISK_DATA_WARNING}
            },
            "overall_risk": 0,
            "message": RISK_DATA_WARNING
        })

# --- Enhanced Shared Hazard Page Generator ---
def hazard_page(hazard, title, color):
    zip_code = session.get("zip_code", "94601")

    try:
        zip_geojson_data = get_zip_boundary(zip_code)
        zip_geojson = json.dumps(zip_geojson_data) if zip_geojson_data else "{}"
    except Exception:
        logger.exception("Hazard page ZIP boundary failed for %s", zip_code)
        zip_geojson = "{}"

    chat_key = f"chat_{hazard}"
    meta_key = f"meta_{hazard}"

    household_size = session.get("household", "Unknown")
    special_needs = session.get("special_needs", "None")
    preparedness_level = session.get("preparedness", "Unknown")

    inputs = (household_size, special_needs, preparedness_level)

    metadata = session.get(meta_key, {})
    chat = session.get(chat_key, [])

    regen_needed = (
        not metadata or
        metadata.get("zip_code") != zip_code or
        metadata.get("inputs") != inputs
    )

    if regen_needed:
        # Get comprehensive data for this ZIP code
        data = zip_risk_data.get(zip_code, {})
        
        # Extract risk scores and explanations
        earthquake_score = data.get("Earthquake_Risk_Score", "Unknown")
        earthquake_explanation = data.get("Earthquake_Risk_Explanation", "No data available")
        flood_score = data.get("Flood_Risk_Score", "Unknown")
        flood_explanation = data.get("Flood_Risk_Explanation", "No data available")
        wildfire_score = data.get("Wildfire_Risk_Score", "Unknown")
        wildfire_explanation = data.get("Wildfire_Risk_Explanation", "No data available")
        
        # Get specific hazard data
        if hazard == "wildfire":
            current_score = wildfire_score
            current_explanation = wildfire_explanation
            wildfire_hazard_level = data.get("Wildfire_Hazard_Level", "Unknown")
            custom_prompt = data.get("Wildfire_Chatbot_Prompt", "")
        elif hazard == "flood":
            current_score = flood_score
            current_explanation = flood_explanation
            flood_control_district = data.get("Flood_Control_District", "Unknown")
            custom_prompt = data.get("Flood_Chatbot_Prompt", "")
        elif hazard == "earthquake":
            current_score = earthquake_score
            current_explanation = earthquake_explanation
            custom_prompt = data.get("Earthquake_Chatbot_Prompt", "")
        
        # Build comprehensive context-aware prompt
        prompt_text = f"""You are a disaster preparedness assistant specializing in {hazard} safety for Alameda County residents. 

LOCATION CONTEXT:
- ZIP Code: {zip_code}
- {hazard.title()} Risk Score: {current_score}/10
- Risk Assessment: {current_explanation}
"""

        # Add hazard-specific context
        if hazard == "wildfire":
            prompt_text += f"- Wildfire Hazard Level: {wildfire_hazard_level}\n"
            prompt_text += f"- All Risk Scores - Wildfire: {wildfire_score}/10, Earthquake: {earthquake_score}/10, Flood: {flood_score}/10\n"
        elif hazard == "flood":
            prompt_text += f"- Flood Control District: {flood_control_district}\n"
            prompt_text += f"- All Risk Scores - Flood: {flood_score}/10, Wildfire: {wildfire_score}/10, Earthquake: {earthquake_score}/10\n"
        elif hazard == "earthquake":
            prompt_text += f"- All Risk Scores - Earthquake: {earthquake_score}/10, Wildfire: {wildfire_score}/10, Flood: {flood_score}/10\n"

        prompt_text += f"""
HOUSEHOLD CONTEXT:
- Household Size: {household_size} {"person" if household_size == "1" else "people"}
- Special Medical Needs: {special_needs if special_needs and special_needs.strip() else "None reported"}
- Current Preparedness Level: {preparedness_level}

INSTRUCTIONS:
1. Start with a personalized greeting that acknowledges their specific risk level and location
2. Reference their {hazard} risk score ({current_score}/10) and explain what this means for them specifically
3. Consider their household size, medical needs, and current preparedness level in all recommendations
4. If they have medical needs, prioritize those considerations in your advice
5. Give 4-6 specific, actionable steps tailored to their situation
6. Include local resources specific to Alameda County when relevant
7. If their risk score is high (7+), emphasize urgency and evacuation planning
8. If their risk score is moderate (4-6), focus on preparation and monitoring
9. If their risk score is low (1-3), focus on basic preparedness and awareness

CUSTOM GUIDANCE:
{custom_prompt if custom_prompt else f"Focus on {hazard}-specific safety measures appropriate for their risk level."}

Remember: Be encouraging but realistic about their risk level. Provide specific, actionable advice they can implement immediately."""

        messages = [
            {"role": "system", "content": f"You are a knowledgeable disaster preparedness assistant specializing in {hazard} safety for Alameda County. You provide personalized advice based on specific risk data, household needs, and local conditions. Always be helpful, encouraging, and specific in your recommendations."},
            {"role": "user", "content": prompt_text}
        ]

        if client:
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.7
                )
                initial_response = response.choices[0].message.content
            except Exception:
                logger.exception("Initial AI response failed for %s", hazard)
                initial_response = build_hazard_fallback_response(hazard, zip_code, current_score, current_explanation)
        else:
            initial_response = build_hazard_fallback_response(hazard, zip_code, current_score, current_explanation)

        # Store comprehensive metadata for continued conversation
        metadata = {
            "zip_code": zip_code,
            "inputs": inputs,
            "initial_prompt": prompt_text,
            "initial_response": initial_response,
            "risk_data": {
                "current_score": current_score,
                "current_explanation": current_explanation,
                "all_scores": {
                    "earthquake": earthquake_score,
                    "flood": flood_score,
                    "wildfire": wildfire_score
                },
                "hazard_specific": data
            }
        }
        session[meta_key] = metadata
        session[chat_key] = []

    # Ensure initial response is in chat
    if not any(msg.get("content") == metadata["initial_response"] for msg in chat):
        chat.insert(0, {"role": "assistant", "content": metadata["initial_response"]})

    reply = None
    if request.method == "POST":
        user_input = request.form.get("message")
        if user_input:
            chat.append({"role": "user", "content": user_input})
            
            # Build context-aware conversation with all the risk data
            context_messages = [
                {"role": "system", "content": f"""You are a disaster preparedness assistant for {hazard} safety in Alameda County. 

CONTINUE THIS CONVERSATION WITH FULL CONTEXT:
- User Location: ZIP {zip_code}
- {hazard.title()} Risk: {metadata['risk_data']['current_score']}/10
- Risk Explanation: {metadata['risk_data']['current_explanation']}
- Household: {household_size} people
- Medical Needs: {special_needs if special_needs and special_needs.strip() else "None"}
- Preparedness Level: {preparedness_level}

Always reference their specific situation and risk level when answering questions. Be helpful and specific."""},
                {"role": "user", "content": metadata["initial_prompt"]},
                {"role": "assistant", "content": metadata["initial_response"]},
            ]
            
            # Add recent conversation history
            context_messages.extend(chat)

            if client:
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=context_messages,
                        max_tokens=600,
                        temperature=0.7
                    )
                    reply = response.choices[0].message.content
                    chat.append({"role": "assistant", "content": reply})
                    session[chat_key] = chat
                except Exception:
                    logger.exception("Follow-up AI response failed for %s", hazard)
                    reply = build_hazard_fallback_response(
                        hazard,
                        zip_code,
                        metadata['risk_data']['current_score'],
                        metadata['risk_data']['current_explanation']
                    )
            else:
                reply = build_hazard_fallback_response(
                    hazard,
                    zip_code,
                    metadata['risk_data']['current_score'],
                    metadata['risk_data']['current_explanation']
                )

    # Load fault data for earthquake pages
    fault_geojson = None
    if hazard == "earthquake":
        fault_data = load_geojson_file("Fault_lines.Geojson")
        if fault_data:
            fault_geojson = json.dumps(fault_data)

    return safe_render(
        f"{hazard}.html",
        zip_code=zip_code,
        zip_geojson=zip_geojson,
        initial_response=metadata["initial_response"],
        chat=chat,
        reply=reply,
        fault_geojson=fault_geojson if hazard == "earthquake" else None,
        # Pass risk data to template for display
        risk_score=metadata['risk_data']['current_score'],
        risk_explanation=metadata['risk_data']['current_explanation'],
        household_size=household_size,
        special_needs=special_needs,
        preparedness_level=preparedness_level
    )

# --- Hazard Routes ---
@app.route("/wildfire", methods=["GET", "POST"])
def wildfire():
    return redirect(url_for("hazard_detail", name="wildfire"))

@app.route("/flood", methods=["GET", "POST"])
def flood():
    return redirect(url_for("hazard_detail", name="flood"))

@app.route("/earthquake", methods=["GET", "POST"])
def earthquake():
    return redirect(url_for("hazard_detail", name="earthquake"))

# --- Live Earthquake Map ---
@app.route("/live-earthquake-map")
def live_earthquake_map():
    zip_code = session.get("zip_code", "94601")
    try:
        zip_geojson_data = get_zip_boundary(zip_code)
        zip_geojson = json.dumps(zip_geojson_data) if zip_geojson_data else "{}"
    except Exception:
        logger.exception("Live earthquake map failed for ZIP %s", zip_code)
        zip_geojson = "{}"

    return safe_render("live_earthquake_map.html", zip_geojson=zip_geojson, zip_code=zip_code)

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
