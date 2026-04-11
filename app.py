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
            error_message="Something went wrong, but your session data is محفوظ (safe). Please try again.",
            helper_message="This page is under development, but your data is still valid.",
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
        session["household"] = household
        session["special_needs"] = special_needs
        session["preparedness"] = preparedness

        for hazard in ["wildfire", "flood", "earthquake"]:
            session.pop(f"chat_{hazard}", None)
            session.pop(f"meta_{hazard}", None)

        return redirect(url_for("risk_summary"))
    except Exception:
        logger.exception("Form processing failed.")
        session["form_error"] = "We could not process that request, but your session data is محفوظ (safe). Please try again."
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
        error_message="Something went wrong, but your session data is محفوظ (safe). Please try again.",
        helper_message="This page does not exist, but you can return home and continue using the app."
    ), 404

@app.errorhandler(500)
def internal_error(error):
    logger.exception("500 error on %s", request.path)
    return safe_render(
        "error.html",
        title="Temporary Error",
        error_heading="Something went wrong",
        error_message="Something went wrong, but your session data is محفوظ (safe). Please try again.",
        helper_message="You can return home or continue to your risk summary."
    ), 500


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.exception("Unhandled exception on %s", request.path)
    return safe_render(
        "error.html",
        title="Temporary Error",
        error_heading="Something went wrong",
        error_message="Something went wrong, but your session data is محفوظ (safe). Please try again.",
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
    return safe_render("home.html", error=error, form_data=form_data)



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
        hazards=hazards_sorted,
        recommended_actions=recommended_actions,
        start_here_steps=start_here_steps,
        checklist_items=checklist_items,
        checklist_total_count=len(all_checklist_items),
        warning_message=warning_message
    )

# --- Unified Hazard Map ---
@app.route("/map", methods=["GET", "POST"])
def map():
    """Unified hazard map showing all risks with toggleable layers"""
    zip_code = session.get("zip_code", "94601")
    address = session.get("address")
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
        map_notice=map_notice
    )

    # --- Get user session data ---
    zip_code = session.get("zip_code", "94601")
    lat = session.get("lat")
    lon = session.get("lon")
    address = session.get("address")

    # --- Get risk data for the ZIP code ---
    data = zip_risk_data.get(zip_code, {})

    # --- Prepare risk scores ---
    risk_scores = {
        'wildfire': {
            'score': data.get("Wildfire_Risk_Score", 0),
            'explanation': data.get("Wildfire_Risk_Explanation", "No data available")
        },
        'earthquake': {
            'score': data.get("Earthquake_Risk_Score", 0), 
            'explanation': data.get("Earthquake_Risk_Explanation", "No data available")
        },
        'flood': {
            'score': data.get("Flood_Risk_Score", 0),
            'explanation': data.get("Flood_Risk_Explanation", "No data available")
        }
    }
    print("DEBUG → Address:", address)
    print("DEBUG → Lat:", lat)
    print("DEBUG → Lon:", lon)
    return render_template(
        "map.html",
        zip_code=zip_code,
        risk_scores=risk_scores,
        user_lat=lat,
        user_lon=lon,
        user_address=address,
        **get_data_sources_context()
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

# --- API Endpoints ---

# Live Earthquake API
@app.route("/api/live-earthquakes")
def api_live_earthquakes():
    """API endpoint for live earthquake data with Alameda County filtering"""
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.exception("Unable to load live earthquake data.")
        return jsonify({
            "type": "FeatureCollection",
            "features": [],
            "message": "Map data temporarily unavailable. You can still view your risk and preparedness steps below."
        })

    # Alameda County bounds (more precise)
    bounds = {
        "min_lat": 37.4, "max_lat": 37.9,
        "min_lon": -122.4, "max_lon": -121.4
    }
    
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

    return jsonify({"type": "FeatureCollection", "features": features})

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
    """API endpoint for flood hazard zones"""
    data = load_geojson_file("FldHaz.geojson")
    if data:
        return jsonify(data)
    return jsonify({
        "type": "FeatureCollection",
        "features": [],
        "message": "Map data temporarily unavailable. You can still view your risk and preparedness steps below."
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
    return hazard_page("wildfire", "Wildfire Risk", "#ff7043")

@app.route("/flood", methods=["GET", "POST"])
def flood():
    return hazard_page("flood", "Flood Risk", "#0288d1")

@app.route("/earthquake", methods=["GET", "POST"])
def earthquake():
    return hazard_page("earthquake", "Earthquake Risk", "#2196f3")

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
