from typing import Dict, Optional

from pydantic_models import LocationResult


def _first_present(*values):
    for value in values:
        if value:
            return str(value)
    return ""


def location_from_session(session_data: Dict) -> LocationResult:
    address = session_data.get("address") or ""
    zip_code = session_data.get("zip_code") or ""
    location_mode = session_data.get("location_mode") or ("address" if address else "zip" if zip_code else "")
    lat = session_data.get("lat")
    lon = session_data.get("lon")

    limitations = []
    if location_mode == "address":
        confidence = "source_backed" if lat is not None and lon is not None else "needs_review"
        precision_note = "Address was geocoded to a latitude/longitude point."
    elif location_mode == "zip":
        confidence = "mixed_support"
        precision_note = "ZIP-only input does not support address-level hazard zone checks."
        limitations.append(precision_note)
    else:
        confidence = "needs_review"
        precision_note = "No saved location was available; countywide fallback guidance may be used."
        limitations.append(precision_note)

    city = session_data.get("city") or ""
    county = session_data.get("county") or ""
    if not city and address:
        address_components = {
            component.strip().lower()
            for component in address.split(",")
            if component.strip()
        }
        for candidate in (
            "Oakland", "Berkeley", "Fremont", "Hayward", "Alameda", "San Leandro",
            "Union City", "Newark", "Dublin", "Pleasanton", "Livermore",
            "Castro Valley", "San Lorenzo", "Ashland", "Cherryland", "Fairview",
            "Sunol", "Hayward Acres", "Emeryville", "Piedmont", "Albany",
        ):
            if candidate.lower() in address_components:
                city = candidate
                break
    if not county and (zip_code or address):
        county = "Alameda County"

    if precision_note and precision_note not in limitations and location_mode != "address":
        limitations.append(precision_note)

    return LocationResult(
        input_address=session_data.get("input_address") or address or zip_code,
        formatted_address=address,
        lat=lat,
        lon=lon,
        city=city,
        county=county,
        zip_code=zip_code,
        neighborhood=session_data.get("neighborhood") or "",
        census_tract=session_data.get("census_tract") or "",
        geocoder=session_data.get("geocoder") or "nominatim",
        geocode_confidence=confidence,
        limitations=limitations,
    )


def location_from_geocode(
    input_address: str,
    formatted_address: str,
    lat: Optional[float],
    lon: Optional[float],
    zip_code: str,
    raw_address: Optional[Dict] = None,
) -> LocationResult:
    raw_address = raw_address or {}
    city = _first_present(
        raw_address.get("city"),
        raw_address.get("town"),
        raw_address.get("village"),
        raw_address.get("hamlet"),
        raw_address.get("municipality"),
    )
    county = _first_present(raw_address.get("county"), "Alameda County" if zip_code else "")
    neighborhood = _first_present(raw_address.get("neighbourhood"), raw_address.get("suburb"))

    return LocationResult(
        input_address=input_address,
        formatted_address=formatted_address,
        lat=lat,
        lon=lon,
        city=city,
        county=county,
        zip_code=zip_code,
        neighborhood=neighborhood,
        geocoder="nominatim",
        geocode_confidence="source_backed" if lat is not None and lon is not None else "needs_review",
        limitations=[] if lat is not None and lon is not None else ["Address could not be resolved to a usable point."],
    )
