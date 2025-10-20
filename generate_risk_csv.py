import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point

# Paths to your data files (adjust if needed)
ZIP_GEOJSON = "static/zipbound.geojson"
FLOOD_GEOJSON = "static/FldHaz.geojson"
WILDFIRE_GEOJSON = "static/FireHaz.geojson"
FAULT_GEOJSON = "static/Fault_lines.Geojson"  # Fault shapefile

OUTPUT_CSV = "static/zip_risk_scores.csv"

# Alameda County ZIP codes
alameda_zips = [
    "94501", "94502", "94536", "94538", "94539", "94541", "94542", "94544", "94545",
    "94546", "94550", "94551", "94552", "94555", "94560", "94566", "94568", "94577",
    "94578", "94579", "94580", "94586", "94587", "94601", "94602", "94603", "94605",
    "94606", "94607", "94608", "94609", "94610", "94611", "94612", "94618", "94619",
    "94621", "94706"
]

def load_geodata(path):
    print(f"Loading {path} ...")
    gdf = gpd.read_file(path)
    print(f"Loaded {len(gdf)} features from {path}")
    return gdf

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Load GeoDataFrames
    zip_gdf = load_geodata(ZIP_GEOJSON)
    flood_gdf = load_geodata(FLOOD_GEOJSON)
    wildfire_gdf = load_geodata(WILDFIRE_GEOJSON)
    fault_gdf = load_geodata(FAULT_GEOJSON)

    # Standardize ZIP code column
    # Load ZIPs from geojson and standardize column
    zip_gdf = load_geodata(ZIP_GEOJSON)

    if 'ZCTA5CE10' in zip_gdf.columns:
        zip_gdf = zip_gdf.rename(columns={'ZCTA5CE10': 'ZIP'})
    elif 'Zip' in zip_gdf.columns:
        pass
    elif 'ZIP_CODE' in zip_gdf.columns:
            zip_gdf = zip_gdf.rename(columns={'ZIP_CODE': 'ZIP'})
    elif 'ZIP' not in zip_gdf.columns:
        raise ValueError("ZIP code column not found in ZIP GeoJSON")

    # Keep only ZIP and geometry
    zip_gdf = zip_gdf[['ZIP', 'geometry']]

    # Use all ZIPs present in geojson
    alameda_zips = zip_gdf['ZIP'].tolist()
    print("Will process these ZIPs:", alameda_zips)

    # --- FLOOD CONTROL RISK BY FEMA FLD_ZONE ---
    print("Calculating flood risk based on FLD_ZONE...")

    # Overlay ZIPs with flood polygons
    flood_intersections = gpd.overlay(zip_gdf, flood_gdf, how='intersection')
    flood_intersections['intersect_area'] = flood_intersections.geometry.area

    # Pick the FLD_ZONE covering the largest area per ZIP
    idx = flood_intersections.groupby('ZIP')['intersect_area'].idxmax()
    dominant_flood = flood_intersections.loc[idx, ['ZIP', 'FLD_ZONE']].copy()
    dominant_flood = dominant_flood.rename(columns={
        'FLD_ZONE': 'Flood_Control_District'
    })

    # Assign flood risk scores by FLD_ZONE
    fld_zone_risk_map = {
        "A": {"score": 8, "explanation": "High flood risk, special flood hazard area."},
        "AE": {"score": 10, "explanation": "Very high flood risk, base flood elevation determined."},
        "X": {"score": 2, "explanation": "Minimal flood risk, outside 500-year floodplain."},
        "OTHER": {"score": 3, "explanation": "Moderate flood risk."}  # fallback
    }

    def get_flood_info(zone):
        if zone in fld_zone_risk_map:
            return fld_zone_risk_map[zone]
        return fld_zone_risk_map["OTHER"]

    flood_info = dominant_flood['Flood_Control_District'].apply(get_flood_info)
    dominant_flood['Flood_Risk_Score'] = flood_info.apply(lambda x: x['score'])
    dominant_flood['Flood_Risk_Explanation'] = flood_info.apply(lambda x: x['explanation'])

    # --- WILDFIRE HAZARD ---
    hazard_rank_map = {
        "Non-Wildland/Non-Urban": 0,
        "Urban Unzoned": 1,
        "Low": 2,
        "Moderate": 3,
        "High": 4,
        "Very High": 5
    }
    if 'HAZ_CLASS' not in wildfire_gdf.columns:
        raise ValueError("HAZ_CLASS column not found in wildfire GeoJSON")

    wildfire_gdf['hazard_rank'] = wildfire_gdf['HAZ_CLASS'].map(hazard_rank_map).fillna(0).astype(int)

    wildfire_join = gpd.sjoin(zip_gdf, wildfire_gdf[['hazard_rank', 'geometry']], how='left', predicate='intersects')
    wildfire_max = wildfire_join.groupby('ZIP')['hazard_rank'].max().reset_index()
    inv_hazard_map = {v: k for k, v in hazard_rank_map.items()}
    wildfire_max['Wildfire_Hazard_Level'] = wildfire_max['hazard_rank'].map(inv_hazard_map)

    # --- EARTHQUAKE RISK based on centroid distance to fault lines ---
    print("Calculating earthquake risk based on fault proximity...")

    # Reproject to a projection in meters for distance calculation
    zip_gdf_m = zip_gdf.to_crs(epsg=3310)
    fault_gdf_m = fault_gdf.to_crs(epsg=3310)

    # Compute centroids in meter projection
    zip_gdf_m['centroid'] = zip_gdf_m.geometry.centroid
    fault_union = fault_gdf_m.unary_union

    # Compute distance from each ZIP centroid to nearest fault (in meters)
    def earthquake_risk(point):
        dist = point.distance(fault_union)  # in meters
        if dist < 500:
            return 10, "Very high earthquake risk due to proximity (<0.5 km) to active fault lines."
        elif dist < 1000:
            return 8, "High earthquake risk due to proximity (<1 km) to active fault lines."
        elif dist < 5000:
            return 5, "Moderate earthquake risk due to proximity (1–5 km) to active fault lines."
        elif dist < 10000:
            return 3, "Low earthquake risk due to moderate distance (5–10 km) from active fault lines."
        else:
            return 1, "Minimal earthquake risk due to distance >10 km from active fault lines."

    # Apply risk calculation
    zip_gdf_m[['Earthquake_Risk_Score', 'Earthquake_Risk_Explanation']] = zip_gdf_m['centroid'].apply(lambda pt: pd.Series(earthquake_risk(pt)))

    # Merge risk scores back into original (WGS84) ZIP GeoDataFrame
    zip_gdf[['Earthquake_Risk_Score', 'Earthquake_Risk_Explanation']] = zip_gdf_m[['Earthquake_Risk_Score', 'Earthquake_Risk_Explanation']]


    # --- MERGE ALL DATA ---
    print("Merging flood, wildfire, and earthquake data...")
    master_df = zip_gdf[['ZIP', 'Earthquake_Risk_Score', 'Earthquake_Risk_Explanation']].merge(
        dominant_flood, on='ZIP', how='left'
    ).merge(
        wildfire_max[['ZIP', 'Wildfire_Hazard_Level']], on='ZIP', how='left'
    )

    # Fill missing flood or wildfire data with default
    master_df['Flood_Control_District'] = master_df['Flood_Control_District'].fillna('UNKNOWN')
    master_df['Wildfire_Hazard_Level'] = master_df['Wildfire_Hazard_Level'].fillna('Unknown')

    # --- FLOOD RISK MAP keyed by FLD_ZONE ---
    fld_zone_risk_map = {
        "A": {
            "score": 10,
            "explanation": "High flood risk (Zone A – no base flood elevations determined).",
            "chatbot_prompt": "Prepare for severe flooding; follow local evacuation orders and flood safety plans."
        },
        "AE": {
            "score": 8,
            "explanation": "High flood risk (Zone AE – base flood elevations determined).",
            "chatbot_prompt": "Expect flooding; secure belongings and follow flood safety protocols."
        },
        "X": {
            "score": 2,
            "explanation": "Minimal flood risk (Zone X – outside 100-year floodplain).",
            "chatbot_prompt": "Flood risk is low; standard precautions recommended."
        },
        "UNKNOWN": {
            "score": 0,
            "explanation": "Flood risk data unavailable.",
            "chatbot_prompt": "Flood risk information is unavailable; stay alert to local weather reports."
        }
    }

    # Apply mapping to master_df
    # Make sure you have a column in master_df representing the FLD_ZONE after your overlay
    master_df['FLD_ZONE'] = master_df['Flood_Control_District']  # or whatever column holds zone info
    flood_info = master_df['FLD_ZONE'].apply(lambda z: fld_zone_risk_map.get(z, fld_zone_risk_map['UNKNOWN']))

    master_df['Flood_Risk_Score'] = flood_info.apply(lambda x: x['score'])
    master_df['Flood_Risk_Explanation'] = flood_info.apply(lambda x: x['explanation'])
    master_df['Flood_Chatbot_Prompt'] = flood_info.apply(lambda x: x['chatbot_prompt'])


    # --- WILDFIRE RISK SCORE & CHATBOT PROMPT ---
    wildfire_risk_map = {
        "Non-Wildland/Non-Urban": {
            "score": 1,
            "explanation": "Very low wildfire risk; this area is mostly urban with little vegetation.",
            "chatbot_prompt": "Low wildfire risk; maintain defensible space and follow local fire safety guidelines."
        },
        "Urban Unzoned": {
            "score": 1,
            "explanation": "Very low wildfire risk; mostly urban unzoned areas with minimal wildfire threat.",
            "chatbot_prompt": "Low wildfire risk; maintain defensible space and follow local fire safety guidelines."
        },
        "Low": {
            "score": 3,
            "explanation": "Low wildfire risk; some vegetation is present but fire spread is unlikely under normal conditions.",
            "chatbot_prompt": "Low to moderate wildfire risk; keep vegetation trimmed and prepare for fire season."
        },
        "Moderate": {
            "score": 5,
            "explanation": "Moderate wildfire risk; sufficient vegetation to allow fire spread under dry conditions.",
            "chatbot_prompt": "Moderate wildfire risk; prepare evacuation plans and emergency kits."
        },
        "High": {
            "score": 8,
            "explanation": "High wildfire risk; significant vegetation and exposure that could allow rapid fire spread.",
            "chatbot_prompt": "High wildfire risk; stay alert during fire season and follow local advisories."
        },
        "Very High": {
            "score": 10,
            "explanation": "Very high wildfire risk; areas with dense vegetation and extreme fire behavior potential.",
            "chatbot_prompt": "Very high wildfire risk; implement all safety measures and evacuate promptly if advised."
        },
        "Unknown": {
            "score": 0,
            "explanation": "Wildfire risk data unavailable for this area.",
            "chatbot_prompt": "Wildfire risk data unavailable; stay informed via local sources."
        }
    }


    wildfire_info = master_df['Wildfire_Hazard_Level'].apply(lambda lvl: wildfire_risk_map.get(lvl, wildfire_risk_map['Unknown']))
    master_df['Wildfire_Risk_Score'] = wildfire_info.apply(lambda x: x['score'])
    master_df['Wildfire_Chatbot_Prompt'] = wildfire_info.apply(lambda x: x['chatbot_prompt'])
    master_df['Wildfire_Risk_Explanation'] = wildfire_info.apply(lambda x: x['explanation'])


    # --- Save CSV ---
    output_columns = [
        "ZIP",
        "Earthquake_Risk_Score",
        "Earthquake_Risk_Explanation",
        "Flood_Control_District",
        "Wildfire_Hazard_Level",
        "Flood_Risk_Score",
        "Flood_Risk_Explanation",
        "Flood_Chatbot_Prompt",
        "Wildfire_Risk_Score",
        "Wildfire_Chatbot_Prompt",
        "Wildfire_Risk_Explanation"
    ]

    master_df[output_columns].to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV} with {len(master_df)} ZIP codes.")

if __name__ == "__main__":
    main()
