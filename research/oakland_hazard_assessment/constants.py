"""Shared constants for the Oakland research-only assessment."""

PLAN_AREAS = [
    "Central East Oakland",
    "Coliseum/Airport",
    "Downtown",
    "East Oakland Hills",
    "Eastlake/Fruitvale",
    "Glenview/Redwood Heights",
    "North Oakland Hills",
    "North Oakland/Adams Point",
    "West Oakland",
]

HAZARDS = ["earthquake", "wildfire", "flood", "landslide", "tsunami"]

METRIC_TYPES = {
    "official_hazard_priority",
    "probability",
    "impact",
    "scenario_hazard_rating",
    "physical_exposure",
    "population_exposure",
    "property_exposure",
    "modeled_loss",
    "community_vulnerability",
    "EPC_context",
    "historical_frequency",
    "preparedness_context",
    "unknown_metric",
}

SOURCE_STATUSES = {"adopted", "draft", "superseded", "official_gis"}

