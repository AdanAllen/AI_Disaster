import re
import unittest
from pathlib import Path

from flask import render_template, session

from app import app


BASE_DIR = Path(__file__).resolve().parents[1]


def _hazard(slug, label):
    return {
        "slug": slug,
        "hazard": label,
        "why_it_matters": "Preparedness context.",
        "priority_label": "Mapped information",
        "evidence_status_label": "Checked",
        "before_actions": [],
        "during_actions": [],
        "after_actions": [],
        "recovery_steps": [],
        "limitations": ["Checked layers are not property-specific determinations."],
        "official_mapped_evidence": [],
    }


def _priority(slug, label, status, summary, interpretation, source="Official test layer", plan_area="Glenview/Redwood Heights"):
    return {
        "slug": slug,
        "hazard": label,
        "mapped_finding_status": status,
        "mapped_finding_category": {
            "significant_official_finding": "Official mapped finding",
            "successful_check_no_significant_match": "Other hazard checked",
            "data_unavailable": "Map information unavailable",
        }[status],
        "mapped_finding_summary": summary,
        "mapped_finding_interpretation": interpretation,
        "mapped_finding_source_name": source,
        "mapped_finding_confidence": "Medium",
        "confidence": "Medium",
        "local_exposure": {
            "data_last_checked": "2026-06-18T12:00:00Z",
            "source_specific_terminology": [summary],
        },
        "successful_sources": [source] if status != "data_unavailable" else [],
        "failed_sources": [source] if status == "data_unavailable" else [],
        "address_specific_notes": "Official map result preserved from the app GIS output.",
        "plan_area": plan_area,
        "boundary_warning": "This location may be near a plan-area boundary." if plan_area else "",
        "sources_used": [{
            "name": "Official source",
            "url": "https://example.gov/a/very/long/source/path/that/should/remain/wrappable",
            "reference": "Reference table",
            "status_label": "Official",
        }],
        "limitations": ["This is not an exact personal risk or property-damage prediction."],
        "displayed_hazard_level": "Unknown",
        "official_lhmp_area_rating": "Unknown",
        "stayready_fallback_rating": "Unknown",
    }


def _resident_plan(city="Oakland"):
    def card(slug, label, outcome, finding, dataset, *, unavailable=False):
        status = "data_unavailable" if unavailable else ("mapped_match" if outcome == "mapped_match" else "checked_no_match")
        attention = "information-unavailable" if unavailable else ("direct-mapped-concern" if outcome == "mapped_match" else "prepare-despite-no-match")
        return {
            "hazard_id": slug,
            "hazard_name": label,
            "descriptor": "Official map evidence and reviewed preparedness context.",
            "attention": {
                "category": attention,
                "label": "Information unavailable" if unavailable else ("Direct mapped concern" if outcome == "mapped_match" else "Prepare despite no mapped match"),
            },
            "address_evidence": {
                "outcome": status,
                "label": "Information unavailable" if unavailable else ("Official mapped match found" if outcome == "mapped_match" else "Checked, no mapped match found"),
                "findings": [] if unavailable else [finding],
                "explanation": finding,
                "proximity": [],
                "has_source_failures": unavailable,
                "records": [{
                    "status": status,
                    "status_label": "Data unavailable" if unavailable else ("Official mapped match found" if outcome == "mapped_match" else "Checked, no mapped match found"),
                    "geographic_scope": "Address point",
                    "dataset_name": dataset,
                    "finding": finding,
                    "source_agency": "Official test agency",
                    "review_status": "official_provisional",
                    "source_url": "https://example.gov/source",
                }],
            },
            "subarea_context": ({
                "status": "available",
                "subarea_name": "Glenview/Redwood Heights",
                "findings": [{"text": "Reviewed Oakland planning-area context.", "reference": "Draft LHMP table"}],
                "claims": [],
            } if city == "Oakland" else {"status": "unavailable", "claims": []}),
            "regional_context": {"status": "unavailable", "claims": []},
            "vulnerability": {"label": "Unknown"},
            "interpretation": "Use this exact map result with the stated source limits.",
            "actions": {"before": [{"instruction": "Use official alert channels for emergency instructions."}]},
            "detail_url": f"/hazards/{slug}",
        }

    cards = [
        card("earthquake", "Earthquake", "mapped_match", "Inside a CGS mapped Alquist-Priolo fault rupture zone.", "CGS Alquist-Priolo Earthquake Fault Zones"),
        card("wildfire", "Wildfire", "mapped_match", "CAL FIRE Fire Hazard Severity Zone: Very High", "CAL FIRE Fire Hazard Severity Zone layer"),
        card("flood", "Flood", "checked_no_match", "FEMA Zone X, not identified as a Special Flood Hazard Area.", "FEMA National Flood Hazard Layer snapshot"),
        card("tsunami", "Tsunami", "checked_no_match", "Outside the official tsunami hazard area.", "California Geological Survey tsunami hazard area layer"),
        card("landslide", "Landslide", "data_unavailable", "Address-specific map information is temporarily unavailable.", "CGS and local landslide layers", unavailable=True),
    ]
    return {
        "address_summary": {
            "display_name": "4183 Observatory Ave, Oakland, CA",
            "city": city,
            "county": "Alameda County",
            "zip_code": "94619",
            "precision_label": "Address point",
        },
        "canonical_summary": {
            "hazards": cards,
            "unknowns": ["Building construction and condition"],
            "result_quality": {
                "precision_label": "Address point",
                "status_label": "4 checks returned; 1 unavailable",
                "unavailable_check_count": 1,
                "geocoding_note": "Address geocoding is a point approximation.",
            },
        },
        "hazard_priorities": [
            _priority(
                "earthquake",
                "Earthquake",
                "significant_official_finding",
                "Inside a CGS mapped Alquist-Priolo fault rupture zone.",
                "This official mapped finding applies to the address point. Fault proximity, if shown, is separate context and is not the same as an Alquist-Priolo polygon match.",
                "CGS Alquist-Priolo Earthquake Fault Zones",
            ),
            _priority(
                "wildfire",
                "Wildfire",
                "significant_official_finding",
                "CAL FIRE Fire Hazard Severity Zone: Very High",
                "The address point intersects the checked Fire Hazard Severity Zone layer.",
                "CAL FIRE Fire Hazard Severity Zone layer",
            ),
            _priority(
                "flood",
                "Flood",
                "successful_check_no_significant_match",
                "FEMA Zone X, not identified as a Special Flood Hazard Area.",
                "This is a checked map result, not a guarantee that flooding or drainage impacts cannot occur.",
                "FEMA National Flood Hazard Layer snapshot",
            ),
            _priority(
                "tsunami",
                "Tsunami",
                "successful_check_no_significant_match",
                "Outside the official tsunami hazard area.",
                "This checked-layer result is not a guarantee of safety and does not replace official instructions during an event.",
                "California Geological Survey tsunami hazard area layer",
            ),
            _priority(
                "landslide",
                "Landslide",
                "data_unavailable",
                "Address-specific map information is temporarily unavailable.",
                "This failed check was kept separate and was not used to lower or raise a hazard priority.",
                "CGS and local landslide layers",
            ),
        ],
        "hazards": [
            _hazard("earthquake", "Earthquake"),
            _hazard("wildfire", "Wildfire"),
            _hazard("flood", "Flood"),
            _hazard("tsunami", "Tsunami"),
            _hazard("landslide", "Landslide"),
        ],
        "household_context": {"has_context": False},
        "household_priorities": [],
        "what_to_do_now": [{
            "action": {
                "title": "Sign up for official alerts",
                "instruction": "Use official alert channels for emergency instructions.",
                "steps": [],
                "citation": {},
            },
        }],
        "recovery_plan": [],
        "checks": {"checked": ["Official mapped layers"], "not_checked": []},
        "additional_local_hazards": [],
        "sources": [],
        "limits": ["Mapped findings are not personal risk predictions."],
    }


def _render(city="Oakland"):
    with app.test_request_context("/risk_summary"):
        session["zip_code"] = "94619"
        return render_template(
            "risk_summary.html",
            resident_plan=_resident_plan(city),
            location_context={"location_mode": "address"},
            warning_message=None,
            empty_state=False,
        )


class RiskSummaryMappedFindingsTests(unittest.TestCase):
    def test_summary_hides_unverified_lhmp_and_fallback_ratings(self):
        html = _render()
        for forbidden in (
            "Unknown hazard priority",
            "Oakland LHMP area rating",
            "Fallback rating",
            "High hazards",
            "Medium hazards",
            "Lower hazards identified",
            "top four",
        ):
            self.assertNotIn(forbidden, html)
        self.assertNotRegex(html, re.compile(r"\b(?:Low|Medium|High) hazard priority\b"))

    def test_summary_separates_significant_checked_and_unavailable_results(self):
        html = _render()
        self.assertIn("Your main hazard picture", html)
        self.assertIn("Direct mapped concern", html)
        self.assertIn("Checked, no mapped match found", html)
        self.assertIn("Information unavailable", html)
        self.assertIn("Inside a CGS mapped Alquist-Priolo fault rupture zone.", html)
        self.assertIn("CAL FIRE Fire Hazard Severity Zone: Very High", html)
        self.assertIn("FEMA Zone X, not identified as a Special Flood Hazard Area.", html)
        self.assertIn("Outside the official tsunami hazard area.", html)
        self.assertIn("Address-specific map information is temporarily unavailable.", html)

    def test_summary_notices_and_plan_area_are_jurisdiction_scoped(self):
        oakland = _render("Oakland")
        self.assertIn("Oakland planning area · Glenview/Redwood Heights", oakland)
        self.assertIn("Draft 2026–2031 LHMP planning context; reviewed claim, not a property rating.", oakland)

        hayward = _render("Hayward")
        self.assertIn("No eligible, exactly cited local-plan category is available", hayward)
        self.assertNotIn("Oakland planning area", hayward)

    def test_failed_or_unknown_data_does_not_erase_valid_mapped_findings(self):
        html = _render()
        self.assertIn("Alquist-Priolo", html)
        self.assertIn("Very High", html)
        self.assertIn("Address-specific map information is temporarily unavailable.", html)
        self.assertIn("Another map source was unavailable; it was not treated as a safer result.", html)

    def test_actions_are_consolidated_not_repeated_in_rows(self):
        html = _render()
        self.assertEqual(html.count("Start with these high-impact actions"), 1)
        self.assertNotIn("What to do next", html)
        self.assertNotIn("During, after, and recovery", html)

    def test_mapped_finding_details_are_native_summary_siblings(self):
        html = _render()
        self.assertIn('<details class="sr-evidence-details">', html)
        self.assertIn("Official source details and all returned checks", html)
        self.assertIn('class="sr-evidence-record-list"', html)
        row_fragment = html.split('<details class="sr-evidence-details">', 1)[1].split("</details>", 1)[0]
        self.assertEqual(row_fragment.count("<summary"), 1)
        self.assertIn("Address point", row_fragment)

    def test_mapped_finding_details_css_contract(self):
        css = (BASE_DIR / "static" / "css" / "stayready.css").read_text(encoding="utf-8")
        card_rule = css.split(".sr-hazard-evidence-card {", 1)[1].split("}", 1)[0]
        grid_rule = css.split(".sr-hazard-evidence-grid {", 1)[1].split("}", 1)[0]
        self.assertIn("border", card_rule)
        self.assertIn("display: grid", grid_rule)
        self.assertIn("1.05fr 1fr 1.3fr", grid_rule)

    def test_mapped_finding_responsive_details_rules_exist(self):
        css = (BASE_DIR / "static" / "css" / "stayready.css").read_text(encoding="utf-8")
        self.assertRegex(
            css,
            re.compile(
                r"@media \(max-width: 991px\).*?\.sr-hazard-evidence-grid\s*\{[^}]*grid-template-columns: 1fr",
                re.S,
            ),
        )
        self.assertRegex(
            css,
            re.compile(
                r"@media \(max-width: 575px\).*?\.sr-hazard-evidence-card > header\s*\{[^}]*grid-template-columns: 40px 1fr",
                re.S,
            ),
        )


if __name__ == "__main__":
    unittest.main()
