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
    return {
        "address_summary": {
            "display_name": "4183 Observatory Ave, Oakland, CA",
            "city": city,
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
        self.assertIn("Important official mapped findings", html)
        self.assertIn("Other hazards checked", html)
        self.assertIn("Map information unavailable", html)
        self.assertIn("Inside a CGS mapped Alquist-Priolo fault rupture zone.", html)
        self.assertIn("CAL FIRE Fire Hazard Severity Zone: Very High", html)
        self.assertIn("FEMA Zone X, not identified as a Special Flood Hazard Area.", html)
        self.assertIn("Outside the official tsunami hazard area.", html)
        self.assertIn("Address-specific map information is temporarily unavailable.", html)

    def test_summary_notices_and_plan_area_are_jurisdiction_scoped(self):
        oakland = _render("Oakland")
        self.assertIn("Oakland sub-area statistics come from the Draft 2026–2031 LHMP.", oakland)
        self.assertIn("They are planning context, not property-risk ratings", oakland)
        self.assertIn("Plan area:</strong> Glenview/Redwood Heights", oakland)
        self.assertIn("This location may be near a plan-area boundary.", oakland)

        hayward = _render("Hayward")
        self.assertIn("Local LHMP area ratings are not yet available for Hayward.", hayward)
        self.assertNotIn("Official Oakland plan-area hazard ratings", hayward)
        self.assertNotIn("Plan area:</strong>", hayward)

    def test_failed_or_unknown_data_does_not_erase_valid_mapped_findings(self):
        html = _render()
        important_section = html.split("Important official mapped findings", 1)[1].split("Other hazards checked", 1)[0]
        unavailable_section = html.split("Map information unavailable", 1)[1]
        self.assertIn("Alquist-Priolo", important_section)
        self.assertIn("Very High", important_section)
        self.assertIn("Address-specific map information is temporarily unavailable.", unavailable_section)

    def test_actions_are_consolidated_not_repeated_in_rows(self):
        html = _render()
        self.assertEqual(html.count("Do these first"), 1)
        self.assertNotIn("What to do next", html)
        self.assertNotIn("During, after, and recovery", html)

    def test_mapped_finding_details_are_native_summary_siblings(self):
        html = _render()
        self.assertIn('<details class="sr-mapped-finding-row', html)
        self.assertIn('<summary class="sr-mapped-finding-summary">', html)
        self.assertIn('class="sr-summary-hazard-details sr-mapped-finding-details"', html)
        self.assertLess(
            html.index('<summary class="sr-mapped-finding-summary">'),
            html.index('class="sr-summary-hazard-details sr-mapped-finding-details"'),
        )
        row_fragment = html.split('<details class="sr-mapped-finding-row', 1)[1].split("</details>", 1)[0]
        self.assertEqual(row_fragment.count("<summary"), 1)
        self.assertNotIn("<footer>", row_fragment)

    def test_mapped_finding_details_css_contract(self):
        css = (BASE_DIR / "static" / "css" / "stayready.css").read_text(encoding="utf-8")
        row_rule = css.split(".sr-mapped-finding-row {", 1)[1].split("}", 1)[0]
        summary_rule = css.split(".sr-mapped-finding-summary {", 1)[1].split("}", 1)[0]
        details_rule = css.split(".sr-mapped-finding-details {", 1)[1].split("}", 1)[0]
        details_grid_rule = next(
            rule for rule in re.findall(r"\.sr-mapped-finding-details-grid\s*\{([^}]*)\}", css)
            if "repeat(2" in rule
        )
        source_url_rule = css.split(".sr-source-url {", 1)[1].split("}", 1)[0]

        self.assertIn("display: block", row_rule)
        self.assertNotIn("grid-template-columns", row_rule)
        self.assertIn("display: grid", summary_rule)
        self.assertIn("minmax(300px", summary_rule)
        self.assertIn("auto", summary_rule)
        self.assertIn("display: block", details_rule)
        self.assertIn("width: 100%", details_rule)
        self.assertIn("repeat(2, minmax(0, 1fr))", details_grid_rule)
        self.assertIn("overflow-wrap: anywhere", source_url_rule)
        self.assertNotIn(".sr-mapped-finding-row details[open]", css)
        self.assertNotIn("word-break: break-all", css)

    def test_mapped_finding_responsive_details_rules_exist(self):
        css = (BASE_DIR / "static" / "css" / "stayready.css").read_text(encoding="utf-8")
        self.assertRegex(
            css,
            re.compile(
                r"@media \(max-width: 991px\).*?\.sr-mapped-finding-details-grid\s*\{[^}]*grid-template-columns: 1fr",
                re.S,
            ),
        )
        self.assertRegex(
            css,
            re.compile(
                r"@media \(max-width: 575px\).*?\.sr-mapped-finding-summary\s*\{[^}]*display: flex[^}]*flex-direction: column",
                re.S,
            ),
        )


if __name__ == "__main__":
    unittest.main()
