import unittest
import xml.etree.ElementTree as ET

from app import app


TRUST_NOTICE = (
    "StayReady is an independent educational tool using official public data. "
    "It is not an official emergency service and does not replace AC Alert, "
    "evacuation orders, 911, or instructions from public agencies."
)


class SEOContractTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_robots_and_sitemap_are_public_and_valid(self):
        robots = self.client.get("/robots.txt")
        self.assertEqual(robots.status_code, 200)
        self.assertIn("User-agent: *", robots.get_data(as_text=True))
        self.assertIn(
            "Sitemap: https://stayreadynow.org/sitemap.xml",
            robots.get_data(as_text=True),
        )

        sitemap = self.client.get("/sitemap.xml")
        self.assertEqual(sitemap.status_code, 200)
        root = ET.fromstring(sitemap.data)
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        locations = [
            node.text for node in root.findall("sm:url/sm:loc", namespace)
        ]
        self.assertIn("https://stayreadynow.org/map", locations)
        self.assertIn("https://stayreadynow.org/hazards/earthquake", locations)
        self.assertNotIn("https://stayreadynow.org/risk_summary", locations)

    def test_required_favicon_assets_are_served(self):
        for path in (
            "/favicon.ico",
            "/static/favicon-48.png",
            "/static/favicon-96.png",
            "/static/apple-touch-icon.png",
            "/static/site.webmanifest",
        ):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertTrue(response.data, path)
            response.close()

    def test_homepage_has_favicon_social_and_canonical_tags(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('<link rel="icon" href="/favicon.ico" sizes="any"', html)
        self.assertIn('sizes="48x48" href="/static/favicon-48.png"', html)
        self.assertIn('sizes="96x96" href="/static/favicon-96.png"', html)
        self.assertIn('<link rel="canonical" href="https://stayreadynow.org/"', html)
        self.assertIn('property="og:title"', html)
        self.assertIn('name="twitter:card" content="summary"', html)
        self.assertIn('"@type": "WebApplication"', html)
        self.assertNotIn('"@type": "GovernmentOrganization"', html)

    def test_homepage_has_visible_independent_tool_disclaimer(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertGreaterEqual(html.count(TRUST_NOTICE), 2)

    def test_static_public_pages_have_unique_canonical_urls(self):
        for path in (
            "/map",
            "/hazards",
            "/sources",
            "/about",
            "/privacy",
            "/terms",
            "/resources",
        ):
            html = self.client.get(path).get_data(as_text=True)
            self.assertIn(
                f'<link rel="canonical" href="https://stayreadynow.org{path}"',
                html,
                path,
            )


if __name__ == "__main__":
    unittest.main()
