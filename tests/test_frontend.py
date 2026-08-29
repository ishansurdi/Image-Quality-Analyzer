import re
import unittest
from pathlib import Path


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = Path("frontend/index.html").read_text(encoding="utf-8")
        cls.javascript = Path("frontend/app.js").read_text(encoding="utf-8")

    def test_html_ids_are_unique(self) -> None:
        identifiers = re.findall(r'id="([^"]+)"', self.html)

        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_javascript_selectors_exist_in_html(self) -> None:
        selectors = re.findall(r'querySelector\("#([^"\)]+)"\)', self.javascript)
        html_ids = set(re.findall(r'id="([^"]+)"', self.html))

        self.assertTrue(set(selectors).issubset(html_ids))

    def test_frontend_uses_expected_api_routes(self) -> None:
        self.assertIn('requestJson("/health")', self.javascript)
        self.assertIn('requestJson("/api/v1/analyses"', self.javascript)
        self.assertIn('requestJson("/api/v1/analyses?limit=10&offset=0")', self.javascript)

    def test_untrusted_content_uses_safe_rendering(self) -> None:
        self.assertNotIn("innerHTML", self.javascript)
        self.assertIn("textContent", self.javascript)

    def test_confidence_types_are_labelled(self) -> None:
        self.assertIn("issue confidence", self.javascript)
        self.assertIn("severity confidence", self.javascript)
        self.assertIn("Issue confidence:", self.html)
        self.assertIn("Severity confidence:", self.html)
        self.assertNotIn("Rule-based severity", self.javascript)
        self.assertIn("hasSeverityConfidence", self.javascript)


if __name__ == "__main__":
    unittest.main()
