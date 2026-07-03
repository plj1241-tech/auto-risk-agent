import unittest

import pandas as pd

from src.agent.report_generator import build_report
from src.agent.pdf_report import generate_pdf
from src.agent.risk_tools import (
    compare_peers,
    explain_shap,
    get_current_risk,
    run_scenario,
)
from src.features.repair_quarterly_flows import repair_legacy_quarterly_flows


class Phase3ToolTests(unittest.TestCase):
    def test_current_risk(self):
        result = get_current_risk("현대모비스")
        self.assertEqual(result["year"], 2024)
        self.assertIn("z_score", result["metrics"])

    def test_compare_peers(self):
        result = compare_peers("현대모비스")
        self.assertGreaterEqual(result["comparisons"]["z_score"]["health_rank"], 1)

    def test_scenario_guardrail_and_prediction(self):
        result = run_scenario("현대모비스", {"base_rate_kr": 2.0})
        self.assertFalse(result["guardrail"]["causal_forecast"])
        self.assertIn("z_score", result["predictions"])
        with self.assertRaises(ValueError):
            run_scenario("현대모비스", {"base_rate_kr": -10})

    def test_shap(self):
        result = explain_shap("현대모비스", "z_score", top_n=3)
        self.assertEqual(len(result["top_macro_effects"]), 3)


class Phase4ReportTests(unittest.TestCase):
    def test_report_five_sections(self):
        result = build_report("현대모비스", {"base_rate_kr": 2.0})
        self.assertEqual(len(result["sections"]), 5)
        self.assertIn("## 5. 점검 제안", result["markdown"])

    def test_pdf_report(self):
        result = build_report("현대모비스", {"base_rate_kr": 2.0})
        pdf = generate_pdf(result)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 5000)


class QuarterlyFlowTests(unittest.TestCase):
    def test_legacy_flow_repair_and_idempotence(self):
        frame = pd.DataFrame({
            "corp_name": ["테스트사"] * 4,
            "year": [2024] * 4,
            "quarter": [1, 2, 3, 4],
            # Old transform: [Q1, Q2-Q1, Q3-Q2, annual-Q3]
            "revenue": [100.0, 20.0, 10.0, 370.0],
            "op_income": [10.0, 2.0, 1.0, 37.0],
            "net_income": [8.0, 1.0, 1.0, 30.0],
            "interest_exp": [2.0, 0.5, 0.5, 7.0],
        })
        repaired = repair_legacy_quarterly_flows(frame)
        self.assertEqual(repaired["revenue"].tolist(), [100.0, 120.0, 130.0, 150.0])
        self.assertEqual(repaired["revenue_reported"].tolist(), [100.0, 120.0, 130.0, 500.0])
        rerun = repair_legacy_quarterly_flows(repaired)
        self.assertEqual(rerun["revenue"].tolist(), repaired["revenue"].tolist())


if __name__ == "__main__":
    unittest.main()
