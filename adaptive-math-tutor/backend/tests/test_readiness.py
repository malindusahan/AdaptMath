import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app import main
from app.core import readiness
from app.core.persistence import check_persistence_ready


class ReadinessTests(unittest.TestCase):
    def test_health_remains_lightweight_liveness_probe(self):
        response = main.health_check()

        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["service"], "adaptmath-backend")

    def test_readiness_reports_ready_when_local_dependencies_pass(self):
        with patch.object(
            readiness,
            "_check_checkpoint_database",
            return_value={"status": "ok", "detail": "db ok"},
        ), patch.object(
            readiness,
            "_check_complexity_model",
            return_value={"status": "ok", "detail": "model ok"},
        ):
            report = readiness.get_readiness_report()

        self.assertEqual(report["status"], "ready")

    def test_readiness_endpoint_returns_503_when_dependency_fails(self):
        failed_report = {
            "status": "not_ready",
            "checks": {
                "checkpoint_database": {
                    "status": "error",
                    "detail": "db unavailable",
                },
                "complexity_model": {
                    "status": "ok",
                    "detail": "model ok",
                },
            },
        }

        with patch.object(
            main,
            "get_readiness_report",
            return_value=failed_report,
        ):
            with self.assertRaises(HTTPException) as context:
                main.readiness_check()

        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(context.exception.detail, failed_report)

    def test_sqlite_checkpoint_store_is_readable(self):
        self.assertTrue(check_persistence_ready())


if __name__ == "__main__":
    unittest.main()
