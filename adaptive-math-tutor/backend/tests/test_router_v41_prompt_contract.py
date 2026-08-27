import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class RouterV41ArchiveTests(unittest.TestCase):
    def test_v41_is_preserved_before_universal_assessment_change(self) -> None:
        archived = (
            BACKEND_ROOT
            / "app"
            / "agents"
            / "router"
            / "router_agent_v41_pre_universal_assessment.py"
        )

        self.assertTrue(
            archived.exists()
        )
        self.assertIn(
            'ROUTER_VERSION = "4.1-development"',
            archived.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
