import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class RouterV4ArchiveTests(unittest.TestCase):
    def test_v4_development_source_is_archived_not_active(self) -> None:
        active = (
            BACKEND_ROOT / "app" / "agents" / "router" / "router_agent.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'ROUTER_VERSION = "5.0-development"',
            active,
        )

        self.assertTrue(
            (
                BACKEND_ROOT
                / "app"
                / "agents"
                / "router"
                / "router_agent_v4_development.py"
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
