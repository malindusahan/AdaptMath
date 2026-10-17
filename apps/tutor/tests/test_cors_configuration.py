import os
import unittest
from unittest.mock import patch

from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings
from app.main import app


_BASE_ENV = {
    "PROFILE_API_URL": "http://profile.test",
    "MEMORY_API_URL": "http://memory.test",
    "STRATEGY_API_URL": "http://strategy.test",
    "GEMINI_API_KEY": "test-key",
}


class CorsConfigurationTests(unittest.TestCase):
    def _settings(self, **overrides) -> Settings:
        env = dict(_BASE_ENV)
        env.update(overrides)

        with patch.dict(os.environ, env, clear=True):
            return Settings(_env_file=None)

    def test_default_origins_are_explicit_local_frontend_origins(self):
        settings = self._settings()

        self.assertEqual(
            settings.allowed_cors_origins,
            [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ],
        )
        self.assertNotIn("*", settings.allowed_cors_origins)

    def test_configured_origins_are_trimmed_deduplicated_and_normalized(self):
        settings = self._settings(
            CORS_ORIGINS=(
                " https://adaptmath.example.com/, "
                "http://localhost:5173,"
                "https://adaptmath.example.com "
            )
        )

        self.assertEqual(
            settings.allowed_cors_origins,
            [
                "https://adaptmath.example.com",
                "http://localhost:5173",
            ],
        )

    def test_wildcard_origin_is_rejected(self):
        settings = self._settings(
            CORS_ORIGINS="*"
        )

        with self.assertRaises(ValueError):
            _ = settings.allowed_cors_origins

    def test_application_registers_cors_middleware(self):
        middleware_classes = [
            middleware.cls
            for middleware in app.user_middleware
        ]

        self.assertIn(
            CORSMiddleware,
            middleware_classes,
        )


if __name__ == "__main__":
    unittest.main()
