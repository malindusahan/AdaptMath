import unittest

from app.core.model_errors import (
    model_service_http_exception,
)


class _FakeResponse:
    def __init__(
        self,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.headers = headers or {}


class RateLimitError(Exception):
    status_code = 429

    def __init__(self) -> None:
        super().__init__(
            "sensitive provider quota details"
        )
        self.response = _FakeResponse(
            {
                "retry-after": "7",
            }
        )


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class APIStatusError(Exception):
    status_code = 500


class ModelErrorMappingTests(
    unittest.TestCase
):
    def test_rate_limit_is_safe_503(
        self,
    ) -> None:
        exc = model_service_http_exception(
            RateLimitError()
        )

        self.assertEqual(
            exc.status_code,
            503,
        )
        self.assertEqual(
            exc.headers,
            {
                "Retry-After": "7",
            },
        )
        self.assertNotIn(
            "sensitive",
            str(exc.detail),
        )

    def test_timeout_is_504(
        self,
    ) -> None:
        exc = model_service_http_exception(
            APITimeoutError()
        )

        self.assertEqual(
            exc.status_code,
            504,
        )

    def test_connection_failure_is_503(
        self,
    ) -> None:
        exc = model_service_http_exception(
            APIConnectionError()
        )

        self.assertEqual(
            exc.status_code,
            503,
        )

    def test_other_upstream_status_is_502(
        self,
    ) -> None:
        exc = model_service_http_exception(
            APIStatusError()
        )

        self.assertEqual(
            exc.status_code,
            502,
        )

    def test_unknown_pipeline_error_is_502(
        self,
    ) -> None:
        exc = model_service_http_exception(
            RuntimeError(
                "internal model trace"
            )
        )

        self.assertEqual(
            exc.status_code,
            502,
        )
        self.assertNotIn(
            "internal model trace",
            str(exc.detail),
        )


if __name__ == "__main__":
    unittest.main()