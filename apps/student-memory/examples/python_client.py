"""Ready-to-use Python client for the Student Personalization Memory Service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests


class MemoryClient:
    """Client for interacting with the Student Personalization Memory REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        service_key: str = "development-secret-key",
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.service_key = service_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-Service-Key": self.service_key,
        })

    def classify_topic(self, question: str) -> Dict[str, Any]:
        """
        Classify a student question into a single canonical topic.

        Returns:
            dict with 'topic', 'skill_id', 'confidence', 'is_math', 'model_version'
        """
        url = f"{self.base_url}/topic/classify"
        resp = self.session.post(url, json={"question": question}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def update_memory(
        self,
        student_id: str,
        topic: str,
        assessment_questions: List[Dict[str, Any]],
        identified_errors: Optional[List[str]] = None,
        overall_feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send Evaluator assessment results to update student memory.

        Returns:
            dict with 'learning_state', 'evidence_level', 'misconceptions', 'memory_updated'
        """
        url = f"{self.base_url}/memory/update"
        payload = {
            "student_id": student_id,
            "topic": topic,
            "assessment_questions": assessment_questions,
            "identified_errors": identified_errors or [],
            "overall_feedback": overall_feedback,
        }
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_tutor_context(self, student_id: str) -> Dict[str, Any]:
        """Retrieve focused personalization context for Tutor agent generation."""
        url = f"{self.base_url}/memory/{student_id}/tutor-context"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_planner_context(self, student_id: str) -> Dict[str, Any]:
        """Retrieve focused student context for Planner curriculum selection."""
        url = f"{self.base_url}/memory/{student_id}/planner-context"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    client = MemoryClient()
    print("Testing Topic Classification...")
    result = client.classify_topic("Solve 3x + 5 = 20.")
    print("Classify Result:", result)
