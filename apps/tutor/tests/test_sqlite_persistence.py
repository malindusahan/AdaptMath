import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.api import tutor as tutor_api
from app.core.persistence import tutor_checkpointer


class _MiniState(TypedDict, total=False):
    prompt: str
    learner_answer: str


def _wait_for_answer(
    state: _MiniState,
) -> dict[str, str]:
    answer = interrupt(
        {
            "type": "assessment_required",
            "question": state["prompt"],
        }
    )

    return {
        "learner_answer": str(answer),
    }


def _build_mini_graph(
    saver: SqliteSaver,
):
    builder = StateGraph(
        _MiniState
    )
    builder.add_node(
        "wait",
        _wait_for_answer,
    )
    builder.add_edge(
        START,
        "wait",
    )
    builder.add_edge(
        "wait",
        END,
    )
    return builder.compile(
        checkpointer=saver
    )


class SQLitePersistenceTests(unittest.TestCase):
    def test_runtime_uses_sqlite_checkpointer(self):
        self.assertIsInstance(
            tutor_checkpointer,
            SqliteSaver,
        )

    def test_tutor_api_has_no_process_memory_session_registry(self):
        source = Path(
            tutor_api.__file__
        ).read_text(
            encoding="utf-8"
        )
        self.assertNotIn(
            "_session_status",
            source,
        )

    def test_workflow_has_no_inmemory_checkpointer(self):
        workflow_path = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "graph"
            / "workflow.py"
        )
        source = workflow_path.read_text(
            encoding="utf-8"
        )
        self.assertNotIn(
            "InMemorySaver",
            source,
        )
        self.assertIn(
            "tutor_checkpointer",
            source,
        )

    def test_checkpoint_survives_graph_rebuild_and_resumes_interrupt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = (
                Path(temp_dir)
                / "restart_test.sqlite3"
            )
            config = {
                "configurable": {
                    "thread_id": "restart-thread",
                }
            }

            connection_one = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
            )
            saver_one = SqliteSaver(
                connection_one
            )
            saver_one.setup()
            graph_one = _build_mini_graph(
                saver_one
            )

            paused = graph_one.invoke(
                {
                    "prompt": "What is 2 + 2?",
                },
                config=config,
            )
            self.assertIn(
                "__interrupt__",
                paused,
            )
            connection_one.close()

            # Simulate a fresh backend process rebuilding the same
            # graph against the same on-disk checkpoint database.
            connection_two = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
            )
            saver_two = SqliteSaver(
                connection_two
            )
            saver_two.setup()
            graph_two = _build_mini_graph(
                saver_two
            )

            resumed = graph_two.invoke(
                Command(
                    resume="4"
                ),
                config=config,
            )
            self.assertEqual(
                resumed[
                    "learner_answer"
                ],
                "4",
            )
            connection_two.close()


if __name__ == "__main__":
    unittest.main()
