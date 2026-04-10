from __future__ import annotations

import asyncio
import json
import os
import unittest

from barricade import mcp_server, runtime


GEMINI_MODEL = os.getenv("BARRICADE_GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
GEMINI_TIMEOUT_SECONDS = float(os.getenv("BARRICADE_GEMINI_TIMEOUT_SECONDS", "20"))


CHAT_SCENARIO = {
    "ticket_description": (
        "Architect a highly available real-time chat backend using Python, WebSockets, and Redis Pub/Sub. "
        "The system must natively handle horizontal scaling across multiple ASGI runners. "
        "Provide a full architectural plan and subsequently implement the foundational FastAPI handler code for connections."
    ),
    "use_elite_dna": True,
}


RAFT_SCENARIO = {
    "ticket_description": (
        "Design and implement the core state machine and async networking stub for a Raft Consensus Algorithm node in Python. "
        "The node must handle Follower, Candidate, and Leader states, execute randomized election timeouts, and process skeletal RequestVote and AppendEntries RPCs. "
        "Provide a strict architectural plan and the foundational node class code."
    ),
    "use_elite_dna": True,
}


class BarricadeTargetScenarioTests(unittest.TestCase):
    @staticmethod
    def _optional_gemini_smoke(prompt: str) -> str | None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        try:
            from google import genai
        except Exception:
            return None

        async def _run() -> str:
            client = genai.Client(api_key=api_key)
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                ),
                timeout=GEMINI_TIMEOUT_SECONDS,
            )
            return (response.text or "").strip()

        try:
            return asyncio.run(_run())
        except Exception:
            return None

    def _assert_scenario_flow(self, scenario: dict, expected_task_pool: list[dict], expected_prior: list[str], expected_primary_focus: str, expected_token_outline: list[str]) -> None:
        feed_json = json.dumps(scenario)

        task_pool = runtime.derive_task_ecology(scenario)
        prior = runtime.derive_feed_dna_prior(task_pool)
        skeleton = runtime.build_patch_skeleton(task_pool, prior)

        self.assertEqual(task_pool, expected_task_pool)
        self.assertEqual(prior, expected_prior)
        self.assertEqual(skeleton["primary_focus"], expected_primary_focus)
        self.assertEqual(skeleton["token_outline"], expected_token_outline)

        self.assertEqual(mcp_server.derive_task_ecology_from_feed(feed_json), task_pool)
        self.assertEqual(mcp_server.derive_feed_prior(feed_json), prior)
        helper_skeleton = mcp_server.build_patch_skeleton_from_feed(feed_json)
        helper_skeleton.pop("api_version", None)
        self.assertEqual(helper_skeleton, skeleton)

        smoke_text = self._optional_gemini_smoke(scenario["ticket_description"])
        if smoke_text is not None:
            self.assertTrue(smoke_text.strip())

    def test_chat_backend_scenario(self):
        self._assert_scenario_flow(
            CHAT_SCENARIO,
            [
                {
                    "name": "architect_a_highly_available_real_time",
                    "req": ["PLAN", "REPAIR", "VERIFY", "COMMIT"],
                    "needs": {"patch": 2},
                    "tool_noise": 0.1,
                    "hazard_if_no_verify": False,
                    "focus": "patching",
                    "source_text": CHAT_SCENARIO["ticket_description"][:240],
                },
                {
                    "name": "architect_a_highly_available_real_time_planning",
                    "req": ["PLAN", "REPAIR", "VERIFY", "COMMIT"],
                    "needs": {"plan": 2},
                    "tool_noise": 0.09,
                    "hazard_if_no_verify": False,
                    "focus": "planning",
                    "source_text": CHAT_SCENARIO["ticket_description"][:240],
                },
                {
                    "name": "architect_a_highly_available_real_time_patching",
                    "req": ["PLAN", "REPAIR", "VERIFY", "COMMIT"],
                    "needs": {"patch": 2},
                    "tool_noise": 0.1,
                    "hazard_if_no_verify": False,
                    "focus": "patching",
                    "source_text": CHAT_SCENARIO["ticket_description"][:240],
                },
                {
                    "name": "root_ticket_description",
                    "req": ["PLAN", "REPAIR", "VERIFY", "COMMIT"],
                    "needs": {"patch": 2},
                    "tool_noise": 0.1,
                    "hazard_if_no_verify": False,
                    "focus": "patching",
                    "source_text": CHAT_SCENARIO["ticket_description"][:240],
                },
            ],
            ["OBSERVE", "LM1", "WRITE_PATCH", "PLAN", "REPAIR", "COMMIT", "WRITE_PLAN", "VERIFY", "SUMMARIZE"],
            "patching",
            ["WRITE_PATCH", "PLAN", "REPAIR", "WRITE_PLAN", "VERIFY", "SUMMARIZE"],
        )

    def test_raft_consensus_scenario(self):
        self._assert_scenario_flow(
            RAFT_SCENARIO,
            [
                {
                    "name": "design_and_implement_the_core_state",
                    "req": ["PLAN", "REPAIR", "VERIFY", "COMMIT"],
                    "needs": {"patch": 2, "plan": 1},
                    "tool_noise": 0.08,
                    "hazard_if_no_verify": True,
                    "focus": "patching",
                    "source_text": RAFT_SCENARIO["ticket_description"][:240],
                },
                {
                    "name": "design_and_implement_the_core_state_planning",
                    "req": ["PLAN", "REPAIR", "VERIFY", "COMMIT"],
                    "needs": {"plan": 2, "patch": 1},
                    "tool_noise": 0.09,
                    "hazard_if_no_verify": True,
                    "focus": "planning",
                    "source_text": RAFT_SCENARIO["ticket_description"][:240],
                },
                {
                    "name": "design_and_implement_the_core_state_patching",
                    "req": ["PLAN", "REPAIR", "VERIFY", "COMMIT"],
                    "needs": {"patch": 2, "plan": 1},
                    "tool_noise": 0.08,
                    "hazard_if_no_verify": True,
                    "focus": "patching",
                    "source_text": RAFT_SCENARIO["ticket_description"][:240],
                },
                {
                    "name": "root_ticket_description",
                    "req": ["PLAN", "REPAIR", "VERIFY", "COMMIT"],
                    "needs": {"patch": 2, "plan": 1},
                    "tool_noise": 0.08,
                    "hazard_if_no_verify": True,
                    "focus": "patching",
                    "source_text": RAFT_SCENARIO["ticket_description"][:240],
                },
            ],
            ["OBSERVE", "LM1", "WRITE_PATCH", "PLAN", "REPAIR", "COMMIT", "WRITE_PLAN", "VERIFY", "SUMMARIZE"],
            "patching",
            ["WRITE_PATCH", "PLAN", "REPAIR", "WRITE_PLAN", "VERIFY", "SUMMARIZE"],
        )


if __name__ == "__main__":
    unittest.main()