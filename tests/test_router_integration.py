"""End-to-end routing on the mock provider: no keys, no network.

conftest forces the mock provider, so these exercise the real pipeline (tool
route, tier assignment, long-context flag, the escalation guard) without an API.
"""
import asyncio

import pytest

from app.router import memory, online
from app.router.router import TIER_NAMES, route_and_answer
from app.schemas import ChatRequest, ChatResponse


@pytest.fixture(autouse=True)
def _fresh_state():
    """Routing memory and the online controller are process-global; a test must
    not see what an earlier test recorded."""
    memory.reset()
    online.reset()


def _run(req) -> ChatResponse:
    return asyncio.run(route_and_answer(req))


def test_calculator_tool_route_is_free_and_exact():
    resp = _run(ChatRequest(message="what is 2 + 2?"))
    assert resp.route.tier_name == "tool"
    assert resp.status == "OK"
    assert resp.cost.est_cost_usd == 0.0
    assert resp.cost.llm_calls == 0
    assert "4" in resp.answer


def test_confident_question_ships_without_abstain():
    resp = _run(ChatRequest(message="What is the capital of France?"))
    assert resp.status == "OK"
    assert resp.route.abstained is False
    assert resp.route.tier in TIER_NAMES
    assert resp.model


def test_long_context_prompt_sets_the_flag():
    resp = _run(ChatRequest(message="summarize this: " + "word " * 20000))
    assert resp.route.long_context is True


def test_response_contract_is_wellformed():
    resp = _run(ChatRequest(message="What is the capital of Japan?"))
    assert resp.route.tier_name == TIER_NAMES[resp.route.tier]
    assert resp.cost.est_cost_usd >= 0.0
    assert resp.cost.routing_overhead_passes == max(0, resp.cost.llm_calls - 1)
    assert resp.provider == "mock"


def test_no_escalation_into_mock_target():
    """escalate_to names a real catalog model, but under force-mock it resolves
    to the mock provider. Escalating there would fake a big-model answer."""
    resp = _run(ChatRequest(message="What is the capital of Japan?", escalate_to="gpt-4o"))
    assert resp.route.escalated is False
