"""Deterministic calculator tool (Tier-0 tool route)."""
from app.tools import calculator


def test_basic_arithmetic():
    r = calculator.solve("what is 2 + 2?")
    assert r["handled"] is True
    assert r["answer"] == "4"


def test_multiplication_word_form():
    r = calculator.solve("12 times 12")
    assert r["handled"] is True
    assert r["answer"] == "144"


def test_square_root():
    r = calculator.solve("square root of 144")
    assert r["handled"] is True
    assert r["answer"] == "12"


def test_power():
    r = calculator.solve("2 to the power of 10")
    assert r["handled"] is True
    assert r["answer"] == "1,024"


def test_declines_word_problem():
    r = calculator.solve("If Alice has 3 apples and eats one, is she happy?")
    assert r["handled"] is False


def test_declines_bare_number():
    assert calculator.solve("42")["handled"] is False


def test_declines_knowledge_question():
    assert calculator.solve("What is the capital of France?")["handled"] is False
