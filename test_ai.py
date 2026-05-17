#!/usr/bin/env python3
"""Smoke tests for ai.py — verifies vLLM connection and all agent capabilities."""

import sys
import json
from pathlib import Path
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from ai import AIAgent, AIConfig, Chat, Text, Assistant, AgentResult, DoneEvent, StructuredOutput

MODEL = "/home/ng6309/datascience/santhosh/models/qwen3-6-27b"
CONFIG = AIConfig(base_url="http://192.168.170.76:8000/v1")


def test_task():
    """Simple blocking task."""
    agent = AIAgent(config=CONFIG)
    chat = agent.task("Reply with exactly: HELLO_WORLD", mode="instruct_reasoning", model=MODEL)
    assert "HELLO_WORLD" in chat.answer, f"Expected HELLO_WORLD, got: {chat.answer}"
    print("  task: OK")


def test_structured():
    """Pydantic structured output."""
    class Person(BaseModel):
        name: str
        age: int

    agent = AIAgent(config=CONFIG)
    result = agent.structured("Give me a person named Alice, age 30.", Person,
                              mode="instruct_reasoning", model=MODEL)
    assert isinstance(result, Person), f"Expected Person, got {type(result)}"
    assert result.name == "Alice" and result.age == 30, f"Got: {result}"
    print("  structured: OK")


def test_batch():
    """Parallel batch of prompts."""
    agent = AIAgent(config=CONFIG)
    results = agent.batch(
        ["What is 2+2?", "What is the capital of Japan?"],
        mode="instruct_reasoning", model=MODEL,
    )
    assert len(results) == 2
    assert "4" in results[0].answer
    assert "Tokyo" in results[1].answer
    print("  batch: OK")


def test_streaming_step():
    """Streaming step call."""
    agent = AIAgent(config=CONFIG)
    chat = Chat("Say one word: hi")
    chunks = []
    for event in agent.step(chat, mode="instruct_reasoning", model=MODEL):
        if isinstance(event, Text) and event.id is None:
            chunks.append(event.content)
    text = "".join(chunks)
    assert len(text.strip()) > 0, f"No text streamed"
    print("  streaming step: OK")


def test_forward_loop():
    """Forward (agentic loop with auto tool execution)."""
    agent = AIAgent(config=CONFIG)
    chat = Chat("What is 3 * 7? Just give the number.")
    events = list(agent.forward(chat, mode="instruct_reasoning", model=MODEL))
    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1, "Expected exactly one DoneEvent"
    assert chat.answer.strip() != "", "Empty answer from forward"
    print("  forward loop: OK")


def test_compress():
    """Conversation compression."""
    agent = AIAgent(config=CONFIG)
    chat = Chat()
    for i in range(6):
        chat.add(f"Tell me about topic {i}", role="user")
        chat.add(f"Topic {i} is about X, Y, Z with many details.", role="assistant")
    assert len(chat.messages) == 12
    compressed = agent.compress(chat, keep_last=2)
    assert len(compressed.messages) < len(chat.messages), "Compression should reduce messages"
    print("  compress: OK")


def test_evaluate():
    """Evaluation against a rubric."""
    agent = AIAgent(config=CONFIG)
    chat = agent.task("Explain recursion in one sentence.", mode="instruct_reasoning", model=MODEL)
    score = agent.evaluate(chat, rubric="Is the answer clear, accurate, and concise?")
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    print(f"  evaluate: OK (score={score:.3f})")


def test_structured_choice():
    """Structured output with choice constraint."""
    agent = AIAgent(config=CONFIG)
    chat = Chat("Is Python good? Answer: yes")
    for chunk in agent.step(chat, mode="instruct_reasoning", model=MODEL,
                            structured_output=StructuredOutput(choice=["yes", "no"])):
        pass
    print("  structured choice: OK")


if __name__ == "__main__":
    print("Running ai.py smoke tests...")
    test_task()
    test_structured()
    test_batch()
    test_streaming_step()
    test_forward_loop()
    test_compress()
    test_evaluate()
    test_structured_choice()
    print("\nAll tests passed!")
