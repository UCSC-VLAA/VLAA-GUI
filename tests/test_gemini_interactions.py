from types import SimpleNamespace

from google.genai import types

from vlaa_gui.core.engine import (
    LMMEngineGemini,
    LMMEngineGeminiSearch,
)


class FakeInteractions:
    def __init__(self, outputs=None, usage=None):
        self.calls = []
        self.outputs = outputs or [SimpleNamespace(type="text", text="ok")]
        self.usage = usage

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(outputs=self.outputs, usage=self.usage)


class FakeClient:
    def __init__(self, interactions):
        self.interactions = interactions


def test_gemini_generate_uses_interactions_api_with_stateless_turns():
    interactions = FakeInteractions()
    client = FakeClient(interactions)
    engine = LMMEngineGemini(
        model="gemini-test",
        api_key="test-key",
        top_p=0.9,
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,aW1hZ2U=",
                        "detail": "high",
                    },
                },
            ],
        },
        types.Content(role="model", parts=[types.Part.from_text(text="Prior answer")]),
        {"role": "user", "content": "Next question"},
    ]

    response = engine._generate_with_genai(
        messages,
        temperature=0.2,
        max_new_tokens=32,
        genai_client=client,
    )

    assert response.outputs[-1].text == "ok"
    call = interactions.calls[0]
    assert "contents" not in call
    assert "config" not in call
    assert call["model"] == "gemini-test"
    assert call["system_instruction"] == "Be concise."
    assert call["store"] is False
    assert call["generation_config"] == {
        "temperature": 0.2,
        "max_output_tokens": 32,
        "top_p": 0.9,
    }
    assert call["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image",
                    "data": "aW1hZ2U=",
                    "mime_type": "image/png",
                    "resolution": "high",
                },
            ],
        },
        {"role": "model", "content": [{"type": "text", "text": "Prior answer"}]},
        {"role": "user", "content": [{"type": "text", "text": "Next question"}]},
    ]


def test_gemini_thinking_response_extracts_thoughts_answer_and_usage(monkeypatch):
    interactions = FakeInteractions(
        outputs=[
            SimpleNamespace(
                type="thought",
                summary=[SimpleNamespace(type="text", text="Used arithmetic.")],
            ),
            SimpleNamespace(type="text", text="The answer is 36."),
        ],
        usage=SimpleNamespace(
            total_input_tokens=3,
            total_output_tokens=4,
            total_tokens=7,
        ),
    )
    client = FakeClient(interactions)
    engine = LMMEngineGemini(
        model="gemini-test",
        api_keys=["test-key"],
        thinking=True,
        thinking_level="low",
    )
    monkeypatch.setattr(engine, "_get_or_create_genai_client", lambda _: client)

    response = engine.generate_with_thinking(
        [{"role": "user", "content": "What is 15% of 240?"}],
        max_new_tokens=64,
    )

    assert "<thoughts>\nUsed arithmetic.\n</thoughts>" in response
    assert "<answer>\nThe answer is 36.\n</answer>" in response
    assert engine.last_usage == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }
    assert interactions.calls[0]["generation_config"]["thinking_level"] == "low"
    assert interactions.calls[0]["generation_config"]["thinking_summaries"] == "auto"


def test_gemini_search_uses_interactions_google_search_tool(monkeypatch):
    interactions = FakeInteractions(
        outputs=[SimpleNamespace(type="text", text="search answer")]
    )
    client = FakeClient(interactions)
    engine = LMMEngineGeminiSearch(
        model="gemini-search-test",
        api_keys=["test-key"],
        top_p=0.8,
    )
    monkeypatch.setattr(engine, "_get_or_create_genai_client", lambda _: client)

    response = engine.generate(
        [
            {"role": "system", "content": "Use current sources."},
            {"role": "user", "content": "Who won?"},
        ],
        temperature=0.4,
        max_new_tokens=128,
    )

    assert response == "search answer"
    call = interactions.calls[0]
    assert call["model"] == "gemini-search-test"
    assert call["system_instruction"] == "Use current sources."
    assert call["tools"] == [{"type": "google_search"}]
    assert call["store"] is False
    assert call["generation_config"] == {
        "temperature": 0.4,
        "max_output_tokens": 128,
        "top_p": 0.8,
    }
    assert call["input"] == [
        {"role": "user", "content": [{"type": "text", "text": "Who won?"}]}
    ]
