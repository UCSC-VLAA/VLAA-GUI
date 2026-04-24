from types import SimpleNamespace

from vlaa_gui.core import engine as engine_module
from vlaa_gui.core.engine import LMMEngineAnthropicBedrockMantle
from vlaa_gui.core.mllm import LMMAgent
from vlaa_gui.core.mllm import ENGINE_REGISTRY


class FakeMantleMessages:
    def __init__(self, response):
        self.calls = []
        self.response = response

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeMantleClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = FakeMantleMessages(
            SimpleNamespace(
                content=[SimpleNamespace(type="text", text="mantle answer")],
                usage=SimpleNamespace(input_tokens=7, output_tokens=11),
            )
        )
        self.__class__.instances.append(self)


def test_anthropic_bedrock_mantle_is_registered(monkeypatch):
    monkeypatch.setattr(engine_module, "AnthropicBedrockMantle", FakeMantleClient)

    engine = ENGINE_REGISTRY["anthropic_bedrock_mantle"](
        model="anthropic.claude-opus-4-7",
        region="us-east-1",
        aws_keys=[
            {
                "aws_access_key_id": "akid",
                "aws_secret_access_key": "secret",
                "aws_session_token": "token",
            }
        ],
    )

    response = engine.generate(
        [
            {"role": "system", "content": [{"type": "text", "text": "Be exact."}]},
            {"role": "user", "content": "Hello"},
        ],
        temperature=0.3,
        max_new_tokens=256,
    )

    assert response == "mantle answer"
    client = FakeMantleClient.instances[0]
    assert client.kwargs == {
        "aws_region": "us-east-1",
        "aws_access_key": "akid",
        "aws_secret_key": "secret",
        "aws_session_token": "token",
    }
    assert client.messages.calls == [
        {
            "model": "anthropic.claude-opus-4-7",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 256,
            "system": "Be exact.",
            "temperature": 0.3,
        }
    ]
    assert engine.last_usage == {
        "prompt_tokens": 7,
        "completion_tokens": 11,
        "total_tokens": 18,
    }


def test_anthropic_bedrock_mantle_uses_adaptive_thinking_for_opus_47(monkeypatch):
    monkeypatch.setattr(engine_module, "AnthropicBedrockMantle", FakeMantleClient)
    engine = ENGINE_REGISTRY["anthropic_bedrock_mantle"](
        model="anthropic.claude-opus-4-7",
        region="us-east-1",
        thinking=True,
    )

    response = engine.generate_with_thinking(
        [{"role": "user", "content": "Solve this."}],
        max_new_tokens=512,
    )

    assert response == (
        "<thoughts>\n\n</thoughts>\n\n<answer>\nmantle answer\n</answer>\n"
    )
    client = FakeMantleClient.instances[-1]
    assert client.messages.calls[0]["thinking"] == {"type": "adaptive"}
    assert client.messages.calls[0]["max_tokens"] == 512


def test_lmm_agent_formats_mantle_messages_like_anthropic():
    engine = LMMEngineAnthropicBedrockMantle(
        model="anthropic.claude-opus-4-7",
        region="us-east-1",
    )
    agent = LMMAgent(engine=engine, system_prompt="System prompt")

    agent.add_message("Look at this", image_content=b"image-bytes", role="user")

    assert agent.messages[-1] == {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aW1hZ2UtYnl0ZXM=",
                },
            },
            {"type": "text", "text": "Look at this"},
        ],
    }
