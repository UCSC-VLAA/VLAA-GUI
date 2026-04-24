from botocore.exceptions import ClientError

from vlaa_gui.core.engine import (
    DEFAULT_MAX_TOKENS,
    BedrockRateLimitError,
    LMMEngineAnthropicBedrock,
)


class FakeBedrockClient:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.response = response or {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
        }
        self.error = error

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def test_bedrock_generate_uses_converse_api_with_messages_and_inference_config():
    client = FakeBedrockClient()
    engine = LMMEngineAnthropicBedrock(
        model="anthropic-test",
        thinking=False,
        temperature=0.2,
        top_p=0.9,
    )
    engine._keys = [None]
    engine._get_or_create_client = lambda _: client

    response = engine.generate(
        [
            {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,aW1hZ2U=",
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "Prior answer"},
        ],
        max_new_tokens=32,
    )

    assert response == "ok"
    assert engine.last_usage == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert client.calls == [
        {
            "modelId": "anthropic-test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "Describe this."},
                        {
                            "image": {
                                "format": "png",
                                "source": {"bytes": b"image"},
                            }
                        },
                    ],
                },
                {"role": "assistant", "content": [{"text": "Prior answer"}]},
            ],
            "system": [{"text": "Be brief."}],
            "inferenceConfig": {
                "maxTokens": 32,
                "temperature": 0.2,
                "topP": 0.9,
            },
        }
    ]


def test_bedrock_generate_with_thinking_uses_additional_model_request_fields():
    client = FakeBedrockClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "Reasoned briefly."}
                            }
                        },
                        {"text": "The answer."},
                    ]
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
        }
    )
    engine = LMMEngineAnthropicBedrock(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        thinking=True,
    )
    engine._keys = [None]
    engine._get_or_create_client = lambda _: client

    response = engine.generate_with_thinking(
        [{"role": "user", "content": "Solve it."}],
        max_new_tokens=256,
    )

    assert "<thoughts>\nReasoned briefly.\n</thoughts>" in response
    assert "<answer>\nThe answer.\n</answer>" in response
    assert client.calls[0] == {
        "modelId": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "messages": [{"role": "user", "content": [{"text": "Solve it."}]}],
        "inferenceConfig": {"maxTokens": 256},
        "additionalModelRequestFields": {
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_MAX_TOKENS,
            }
        },
    }


def test_bedrock_converse_rotates_credentials_on_throttling():
    throttling_error = ClientError(
        {
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Rate exceeded",
            }
        },
        "Converse",
    )
    first_client = FakeBedrockClient(error=throttling_error)
    second_client = FakeBedrockClient(
        response={"output": {"message": {"content": [{"text": "rotated"}]}}}
    )
    engine = LMMEngineAnthropicBedrock(
        model="anthropic-test",
        aws_keys=[
            {"aws_access_key_id": "a", "aws_secret_access_key": "s"},
            {"aws_access_key_id": "b", "aws_secret_access_key": "s"},
        ],
        thinking=False,
    )
    clients = [first_client, second_client]
    engine._get_or_create_client = lambda _: clients.pop(0)

    response = engine.generate([{"role": "user", "content": "Hi"}])

    assert response == "rotated"
    assert first_client.calls
    assert second_client.calls


def test_bedrock_converse_raises_rate_limit_when_all_credentials_throttle():
    throttling_error = ClientError(
        {
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Rate exceeded",
            }
        },
        "Converse",
    )
    engine = LMMEngineAnthropicBedrock(
        model="anthropic-test",
        thinking=False,
    )
    engine._keys = [None]
    engine._get_or_create_client = lambda _: FakeBedrockClient(error=throttling_error)

    try:
        engine.generate([{"role": "user", "content": "Hi"}])
    except BedrockRateLimitError:
        pass
    else:
        raise AssertionError("Expected BedrockRateLimitError")
