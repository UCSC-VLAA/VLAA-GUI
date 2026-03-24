import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from osworld_setup.cli_args import add_and_finalize_args


def test_finalize_sets_aws_password_and_model_dir(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--provider_name",
            "aws",
            "--model",
            "global.anthropic.claude-sonnet-4-6",
        ],
    )
    parser = argparse.ArgumentParser()
    args = add_and_finalize_args(parser, include_num_envs=True)

    assert args.client_password == "osworld-public-evaluation"
    assert args.model_dir_name == "claude-sonnet-4-6"


def test_finalize_normalizes_lists_and_reasoning(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--region",
            "us-west-2",
            "--model",
            "gpt-5",
            "--model_api_keys",
            "k1,k2",
            "--model_aws_keys",
            "a1,a2",
            "--coding_model_api_keys",
            "c1,c2",
            "--searcher_api_keys",
            "s1,s2",
            "--search_engine",
            "None",
            "--model_thinking",
            "--model_thinking_level",
            "high",
        ],
    )
    parser = argparse.ArgumentParser()
    args = add_and_finalize_args(parser, include_num_envs=True)

    assert args.model_api_keys == ["k1", "k2"]
    assert args.model_aws_keys == ["a1", "a2"]
    assert args.coding_model_api_keys == ["c1", "c2"]
    assert args.searcher_api_keys == ["s1", "s2"]
    assert args.search_engine is None
    assert args.model_region == "us-west-2"
    assert args.grounding_model_region == "us-west-2"
    assert args.model_reasoning_effort == "high"
    assert args.reflection_thinking is True
