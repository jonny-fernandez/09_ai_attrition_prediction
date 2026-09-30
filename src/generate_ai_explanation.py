import json
import os

import pandas as pd
from dotenv import load_dotenv

from src.ai_explanation import (
    build_ai_fact_package,
    build_ai_instructions,
    build_ai_schema,
    render_ai_markdown,
    save_json,
    save_text,
    validate_ai_output,
)


def main():
    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------
    load_dotenv()

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Add it to your local .env file. "
            "Do not commit the key to Git."
        )

    model_name = os.getenv(
        "OPENAI_MODEL",
        "gpt-6-astra",
    )

    # ---------------------------------------------------------
    # Load verified deterministic artifacts only.
    #
    # No raw employee-level dataset is sent to the API.
    # ---------------------------------------------------------
    with open(
        "artifacts/model_metadata.json",
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    with open(
        "outputs/final_test_metrics.json",
        "r",
        encoding="utf-8",
    ) as f:
        final_metrics = json.load(f)

    top_risk = pd.read_csv(
        "outputs/"
        "final_test_top_risk.csv"
    )

    permutation_importance = pd.read_csv(
        "outputs/"
        "validation_permutation_importance.csv"
    )

    shap_summary = pd.read_csv(
        "outputs/"
        "validation_shap_summary.csv"
    )

    subgroup_metrics = pd.read_csv(
        "outputs/"
        "validation_subgroup_metrics.csv"
    )

    fact_package = (
        build_ai_fact_package(
            metadata=metadata,
            final_metrics=final_metrics,
            top_risk=top_risk,
            permutation_importance=permutation_importance,
            shap_summary=shap_summary,
            subgroup_metrics=subgroup_metrics,
        )
    )

    # ---------------------------------------------------------
    # OpenAI Responses API
    #
    # AI explains existing results only.
    # ---------------------------------------------------------
    from openai import OpenAI

    client = OpenAI()

    response = client.responses.create(
        model=model_name,
        instructions=(
            build_ai_instructions()
        ),
        input=json.dumps(
            fact_package,
            indent=2,
        ),
        text={
            "format": (
                build_ai_schema()
            )
        },
        store=False,
    )

    if response.status != "completed":
        raise RuntimeError(
            "OpenAI response did not complete successfully. "
            f"Status: {response.status}"
        )

    if not response.output_text:
        raise RuntimeError(
            "OpenAI returned no text output."
        )

    payload = json.loads(
        response.output_text
    )

    validate_ai_output(
        payload
    )

    # ---------------------------------------------------------
    # Persist AI outputs separately from deterministic results
    # ---------------------------------------------------------
    json_path = (
        "outputs/"
        "ai_model_interpretation.json"
    )

    markdown_path = (
        "outputs/"
        "ai_model_interpretation.md"
    )

    save_json(
        payload=payload,
        output_path=json_path,
    )

    markdown = render_ai_markdown(
        payload=payload,
        model_name=model_name,
    )

    save_text(
        text=markdown,
        output_path=markdown_path,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(
        "Project 9 - Controlled AI Interpretation"
    )
    print(
        "=" * 40
    )

    print(
        f"API model: {model_name}"
    )

    print()

    print(
        "AI received aggregate deterministic "
        "model results only."
    )

    print(
        "No employee-level rows were sent."
    )

    print(
        "AI did not generate predictions, metrics, "
        "thresholds, SHAP values, or subgroup metrics."
    )

    print()

    print(
        f"Saved: {json_path}"
    )

    print(
        f"Saved: {markdown_path}"
    )

    print()

    print(
        "Executive summary:"
    )

    print(
        payload[
            "executive_summary"
        ]
    )


if __name__ == "__main__":
    main()