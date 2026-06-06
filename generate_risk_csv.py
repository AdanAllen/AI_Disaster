"""Retired legacy ZIP-score generator.

This script previously produced generalized risk scores, low-risk language, and
chatbot prompts. Those outputs do not meet StayReady's current provenance and
Action Library requirements. It is intentionally non-executable so it cannot be
mistaken for the supported data pipeline.
"""


def main():
    raise SystemExit(
        "Retired: use reviewed geospatial evidence and the Action Library. "
        "This script must not generate public-facing risk claims or advice."
    )


if __name__ == "__main__":
    main()
