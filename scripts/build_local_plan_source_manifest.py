#!/usr/bin/env python3
"""Record reproducible local-plan source metadata without approving claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/lhmp/plan_registry.json"
OUTPUT = ROOT / "data/lhmp/source_manifest.json"


def build() -> dict:
    plans = json.loads(REGISTRY.read_text(encoding="utf-8"))
    sources = []
    for plan in plans:
        path = ROOT / plan["local_pdf_path"]
        raw = path.read_bytes()
        sources.append(
            {
                "jurisdiction": plan["city_display_name"],
                "slug": plan["city"],
                "plan_title": plan["plan_title"],
                "edition": str(plan["plan_year"]),
                "document_type": plan.get(
                    "document_type",
                    "local_hazard_mitigation_plan_or_annex_requires_review",
                ),
                "adoption_status": plan.get(
                    "adoption_status",
                    "adoption details require official-jurisdiction confirmation",
                ),
                "adoption_evidence": plan.get("adoption_evidence"),
                "official_page_url": plan["source_url"],
                "official_pdf_url": plan.get("official_pdf_url"),
                "local_pdf_path": plan["local_pdf_path"],
                "pdf_sha256": hashlib.sha256(raw).hexdigest(),
                "pdf_bytes": len(raw),
                "source_checked_at": plan.get("source_checked_at", "2026-07-27"),
                "supersedes": plan.get("supersedes"),
                "publication_status": "withheld_pending_named_claim_review",
                "notes": plan.get("notes", ""),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": "2026-07-27",
        "policy": (
            "Checksums and source identities are technical controls, not plan-status "
            "or claim approval. Drafts, annexes, and non-LHMP documents remain distinct."
        ),
        "jurisdiction_count": len(sources),
        "externally_approved_claim_count": 0,
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build()
    if args.check:
        current = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if current != payload:
            print("Local-plan source manifest is stale.")
            return 1
    else:
        OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Local-plan source manifest: {payload['jurisdiction_count']} jurisdictions; "
        "0 externally approved claims."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
