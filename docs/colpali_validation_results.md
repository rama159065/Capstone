# ColPali Qdrant: actual model validation

Model: official vidore/colpali-v1.3-hf, CPU NF4 quantization. No synthetic vectors.

Storage: `indexes/multimodal_full/qdrant`; summary vectors plus 128-dimensional ColPali multivectors with MaxSim.

Collection counts: {'docs_text': 3, 'visual_pages': 2}

| Document | Page | Patch vectors | Dimension |
|---|---|---|---|
| chart_revenue | 1 | 1030 | 128 |
| table_budget | 1 | 1030 | 128 |

| Query | Top result | MaxSim | Result |
|---|---|---|---|
| What is the North regional budget in million USD? | table_budget | 18.98940996464171 | PASS |
| Which quarter has the highest revenue in the chart? | chart_revenue | 17.437717825329635 | PASS |

Domain isolation: True. Idempotent backfill: True. Reopened database persistence: True.

Merged text/visual retrieval: True. Visual FastAPI endpoint: True.
