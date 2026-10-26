# Local multimodal RAG: actual inference results

Model: Qwen3-VL-2B-Instruct Q4_K_M with Q8 image projector, llama.cpp CPU.

Retrieval: MiniLM dense embeddings + sparse BM25-style vectors, Qdrant RRF, MS MARCO cross-encoder reranking.

Chart PDF text layer: empty. No mocked model outputs.

| Case | Modality | Correct source | Correct answer | Images | Result |
|---|---|---|---|---|---|
| text_code | text | True | True | 0 | PASS |
| text_deadline | text | True | True | 0 | PASS |
| table_north | table | True | True | 1 | PASS |
| table_south | table | True | True | 1 | PASS |
| table_difference | table | True | True | 1 | PASS |
| chart_trend | chart | True | True | 1 | PASS |
| chart_maximum | chart | True | True | 1 | PASS |
| chart_value | chart | True | True | 1 | PASS |
| unsupported | unsupported | True | True | 0 | PASS |

## text_code

What is the annual review code?

ORCHID
Sources: [d78debb8-8060-5bca-a275-f2842d07f0c7]

## text_deadline

Within how many days must expense reports be submitted?

14
Sources: [d78debb8-8060-5bca-a275-f2842d07f0c7]

## table_north

What is the North regional budget in million USD?

73
Sources: [618b20a7-846f-59ea-a50a-f376691623f6]

## table_south

What is the South regional budget in million USD?

46
Sources: [618b20a7-846f-59ea-a50a-f376691623f6]

## table_difference

How much larger is the North budget than the South budget in million USD?

27
Sources: [618b20a7-846f-59ea-a50a-f376691623f6]

## chart_trend

Does quarterly revenue increase or decrease from Q1 to Q4?

Increases
Sources: [62f5df09-571b-5909-b1d2-8e5ac1d1fa2f]

## chart_maximum

Which quarter has the highest revenue in the chart?

Q4
Sources: [62f5df09-571b-5909-b1d2-8e5ac1d1fa2f]

## chart_value

What is Q4 revenue in million USD in the chart?

40
Sources: [62f5df09-571b-5909-b1d2-8e5ac1d1fa2f]

## unsupported

What is the chief executive's private phone number?

No evidence found in this domain.
