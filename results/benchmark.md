# ScamLens Phase 3 - model benchmark

CPU single-message latency over 200 `test_b` messages (median / p95), after 10 warmup runs.
Environment: Python 3.14.6, torch 2.14.1+cpu, Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 10 CPU thread(s).

| model | parameters | disk MB | median ms | p95 ms | notes |
|---|---|---|---|---|---|
| TF-IDF + LR | 22,250 | 1.00 | 1.80 | 4.03 | 22,249 TF-IDF features |

Parameters are trainable weights. Disk MB is the shipped artefact (checkpoint for a transformer, joblib for TF-IDF). The transformer tokenizer folder adds a few MB but is shared across runs. Latency is single-message, no batching - Phase 5 can batch to improve throughput.
