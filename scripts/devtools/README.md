# Manual dev checks — not the test suite

These are ad-hoc scripts for poking at live infrastructure by hand. They are
**not** automated tests: several require Kafka, Neo4j, or a downloaded model,
and none assert anything.

The real suite is `tests/`, runs offline, and is what CI should gate on:

    pytest tests/ -q

They live here rather than in `scripts/` because filenames beginning `test_`
next to a `tests/` directory invite exactly the confusion of running the wrong
thing and believing the project is verified when it is not.

| script | needs | does |
|---|---|---|
| `test_neo4j_connection.py` | live Neo4j | connectivity smoke check |
| `test_indicer.py` | HF model download | prints raw IndicNER output |
| `test_native_script.py` | none | prints normalizer behaviour on sample text |
| `test_stix_formatter.py` | none | prints a STIX bundle |
| `simulate_pipeline.py` | Kafka | pushes synthetic messages onto the raw topic |
| `debug.py` | varies | scratch |
