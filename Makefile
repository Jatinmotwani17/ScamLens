# ScamLens - common commands. Windows users: run these in Git Bash.
#
#   make setup   rebuild data + models (slow, needs network the first time)
#   make test    the full pytest suite
#   make eval    evaluation + calibration + Phase 4 + Phase 5 reports
#   make serve   run the API on http://127.0.0.1:8000
#   make scan    PII / secret scan of everything git can see
#   make gate    CI regression gate (test_b F1 + real-ham FPR)

PY ?= venv/Scripts/python.exe
ifeq ($(OS),)
	PY = python
endif

.PHONY: help setup test eval serve scan gate clean docker

help:
	@echo "setup  : rebuild dataset, baselines, calibration"
	@echo "test   : pytest tests -q"
	@echo "eval   : evaluate + calibrate + phase4 + suspicious band"
	@echo "serve  : uvicorn src.api:app on port 8000"
	@echo "scan   : scripts/scan_repo.py"
	@echo "gate   : scripts/ci_gate.py"
	@echo "docker : docker build -t scamlens:0.5 ."

setup:
	$(PY) -m src.synth.generate
	$(PY) -m src.synth.safe_v2
	$(PY) -m src.external_data
	$(PY) -m src.build_dataset
	$(PY) -m src.data_checks
	$(PY) -m src.baselines
	$(PY) -m src.nn_sim
	$(PY) -m src.calibrate --model verdict_lr_plus

test:
	$(PY) -m pytest tests -q

eval: 
	$(PY) -m src.evaluate
	$(PY) -m src.audit
	$(PY) -m src.calibrate --model verdict_lr_plus
	$(PY) -m src.novelty_metrics --model verdict_lr_plus
	$(PY) -m src.error_analysis --model verdict_lr_plus
	$(PY) -m src.robustness --model verdict_lr_plus
	$(PY) -m src.phase4 --model verdict_lr_plus
	$(PY) -m src.suspicious --model verdict_lr_plus

serve:
	$(PY) -m uvicorn src.api:app --host 127.0.0.1 --port 8000

scan:
	$(PY) scripts/scan_repo.py

gate:
	$(PY) scripts/ci_gate.py

docker:
	docker build -t scamlens:0.5 .

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
