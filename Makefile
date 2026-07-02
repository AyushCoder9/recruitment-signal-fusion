PY := .venv/bin/python
DATA := [PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge

.PHONY: help eda precompute label train rank validate test charts sensitivity fairness deck ci clean

help:
	@echo "Targets: eda precompute label train rank validate test charts sensitivity fairness ci deck clean"

eda:
	$(PY) notebooks/eda.py

precompute:
	$(PY) precompute.py

label:
	$(PY) label_with_llm.py

train:
	$(PY) train_ranker.py

rank:
	$(PY) rank.py --candidates "$(DATA)/candidates.jsonl" --out submission/team_nullset.csv

validate:
	$(PY) "$(DATA)/validate_submission.py" submission/team_nullset.csv

test:
	$(PY) -m pytest -q tests/

charts:
	$(PY) scripts/gen_architecture.py
	$(PY) scripts/gen_report_charts.py

sensitivity:
	$(PY) scripts/gen_sensitivity.py

fairness:
	$(PY) scripts/gen_fairness.py

ci:
	$(PY) -c "from src.eval.bootstrap_ci import run_bootstrap_ci, ci_table_markdown; import json; r=run_bootstrap_ci(); print(ci_table_markdown(r)); json.dump(r, open('artifacts/bootstrap_ci.json','w'), indent=2)"

deck:
	$(PY) scripts/make_deck.py

# Safe: removes only regenerable heavy artifacts + outputs. Keeps labels.parquet,
# lgbm_ranker.txt and features.parquet (the expensive, hard-to-reproduce work).
clean:
	rm -f artifacts/embeddings.npz artifacts/bm25.pkl submission/team_nullset.csv submission/team_nullset.xlsx
