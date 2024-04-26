.PHONY: development_cheat_sheet
.PHONY: run

run:
	pip install -r requirements.txt
	streamlit run src/app.py

development_cheat_sheet:
	python3.10 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	pre-commit install
