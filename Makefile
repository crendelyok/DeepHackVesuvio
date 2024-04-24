.PHONY: install

install:
	python3.10 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	pre-commit install
