export PYTHONPATH := $(shell pwd)

.PHONY: test

lint-ipynb:
	nbqa mypy --strict notebooks/*.ipynb

lint-py:
	mypy --strict prml/*.py
	mypy --strict tests/*.py

lint: lint-py lint-ipynb

fmt-ipynb:
	nbqa black notebooks/*.ipynb
	nbqa isort notebooks/*.ipynb
	nbqa pyupgrade notebooks/*.ipynb --py36-plus
	nbqa mdformat notebooks/*.ipynb --nbqa-md --nbqa-diff

fmt-py:
	nbqa black notebooks/*.ipynb
	nbqa isort notebooks/*.ipynb
	nbqa pyupgrade notebooks/*.ipynb --py36-plus
	nbqa mdformat notebooks/*.ipynb --nbqa-md --nbqa-diff

fmt: fmt-ipynb fmt-py

test:
	pytest tests

env:
	echo "$$PYTHONPATH"

all: fmt lint test
