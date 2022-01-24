export PYTHONPATH := $(shell pwd)

.PHONY: test

lint-ipynb:
	nbqa mypy --config-file=./mypy.ini notebooks/*.ipynb

lint-py:
	mypy --config-file=./mypy.ini prml tests

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

all: fmt lint test
