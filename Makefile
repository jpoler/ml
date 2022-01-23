export PYTHONPATH := $(shell pwd)
export MYPYPATH := $(shell pwd)/prml

.PHONY: test

lint-ipynb:
	nbqa mypy --config-file=./mypy.ini notebooks/*.ipynb

lint-py:
	echo "$$MYPYPATH"
	mypy --config-file=./mypy.ini prml/*.py tests/*.py

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
