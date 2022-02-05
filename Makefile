export PYTHONPATH := $(shell pwd)/src

.PHONY: test

lint-ipynb:
	nbqa mypy --config-file=./mypy.ini notebooks/*.ipynb

lint-py:
	mypy --config-file=./mypy.ini src tests
	pyflakes src tests

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
	pytest -v tests

test-focused:
	pytest -v -m focus tests

all: fmt lint test
