lint-ipynb:
	nbqa mypy *.ipynb

lint-py:
	mypy *.py

lint: lint-ipynb lint-py

fmt-ipynb:
	nbqa black *.ipynb
	nbqa isort *.ipynb
	nbqa pyupgrade *.ipynb --py36-plus
	nbqa mdformat *.ipynb --nbqa-md --nbqa-diff

fmt-py:
	nbqa black *.ipynb
	nbqa isort *.ipynb
	nbqa pyupgrade *.ipynb --py36-plus
	nbqa mdformat *.ipynb --nbqa-md --nbqa-diff

fmt: fmt-ipynb fmt-py
