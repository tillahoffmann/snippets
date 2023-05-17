.PHONY : docs tests

requirements.txt : requirements.in setup.py
	pip-compile -v --resolver=backtracking

docs :
	rm -rf docs/_build
	sphinx-build -W . docs/_build

doctests :
	rm -rf docs/_build
	sphinx-build -W -b doctest . docs/_build

tests :
	pytest -v --cov=snippets --cov-report=term-missing --cov-fail-under=100
