.PHONY : docs doctests lint tests

all : lint doctests docs tests

clean :
	rm -rf docs/_build htmlcov
	rm .coverage*

requirements.txt : requirements.in setup.py
	pip-compile -v

docs :
	rm -rf docs/_build
	sphinx-build -W . docs/_build

doctests :
	rm -rf docs/_build
	sphinx-build -W -b doctest . docs/_build

tests :
	rm -f .coverage*
	pytest -v --cov=snippets --cov-report=term-missing --cov-fail-under=100

lint:
	black --check .

MODULE_PATHS = $(filter-out __%,$(notdir $(wildcard snippets/*)))
MODULES = ${MODULE_PATHS:.py=}

$(addprefix coverage/,${MODULES}) : coverage/% :
	pytest -v --cov=snippets.$* --cov-report=term-missing tests/test_$*.py
