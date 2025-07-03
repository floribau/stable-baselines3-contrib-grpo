SHELL=/bin/bash
MAIN_FILES=quick_train.py experiment_rliable_train.py experiment_rliable_all_plots.py experiment.py experiment_eval.py
LINT_PATHS=sb3_contrib/ tests/ setup.py docs/conf.py experiments/ ${MAIN_FILES}

pytest:
	./scripts/run_tests.sh

mypy:
	mypy ${LINT_PATHS}

type: mypy

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff check ${LINT_PATHS} --select=E9,F63,F7,F82 --output-format=full
	# exit-zero treats all errors as warnings.
	ruff check ${LINT_PATHS} --exit-zero --output-format=concise

format:
	# Sort imports
	ruff check --select I ${LINT_PATHS} --fix
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	ruff check --select I ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}


commit-checks: format type lint

doc:
	cd docs && make html

spelling:
	cd docs && make spelling

# PyPi package release
release:
	python -m build
	twine upload dist/*

# Test PyPi package release
test-release:
	python -m build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: lint format check-codestyle commit-checks doc spelling
