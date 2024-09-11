ENV := env
PY_VERSION := 3.8.10

all:
	export PYENV_ROOT="$$HOME/.pyenv" && \
	export PATH="$$PYENV_ROOT/bin:$$PATH" && \
	eval "$$(pyenv init --path)" && \
	pyenv install ${PY_VERSION} -s && \
	pyenv local ${PY_VERSION} && \
	python -m pip install virtualenv && \
	virtualenv --quiet --python python3.8.10 ${ENV}
	${ENV}/bin/pip install --quiet -r  requirements_py38.txt

