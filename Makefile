.PHONY: clean virtualenv test docker dist dist-upload

clean:
	find . -name '*.py[co]' -delete
	rm -rf *.lock *.dirlock worker-* build *.egg-info .pytest_* coverage-report .coverage*
	rm -rf pgdata

virtualenv:
	rm -rf venv/
	virtualenv --python python3 --prompt '|> ts-mule <| ' venv
	venv/bin/pip install -r requirements.txt
	venv/bin/python setup.py develop
	@echo
	@echo "VirtualENV Setup Complete. Now run: source venv/bin/activate"
	@echo

virtualenv-dev:
	rm -rf venv/
	virtualenv --python python3 --prompt '|> ts-mule <| ' venv
	venv/bin/pip install -r requirements-dev.txt
	venv/bin/python setup.py develop
	@echo
	@echo "VirtualENV Setup Complete. Now run: source venv/bin/activate"
	@echo

test:
	@echo "-------------"
	@echo "Running tests"
	@echo "-------------"
	@echo
	python -m pytest \
		-v \
		--cov=i-metro \
		--cov-report=term \
		--cov-report=html:coverage-report \
		tests/
zipfile:
	find . -not \
		-name env -o \
		-name "*.zip" -o \
		-name tests -o

setup:
	pip install -r requirements-dev.txt
	python setup.py develop

lint:
	@echo "---------------------------"
	@echo "Linting project source code"
	@echo "---------------------------"
	@echo
	flake8 --extend-ignore=E501 --exclude=env/
	@echo

html-docs:
	@echo "---------------------------"
	@echo "Building html documentation"
	@echo "---------------------------"
	@echo
	make -C docs html
	@echo

docstyle:
	pydocstyle

wheel: clean
	python setup.py bdist_wheel -d whl

docker: clean
	docker build -t me/miniconda:latest .

clean-docker:
	docker stop $(docker container ls -aq)
	docker rm $(docker container ls -aq)
	docker rmi $(docker images --filter dangling=true -q --no-trunc)

jupyter:
	docker run -p 8888:8888 me/miniconda