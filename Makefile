.PHONY: clean virtualenv test docker dist dist-upload

project = 'tsmule'
clean:
	@echo "-------------"
	@echo "Cleaning		"
	@echo "-------------"
	find . -name '*.py[co]' -delete
	find . -name '*__pycache__' -delete
	rm -rf *.lock *.dirlock worker-* .pytest_* coverage-report .coverage*
	rm -rf docs/build build whl/
	rm -rf $(project)/*.egg-info
virtualenv:
	rm -rf venv/
	virtualenv --python python3 --prompt '|> $(project) <| ' venv
	venv/bin/pip install -r requirements.txt
	venv/bin/python setup.py develop
	@echo
	@echo "VirtualENV Setup Complete. Now run: source venv/bin/activate"
	@echo

virtualenv-dev:
	rm -rf venv/
	virtualenv --python python3 --prompt '|> $(project) <| ' venv
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
		--cov=$(project) \
		--cov-report=term \
		--cov-report=html:.coverage-report \
		tests/
		
simple-test:
		python -m pytest tests/

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
	flake8 --extend-ignore=E501 \
		--exclude=env/ \
		--exclude=examples/
	@echo

html-docs: clean
	@echo "---------------------------"
	@echo "Building html documentation"
	@echo "---------------------------"
	@echo
	make -C docs html
	@echo

docstyle:
	pydocstyle $(project)

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

