python = titanic_env/bin/python
pip = titanic_env/bin/pip

setup:
		python3 -m venv titanic_env
		$(python) -m pip install --upgrade pip
		$(pip) install -r requirements.txt

train:
		$(python) main.py

fastapi:
		sudo docker build -t titanic_fastapi .
		sudo docker run -d -p 8000:8000 titanic_fastapi > docker_id.txt
		
test:

		$(python) -m pytest

remove:
		sudo docker stop $(shell head -n 1 docker_id.txt)
		sudo docker rm $(shell head -n 1 docker_id.txt)
		rm -rf titanic_env
		rm -rf mlruns
		rm -rf models
		rm -rf __pycache__
		rm -rf example_log.log
		rm -rf Conf_matrix.png
		rm -rf docker_id.txt
		rm -rf .pytest_cache

