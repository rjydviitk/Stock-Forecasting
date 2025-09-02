.PHONY: install run lint test docker-build docker-run clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest flake8

run:
	streamlit run dashboard/app.py

lint:
	flake8 . --max-line-length=120 --extend-exclude=.venv,venv

test:
	pytest -q

docker-build:
	docker build -t stock-forecasting:latest .

docker-run:
	docker run --rm -p 8501:8501 stock-forecasting:latest

clean:
	rm -rf dist build **/__pycache__ *.egg-info
