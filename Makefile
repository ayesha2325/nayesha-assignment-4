FLASK_APP = app.py
VENV = venv

install:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

run:
	FLASK_APP=$(FLASK_APP) FLASK_ENV=development ./$(VENV)/bin/flask run --port 3000
