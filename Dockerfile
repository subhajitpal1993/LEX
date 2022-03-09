FROM python:3.8-slim-buster
ADD . /Python_Docker
WORKDIR /Python_Docker
CMD ["python", "app.py"]