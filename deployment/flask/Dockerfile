FROM python:3.7-stretch

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

RUN python -m pytest

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
