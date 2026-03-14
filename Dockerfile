FROM python:3.10.6-slim

WORKDIR /prod

COPY requirements_prod.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY renal_sight renal_sight
COPY models models

CMD uvicorn renal_sight.api.fast:app --host 0.0.0.0 --port $PORT
