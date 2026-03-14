FROM python:3.10.6-slim

WORKDIR /prod

RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements_prod.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY renal_sight renal_sight
COPY models models

CMD uvicorn renal_sight.api.fast:app --host 0.0.0.0 --port $PORT
