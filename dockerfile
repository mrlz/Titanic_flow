FROM python:3.12.3

WORKDIR /app

COPY app.py .
COPY data_preprocessing.py .
COPY models/ ./models/

ENV LOG_LEVEL=DEBUG

COPY requirements_container.txt .
RUN pip install --no-cache-dir -r requirements_container.txt


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
