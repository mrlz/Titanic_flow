FROM python:3.12.3

WORKDIR /app

COPY app.py .
COPY data_preprocessing.py .
COPY models/ ./models/

ENV LOG_LEVEL=DEBUG

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
