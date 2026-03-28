FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
COPY src/ src/
COPY model/model.pkl model/model.pkl
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --upgrade --force-reinstall -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
