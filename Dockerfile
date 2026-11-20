FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home aifit && mkdir -p /data && chown aifit:aifit /data
USER aifit
ENV DATA_DIR=/data
EXPOSE 8080
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8080", "--threads=8", "--call", "app:create_app"]
