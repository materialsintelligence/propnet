FROM python:3.6.2
ADD . /code
WORKDIR /code
RUN pip install --no-cache-dir -r requirements.txt
CMD ["gunicorn", "app:server", "--timeout", "300", "--workers", "16", "--bind", "8500"]
