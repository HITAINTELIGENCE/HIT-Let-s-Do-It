# base image
FROM python:3.10.12-slim

# work folder
WORKDIR /app

# copy file in container
COPY requirements.txt .

#setup
RUN pip install --no-cache-dir -r requirements.txt

# copy project in container
COPY . .

# CLI run fastapi
CMD ["python", "main.py"]