FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app/miniwob
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["pytest", "-vs"]
