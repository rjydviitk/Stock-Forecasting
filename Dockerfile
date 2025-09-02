# Use slim Python base to keep image small
FROM python:3.11-slim

# System deps for Prophet/plotly-kaleido builds
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     libatlas-base-dev     libfreetype6-dev     libpng-dev     libffi-dev     git     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

EXPOSE 8501

# Streamlit config to play nice in containers
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "dashboard/app.py"]
