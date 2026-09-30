# Single image used by both the API and dashboard services — see
# docker-compose.yml for how each service overrides the startup command.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, separately from copying the rest of the code.
# Docker caches this layer — as long as requirements.txt doesn't change,
# rebuilds skip reinstalling everything, making iteration much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY . .

# Default command runs the API. docker-compose.yml overrides this for
# the dashboard service to run Streamlit instead.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]