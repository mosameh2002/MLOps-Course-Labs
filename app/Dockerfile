# Use official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r req.txt

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "appp:app", "--host", "0.0.0.0", "--port", "8000"]
