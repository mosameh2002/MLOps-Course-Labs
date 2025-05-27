from fastapi import FastAPI, Request, Response
from prometheus_fastapi_instrumentator import Instrumentator
import time
from prometheus_client import Counter, Gauge, Histogram

app = FastAPI(title="Monitored FastAPI App")

# Create metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total count of HTTP requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint", "status_code"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests in progress",
    ["method", "endpoint"]
)

API_CALLS = Counter(
    "api_calls_total",
    "Total count of calls by endpoint",
    ["endpoint"]
)

# Add middleware to track metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    
    # Skip metrics endpoint
    if endpoint == "/metrics":
        return await call_next(request)
    
    # Track in-progress requests
    REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
    
    # Track API calls
    API_CALLS.labels(endpoint=endpoint).inc()
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
        
        # Track request count and latency
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint, status_code=status_code).observe(
            time.time() - start_time
        )
        
        return response
    except Exception as e:
        # Handle exceptions (return 500 for unhandled exceptions)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=500).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint, status_code=500).observe(
            time.time() - start_time
        )
        raise e
    finally:
        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()

# Initialize the instrumentator (just for the /metrics endpoint)
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/slow")
async def slow_endpoint():
    time.sleep(2)  # Simulate slow processing
    return {"message": "This was a slow request"}

@app.get("/fast")
async def fast_endpoint():
    return {"message": "This was a fast request"}

@app.get("/error")
async def error_endpoint():
    # Simulate an error
    if True:
        raise ValueError("This is a simulated error")
    return {"message": "You won't see this"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
