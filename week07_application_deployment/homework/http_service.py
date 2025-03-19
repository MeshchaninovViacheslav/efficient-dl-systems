# http_service.py
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest
from fastapi.responses import PlainTextResponse
from model import model  # from model.py

app = FastAPI()

# Prometheus counter
inference_counter = Counter("app_http_inference_count", "Number of HTTP inference requests")

# Request model
class PredictRequest(BaseModel):
    url: str

@app.post("/predict")
async def predict(req: PredictRequest):
    print(f"Received request: {req.url}", flush=True)
    inference_counter.inc()  # Increment the counter
    objects = model.predict(req.url)
    return {"objects": objects}

@app.get("/metrics")
async def metrics():
    # Return the metrics in plain text format
    return PlainTextResponse(generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()

"""
curl http://localhost:8080/docs

curl -XPOST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"url": "http://images.cocodataset.org/val2017/000000001268.jpg"}'

curl http://localhost:8080/metrics
"""