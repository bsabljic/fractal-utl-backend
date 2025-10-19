from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fractal-UTL API")

# CORS (dozvoli lokalni frontend i sve dok ne zaključamo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # po potrebi suzi na http://localhost:5173
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/ping")
def ping():
    return {"status": "ok"}

# demo upload endpoint – vraća "public_key" da frontend ima što prikazati
@app.post("/api/compare")
async def compare(file: UploadFile = File(...)):
    # TODO: ovdje će ići tvoja realna logika
    return {"public_key": "UTL-DEMO-PUBKEY-123"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
