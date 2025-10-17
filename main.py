from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/api/public-key')
def get_public_key():
    try:
        with open('public.pem', 'r') as f:
            return f.read()
    except Exception as e:
        return {'error': str(e)}

