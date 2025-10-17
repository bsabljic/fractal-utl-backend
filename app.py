from flask_cors import CORS
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/api/public-key')
def get_public_key():
    try:
        with open('public.pem', 'r') as f:
            key = f.read()
        return key, 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        return str(e), 500

