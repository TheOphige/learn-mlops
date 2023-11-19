from flask import Response, Flask
import prometheus_client
from prometheus_client import Counter, Histogram
import time
import random

app = Flask(__name__)



REQUESTS = Counter(
    'requests', 'Application Request Count',
    ['endpoint']
)

@app.route('/')
def index():
    REQUESTS.labels(endpoint='/').inc()
    return '<h1>Development Prometheus-backed Flask App</h1>'

@app.route('/metrics/')
def metrics():
    return Response(
        prometheus_client.generate_latest(),
        mimetype='text/plain; version=0.0.4; charset=utf-8'
    )


TIMER = Histogram(
    'slow', 'Slow Requests',
    ['endpoint']
)

@app.route('/database/')
def database():
    with TIMER.labels('/database').time():
        # simulated database response time
        time.sleep(random.uniform(1, 3))
    return '<h1>Completed expensive database operation</h1>'


if __name__ == '__main__':
    app.run(host='0.0.0.0')