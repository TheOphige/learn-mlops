from flask  import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World! I can and I will"

if __name__ == '__main__':
    app.run(debug=True) 