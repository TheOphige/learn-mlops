from flask  import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World! I can and I will"

if __name__ == '__main__':
    app.run(debug=True) 



# ping pong
# from flask import Flask

# app = Flask('ping')

# @app.route('/ping', methods=['GET'])
# def ping():
#     return "PONG"

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=9696)