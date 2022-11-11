from flask import Flask, json, request
from main import optimize
import traceback

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def hello_world():
    data = request.get_json()
    x = optimize(data)

if __name__ == '__main__':
    app.run(debug=True)