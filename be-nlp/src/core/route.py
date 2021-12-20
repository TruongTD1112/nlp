from flask import Flask, request, jsonify
from reference import add_diacritic
from flask_cors import CORS, cross_origin
app = Flask(__name__)

CORS(app)


@app.route('/postData', methods=['POST'])
def hello_world():
    string = request.json['string']
    result = add_diacritic(string)
    return jsonify({"result" : result})


if __name__ == '__main__':
    app.run()
