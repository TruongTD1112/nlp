from flask import Flask, request, jsonify
from reference import add_diacritic
from flask_cors import CORS, cross_origin
app = Flask(__name__)

CORS(app)


@app.route('/postData', methods=['POST'])
def hello_world():
    string = request.json['string']
    restore_tone = request.json['restore_tone']
    keep_special_character = request.json['keep_special_character']

    result = add_diacritic(string, restore_tone, keep_special_character)
    return jsonify({"result" : result})


if __name__ == '__main__':
    app.run()
