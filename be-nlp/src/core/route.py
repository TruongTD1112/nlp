from flask import Flask, request, jsonify
from reference import predict
from validate import preprocessing
from flask_cors import CORS, cross_origin
app = Flask(__name__)

CORS(app)


@app.route('/postData', methods=['POST'])
def hello_world():
    string = request.json['string']
    restore_tone = request.json['restore_tone']
    keep_special_character = request.json['keep_special_character']
    totalWord = len(string.split(" "))
    preProcess = preprocessing(string)
    stringPreProcess = preProcess[0]
    stringPredict = predict(stringPreProcess, restore_tone, keep_special_character)
    result = getResult(stringPredict, preProcess[1], preProcess[2], totalWord)
    return jsonify({"result" : result})


def getResult(stringPredict, listStringAfterRemove, listStringRemove, length):
    print(listStringRemove)
    resultArray = [None] * length
    print(resultArray)
    stringPredictSplit = stringPredict.split(" ")
    listStringAfterRemovePredict = []
    for i in range(len(listStringAfterRemove)):
        listStringAfterRemovePredict.append((listStringAfterRemove[i][1], stringPredictSplit[i]))
    # print(listStringAfterRemovePredict)
    for i in range(len(listStringAfterRemovePredict)):
        resultArray[listStringAfterRemovePredict[i][0]] =  listStringAfterRemovePredict[i][1]
    for i in range(len(listStringRemove)):
        resultArray[listStringRemove[i][1]] =  listStringRemove[i][0]

    result = ""
    for i in range(length):
        result += resultArray[i] + " "
    return result


if __name__ == '__main__':
    app.run()

