import json
from watson_developer_cloud import VisualRecognitionV3

visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    iam_apikey='tDYM20kS5sD2jkAApAOrb83RdBO0I2nl-rCuVp10gsJH')

with open('./image.jpg', 'rb') as images_file:
    classes = visual_recognition.classify(
        images_file,
        threshold='0.6',
	classifier_ids='DefaultCustomModel_877446449').get_result()
a=classes['images'][0]['classifiers'][0]['classes'][0]['class']
print(a)

if  a=='persons' :
    print("authorized person entered")
else:
    print("unauthorized person not entered")



