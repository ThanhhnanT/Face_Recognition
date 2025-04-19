import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017/')

print(client.list_database_names())

database = client.get_database('face_recog')
StudentModel = database['student_id']
student = StudentModel.find_one({
    'fullName': "Trump"
})

print(student)