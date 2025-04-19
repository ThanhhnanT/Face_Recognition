import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017/')

database = client.get_database('face_recog')
StudentModel = database['student_id']

