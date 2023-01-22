import urllib.request
import zipfile

# url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
# url = "https://www.kaggle.com/datasets/sanikamal/horses-or-humans-dataset/download?datasetVersionNumber=1"

file_name = "horse-or-human.zip"
training_dir = 'horse-or-human/training/'
# urllib.request.urlretrieve(url, file_name)

zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()