
import urllib.request
import zipfile

training_name = "horse-or-human.zip"
training_url="https://storage.googleapis.com/learning-datasets/" + training_name

urllib.request.urlretrieve(training_url, training_name)

training_dir = 'horse-or-human/training/'
zip_ref = zipfile.ZipFile(training_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()


validation_name = "validation-horse-or-human.zip"
validation_url="https://storage.googleapis.com/learning-datasets/" + validation_name

urllib.request.urlretrieve(validation_url, validation_name)

validation_dir = 'horse-or-human/validation/'

zip_ref = zipfile.ZipFile(validation_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()

