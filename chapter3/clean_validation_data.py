import urllib.request
import zipfile

validation_file_name = "validation-horse-or-human.zip"
validation_dir = 'horse-or-human/validation/'
# urllib.request.urlretrieve(validation_url, validation_file_name)

zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()