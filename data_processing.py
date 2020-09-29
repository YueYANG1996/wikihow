import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import cv2
from sentence_transformers import SentenceTransformer

def image_encoder():
    image_path = ""
    img_names = os.listdir(image_path)
    real_img_ids = []
    for name in real_img_names:
        real_img_ids.append(name.split('.')[0])

    model = InceptionV3(weights='imagenet')
    new_input = model.input
    hidden_layer = model.layers[-2].output
    model_new = Model(new_input, hidden_layer)

    encoded_images = {}
    failed = []
    for image_name in img_names:
      try:
        img = image.load_img(image_path + image_name, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        image_id = image_name.split('.')[0]
        encoded_images[image_id] = model_new.predict(x)
      except:
        print('fail')
        failed.append(image_name)
    print(len(encoded_images))
    pickle.dump(encoded_images, open(image_path + "/encoded_image/encoded_images_new.p", "wb" ))

def sentence_encoder():
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    titles = []
    for sample in sample_data:
      titles.append(sample[0])
    all_titles = set(titles)
    encoded_titles = {}
    for title in tqdm(all_titles):
      encoded_titles[title] = model.encode(title)
    print(len(encoded_titles))
    pickle.dump(encoded_titles, open(project_path + "/encoded_title/new_encoded_titles.p", "wb" ))


