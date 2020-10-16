import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import cv2
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def get_images_titles():
    train_sample = pickle.load(open("splits/train_wiki_full.pickle", "rb"))
    test_sample = pickle.load(open("splits/test_wiki_full.pickle", "rb"))
    image_ids = []
    titles = []
    all_sample = train_sample + test_sample
    print("n of all samples: ", len(all_sample))
    for sample in all_sample:
        image_ids.append(sample[1])
        titles.append(sample[0])
    image_ids = set(image_ids)
    titles = set(titles)
    pickle.dump(image_ids, open("data/image_ids.p", "wb"))
    pickle.dump(titles, open("data/titles.p", "wb"))
    print("n of all images: ", len(image_ids))
    print("n of all titles: ", len(titles))
    """
    n of all samples:  2558720
    n of all images:  623678
    n of all titles:  42550
    """

def image_encoder():
    image_path = '/../nlp/data/lyuqing-zharry/wikihow_probing/data/wikihow_data/multimodal/data/train'
    img_names = os.listdir(image_path)
    img_ids = []
    for name in img_names:
        img_ids.append(name.split('.')[0])

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

def sentence_encoder(all_titles):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    encoded_titles = {}
    for title in tqdm(all_titles):
        encoded_titles[title] = model.encode(title)
    pickle.dump(encoded_titles, open("encoded/encoded_titles.p", "wb" ))
    # pickle.dump(encoded_titles, open("/../nlp/data/lyuqing-zharry/wikihow_probing/data/wikihow_data/multimodal/encoded/encoded_titles.p", "wb" ))

if __name__ == '__main__':
    all_titles = pickle.load(open("data/titles.p", "rb"))
    print(len(all_titles))
    sentence_encoder(all_titles)


    
