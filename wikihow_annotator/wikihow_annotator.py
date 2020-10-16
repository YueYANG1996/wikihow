import pickle
import cv2
import random
import time
import copy
import sys
from PIL import Image

def reset():
    samples = pickle.load(open("1k_sample.p", "rb"))

    i = 1
    annotation = {}
    for sample in samples:
        annotation[sample[0]] = []
        if len(annotation) == 200:
            pickle.dump(annotation, open('annotation_{}.p'.format(i), 'wb'))
            annotation = {}
            i += 1
def get_correct_answer():
    samples = pickle.load(open("1k_sample.p", "rb"))

    ground_truth = {}
    for sample in samples:
        ground_truth[sample[0]] = sample[2]
    pickle.dump(ground_truth, open('ground_truth.p', 'wb'))

def calculate_acc(annotation_file):
    annotation = pickle.load(open(annotation_file, "rb"))
    ground_truth = pickle.load(open('ground_truth.p', "rb"))
    correct = 0
    for sample in annotation:
        if annotation[sample][0] == ground_truth[sample]:
            correct += 1
    print("Accuracy: ", correct / 200)
    return correct / 200

def select_answer(event, x, y, flags, param):
    global image, title, xi, yi
    img = copy.deepcopy(image)
    if event == cv2.EVENT_LBUTTONDOWN:
        xi, yi = x, y
        cv2.circle(img, (x, y), 5, (255, 0, 0), 5)
        cv2.imshow(title, img)


def main(annotation_file):
    annotation = pickle.load(open(annotation_file, "rb"))
    ground_truth = pickle.load(open('ground_truth.p', "rb"))
    non_annotated = []

    for sample in annotation:
        if len(annotation[sample]) == 0:
            non_annotated.append(sample)

    if len(non_annotated) == 0:
        print("This shard is all annotated")
        return None
    # try:
    while len(non_annotated) != 0:
        global image, title, xi, yi
        print()
        xi, yi = -1, -1
        title = random.choice(non_annotated)
        image = cv2.imread("images/" + title + '.png')
        # print(title)
        # image = Image.open("images/" + title + '.png')
        image = cv2.putText(image, "Samples left: {}".format(len(non_annotated)), (220, 390), cv2.FONT_HERSHEY_SIMPLEX,  
               0.5, (255, 0, 0) , 2, cv2.LINE_AA) 
        cv2.imshow(title, image)
        cv2.setMouseCallback(title, select_answer)
        key = cv2.waitKey(100000)
        if key == ord('q'):
            correct = False
            if xi < 0 or yi < 0:
                print("Invalid!")
            elif xi < 300 and yi < 200:
                print(0)
                annotation[title].append(0)
                if ground_truth[title] == 0:
                    correct = True
            elif xi > 300 and yi < 200:
                print(1)
                annotation[title].append(1)
                if ground_truth[title] == 1:
                    correct = True
            elif xi < 300 and yi > 200:
                print(2)
                annotation[title].append(2)
                if ground_truth[title] == 2:
                    correct = True
            elif xi > 300 and yi > 200:
                print(3)
                annotation[title].append(3)
                if ground_truth[title] == 3:
                    correct = True
            if correct:
                print("Correct") 
            else:
                print("Wrong")
            non_annotated.remove(title)
            cv2.destroyAllWindows()
            pickle.dump(annotation, open(annotation_file, "wb"))
        elif key == ord('w'):
            print("Skip")
            non_annotated.remove(title)
            cv2.destroyAllWindows()
            continue
        elif key == ord('e'):
            print("exit")
            cv2.destroyAllWindows()
            pickle.dump(annotation, open(annotation_file, "wb"))
            break
    # except:
    #     # save
        pickle.dump(annotation, open(annotation_file, "wb"))
    pickle.dump(annotation, open(annotation_file, "wb"))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        shard_id = sys.argv[1]
        print("Load shard {}".format(shard_id))
        main('annotation_{}.p'.format(shard_id))
    else:
        print("Choose shard one by default")
        main('annotation_1.p')


    ### compute the accuracy after finish one shard
    # calculate_acc('annotation_1.p')




