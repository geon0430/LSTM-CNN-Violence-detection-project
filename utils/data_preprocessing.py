import os
import cv2
import random
import numpy as np
import imgaug.augmenters as iaa

NonViolnceVideos_Dir = "../datasets/NonViolence/"
ViolnceVideos_Dir = "../datasets/Violence/"

NonViolence_files_names_list = os.listdir(NonViolnceVideos_Dir)
Violence_files_names_list = os.listdir(ViolnceVideos_Dir)

Random_NonViolence_Video = random.choice(NonViolence_files_names_list)
Random_Violence_Video = random.choice(Violence_files_names_list)

IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224
 
SEQUENCE_LENGTH = 15

DATASET_DIR = "../datasets/"
 
CLASSES_LIST = ["NonViolence","Violence"]


def frames_extraction(video_path):
 
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
 
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, image = video_reader.read()
        zoom = iaa.Affine(scale=1.3)
        random_brightness = iaa.Multiply((1, 1.3))
        image_aug = random_brightness(image = image)
        image_aug = zoom(image = image_aug)
        
        if not success:
            break

        resized_frame = cv2.resize(image_aug, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    video_reader.release()
 
    return frames_list

def create_dataset():
 
    features = []
    labels = []
    video_files_paths = []
    
    for class_index, class_name in enumerate(CLASSES_LIST):
        
        print(f'Extracting Data of Class: {class_name}')   
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        for file_name in files_list[:90]: 
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
 
            frames = frames_extraction(video_file_path)
 
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)
 
    features = np.asarray(features)
    labels = np.array(labels)  

    return features, labels, video_files_paths

features, labels, video_files_paths = create_dataset()

np.save("../features.npy",features)
np.save("../labels.npy",labels)
np.save("...video_files_paths.npy",video_files_paths)
