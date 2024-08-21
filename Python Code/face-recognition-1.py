import os
import imutils
import cv2
import logging
import json
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
from shutil import rmtree
import numpy as np
import torch
import boto3
import subprocess
import tempfile
os.environ['TORCH_HOME'] = '/tmp/'

def lambda_handler(event, context):
    output_bucket = event['output_bucket']
    file_name = event['file_name']

    s3_client = boto3.client('s3')
    download_path = '/tmp/' + file_name
    s3_client.download_file(output_bucket, file_name, download_path)

    output_text = face_recognition_function(download_path)
    base_filename = os.path.splitext(file_name)[0]

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(output_text)
        temp_file_name = temp_file.name

    s3_client.upload_file(temp_file_name, '1-output', base_filename + '.txt')

    os.remove(download_path)
    os.remove(temp_file_name)

    return {
        'statusCode': 200,
        'body': 'Processing completed'
    }



mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

def face_recognition_function(key_path):
    # Face extraction
    img = cv2.imread(key_path, cv2.IMREAD_COLOR)
    boxes, _ = mtcnn.detect(img)

    # Face recognition
    key = os.path.splitext(os.path.basename(key_path))[0].split(".")[0]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    face, prob = mtcnn(img, return_prob=True, save_path=None)
    saved_data = torch.load('data.pt')  # loading data.pt file
    if face != None:
        emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false
        embedding_list = saved_data[0]  # getting embedding data
        name_list = saved_data[1]  # getting list of names
        dist_list = []  # list of matched distances, minimum distance is used to identify the person
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)
        idx_min = dist_list.index(min(dist_list))

        # Save the result name in a file
        with open("/tmp/" + key + ".txt", 'w+') as f:
            f.write(name_list[idx_min])
        return name_list[idx_min]
    else:
        print(f"No face is detected")
    return
