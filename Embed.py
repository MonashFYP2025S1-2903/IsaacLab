import cv2
import torch
from torchvision import models, transforms
import numpy as np
import csv
import time
import pandas as pd


def embed_video(video_paths,it,frameidx):
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    caps = [cv2.VideoCapture(path) for path in video_paths]
    if not all(cap.isOpened() for cap in caps):
        print("Error: Could not open one or more video sources.")
        exit()


    all_data = []

    #For every camera, create a new list to store the embeddings
    for i in range(len(video_paths)):
        all_data.append([])
    


    try:
        while True:
            frames = [cap.read()[1] for cap in caps if cap.read()[0]]
            if not frames:
                break
            for i, frame in enumerate(frames):
                input_tensor = preprocess(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model(input_tensor).view(-1).cpu().numpy()
                timestamp = time.time()
                row_data = [timestamp, f"camera{i+1}"] + embedding.tolist()
                all_data[i].append(row_data)
                cv2.imshow(f"Camera {i+1}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()        
        
        # Create and save DataFrame for all cameras
        for i in range(0,len(all_data)):
            df = pd.DataFrame(all_data[i], columns=["timestamp", "camera"] + [f"f{i}" for i in range(2048)])
            df.to_pickle(f"video_embeddings.pkl")
        
        print("Saved DataFrame to image_embeddings.pkl")

def embed_images(image_paths:list,frameidx, it):
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_data = []

    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}.")
            continue
        input_tensor = preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(input_tensor).view(-1).cpu().numpy()
        timestamp = time.time()
        row_data = [timestamp, image_path] + embedding.tolist()
        all_data.append(row_data)

    df = pd.DataFrame(all_data, columns=["timestamp", "image"] + [f"f{i}" for i in range(2048)])
    df.to_pickle(f"Image_Embeddings_Out/image_embeddings_camera{it}-{frameidx}.pkl")
    print("Saved DataFrame to image_embeddings.pkl")


def verify_embedding(video_paths):
    try:
        for i in range(len(video_paths)):
            df = pd.read_pickle(f"image_embeddings_camera{i+1}.pkl")
            print("DataFrame loaded successfully.")
            print(df.head())
    except Exception as e:
        print(f"Error loading DataFrame: {e}")

if __name__ == "__main__":
    image_paths = ["rgb_out0001.png", "rgb_out0002.png", 
                   "rgb_out0003.png", "rgb_out0004.png"]
    embed_images(image_paths)
    df = pd.read_pickle("image_file_embeddings.pkl")
    print(df)


    



