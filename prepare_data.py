import os
import numpy as np
import cv2
from glob import glob
import json
from sklearn.model_selection import train_test_split

def get_data (dataset_path, train = True):
    # Determine if we are processing traning or testin data
    data_type = "train" if train else "test"

    # Load questions and answers from the JSON file
    with open(os.path.join(dataset_path, data_type, "questions.json"), "r") as file:
        data = json.load(file)

    questions, answers, imgae_paths = [], [], []
    # Parse the questions, answers, and corresponding image paths
    for q, a, p in data:
        questions.append(q)
        answers.append(a)
        imgae_paths.append(os.path.join(dataset_path, data_type, "images", f"{p}.png"))
    
    return questions, answers, imgae_paths

def get_answer_labels(dataset_path):
    with open (os.path.join(dataset_path, "answers.txt"), "r") as file:
        data = file.read().strip().split("\n")
        return data

def main():
    dataset_path = "data"

    # Load training and testing data
    # Q - questions, A - answers, I - images
    trainQ, trainA, trainI = get_data(dataset_path, train=True)
    testQ, testA, testI = get_data(dataset_path, train=False)

    # Split the training data into training and validation sets
    trainQ, valQ, trainA, valA, trainI, valI = train_test_split (trainQ, trainA, trainI, test_size=0.2, random_state=42)

    # Print statistic
    print(f"Train -> Questions: {len(trainQ)} - Answers: {len(trainA)} - Images: {len(trainI)}")
    print(f"Train -> Questions: {len(valQ)} - Answers: {len(valA)} - Images: {len(valI)}")
    print(f"Train -> Questions: {len(testQ)} - Answers: {len(testA)} - Images: {len(testI)}")

if __name__ == "__main__":
    main()
