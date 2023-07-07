import csv
import os
import cv2
import mediapipe as mp
from app import calc_landmark_list, pre_process_landmark


def main():
    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Directory containing images
    train_directory = 'ASL_Dataset/Train/A'
    csv_path = 'model/keypoint_classifier/keypoint.csv'

    # Loop through each sub-directory in the Train directory
    index = 0
    for letter_dir in os.listdir(train_directory):
        # letter_dir_path = os.path.join(train_directory, letter_dir)
        print(letter_dir)


if __name__ == "__main__":
    main()
