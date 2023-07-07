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
    train_directory = 'ASL_Dataset/Train'
    csv_path = 'model/keypoint_classifier/keypoint.csv'

    index = 0
    for letter_dir in sorted(os.listdir(train_directory)):
        letter_dir_path = os.path.join(train_directory, letter_dir)
        if os.path.isdir(letter_dir_path):
            if index < 22:
                index += 1
                continue
            # Loop through each file in the subdirectory
            print(letter_dir_path)
            print(index)
            for filename in os.listdir(letter_dir_path):
                image = cv2.imread(os.path.join(letter_dir_path, filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract the landmark coordinates using the imported function
                        landmark_list = calc_landmark_list(image, hand_landmarks)
                        normalized_landmarks = pre_process_landmark(landmark_list)
                        # Output the landmark coordinates
                        # Store the normalized landmarks in the keypoint.csv file
                        with open(csv_path, 'a', newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([index, *normalized_landmarks])
            index += 1


# Run the main function
if __name__ == "__main__":
    main()
