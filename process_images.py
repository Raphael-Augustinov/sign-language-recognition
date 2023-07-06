import os
import cv2
import mediapipe as mp


def calc_landmark_list(image, hand_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(hand_landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def main():
    # Initialize MediaPipe Hand
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Directory containing images
    image_directory = 'ASL_Dataset/Train/A'

    # Loop through each image in the directory
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg"):
            # Read image
            image = cv2.imread(os.path.join(image_directory, filename))

            # Convert the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image to extract landmarks
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract the landmark coordinates
                    landmark_list = calc_landmark_list(image, hand_landmarks)

                    # Output the landmark coordinates
                    print(f"Landmarks for {filename}: {landmark_list}")


# Run the main function
if __name__ == "__main__":
    main()
