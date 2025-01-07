import face_recognition
import cv2
import os
import csv
from datetime import datetime

# Paths
IMAGE_FOLDER = "C:/Users/ironm/FaceRecognitionProject/student_images"
ATTENDANCE_FILE = "C:/Users/ironm/FaceRecognitionProject/attendance.csv"

# Load student images and names
def load_students(image_folder):
    student_encodings = {}
    for file_name in os.listdir(image_folder):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(image_folder, file_name)
            student_image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(student_image)[0]
            student_name = os.path.splitext(file_name)[0]  # Name from file name
            student_encodings[student_name] = encoding
    return student_encodings

# Mark attendance
def mark_attendance(name):
    with open(ATTENDANCE_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        writer.writerow([name, date, time])
        print(f"Marked attendance for {name} at {time} on {date}")

# Main function
def recognize_faces():
    student_encodings = load_students(IMAGE_FOLDER)
    known_faces = list(student_encodings.values())
    known_names = list(student_encodings.keys())

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    print("Press 'Q' to quit and save attendance.")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to access webcam.")
            break

        # Convert the frame to RGB
        rgb_frame = frame[:, :, ::-1]

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            # Mark attendance
            mark_attendance(name)

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Show the video
        cv2.imshow("Face Recognition Attendance", frame)

        # Quit the program on 'Q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

# Run the face recognition attendance system
if __name__ == "__main__":
    # Create the attendance file if it doesn't exist
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date", "Time"])  # Header row
    recognize_faces()
