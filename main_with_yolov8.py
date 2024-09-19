import cv2
import face_recognition
import pickle

# Load known face encodings and names from file
file_path = 'known_faces.pkl'  # Path to your saved known faces
with open(file_path, 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Initialize video input (0 for webcam, or provide a path to a video file)
video_capture = cv2.VideoCapture('face_recog/vid.mkv')  # Use 0 for the default webcam, or replace with a video file path

# Tolerance for face comparison, default is 0.6 (lower = more strict, higher = more lenient)
tolerance = 0.5  # Adjust tolerance if needed

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    if not ret:
        break  # Break the loop if there's an issue with video capture

    # Resize frame for faster face recognition processing (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = small_frame[:, :, ::-1]

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the current frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
        name = "mencurigakan"  # Default label for unknown faces

        # Calculate face distances for debugging
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        print(f"Face distances: {face_distances}")  # Debugging info to check match distances

        # If there's a match
        if True in matches:
            best_match_index = face_distances.argmin()  # Find the closest match
            name = known_face_names[best_match_index]

        # Set the bounding box color: red for "mencurigakan", green for known person
        box_color = (0, 0, 255) if name == "mencurigakan" else (0, 255, 0)  # Red for unknown, Green for known

        # Scale back up face locations since the frame we detected on was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face and label the name
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    # Display the result frame
    cv2.imshow('Video', frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
