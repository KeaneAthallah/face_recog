import cv2
import face_recognition
import pickle

# Load known face encodings and names from file
file_path = 'known_faces.pkl'  # Path to your saved known faces
with open(file_path, 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Path to the new image you want to check
image_path = 'dika.jpg'  # Update this with the actual image path

# Load the new image
new_image = face_recognition.load_image_file(image_path)

# Find face locations and encodings in the new image
face_locations = face_recognition.face_locations(new_image)
face_encodings = face_recognition.face_encodings(new_image, face_locations)

# Convert the image to displayable format using OpenCV
image_to_show = cv2.imread(image_path)

# Tolerance for face comparison, default is 0.6 (lower = more strict, higher = more lenient)
tolerance = 0.5  # Adjust tolerance if needed

# Loop through each face found in the new image
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

    # Draw a box around the face and label the name
    cv2.rectangle(image_to_show, (left, top), (right, bottom), box_color, 2)
    cv2.putText(image_to_show, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

# Display the result
cv2.imshow('Image', image_to_show)
cv2.waitKey(0)  # Press any key to close the window
cv2.destroyAllWindows()

if name != "mencurigakan":
    print(f"Person detected: {name}")
else:
    print("No known person detected.")
