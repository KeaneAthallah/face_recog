import face_recognition
import os
import pickle

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for person_dir in os.listdir(known_faces_dir):
        person_path = os.path.join(known_faces_dir, person_dir)
        if os.path.isdir(person_path):
            # Load each image of the person
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                image = face_recognition.load_image_file(image_path)
                try:
                    # Encode the face
                    face_encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_dir)  # Save the person's name
                except IndexError:
                    print(f"No face found in {image_file}, skipping.")

    # Save to a file using pickle
    with open('known_faces.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    return known_face_encodings, known_face_names

# Path to the directory with known faces
known_faces_dir = '/home/kenn/known_faces'
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
