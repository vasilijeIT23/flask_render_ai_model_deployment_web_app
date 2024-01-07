from flask import Flask, render_template, request
import cv2
import pickle

app = Flask(__name__)

pickle_file_path = 'models/cv.pkl'
with open(pickle_file_path, 'rb') as file:
    cv = pickle.load(file)

pickle_file_path = 'models/clf.pkl'
with open(pickle_file_path, 'rb') as file:
    clf = pickle.load(file)

def extract_faces_from_images(img_path):
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if img_path.endswith(('.jpg', '.jpeg', '.png')):
        # Read an image
        img = cv2.imread(img_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Save each detected face as a new image in the output folder
        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
        
        face = cv2.resize(img, (200, 200))
        face = face / 255.0
        face = img.reshape((1, 200, 200, 1))

        return face

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('content')
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)