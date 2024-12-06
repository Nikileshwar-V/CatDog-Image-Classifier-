import os
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # For saving the trained model


# Function to load and flatten images for model training
import os
import numpy as np
from tensorflow.keras.preprocessing import image


def load_and_flatten_images(directory, img_size=(64, 64)):
    X = []  # List to hold image data
    y = []  # List to hold labels (cat = 0, dog = 1)

    # Traverse the directory and subdirectories
    for label, class_name in enumerate(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)  # cat or dog folder path
        if os.path.isdir(class_dir):  # Check if it's a folder
            for img_name in os.listdir(class_dir):  # Iterate over images
                img_path = os.path.join(class_dir, img_name)
                if img_path.endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image files
                    # Load the image and resize it
                    img = image.load_img(img_path, target_size=img_size)
                    img_array = image.img_to_array(img)  # Convert image to array
                    img_array = img_array.flatten()  # Flatten the image
                    X.append(img_array)  # Add to X
                    y.append(label)  # Add label (0 for cat, 1 for dog based on folder)

    X = np.array(X)  # Convert X to numpy array
    y = np.array(y)  # Convert y to numpy array
    return X, y  # Return the data and labels

# Load and flatten images from the 'data/images' folder
X, y = load_and_flatten_images('data/images/')

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'image_classifier_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model training completed. Accuracy: {accuracy}")
