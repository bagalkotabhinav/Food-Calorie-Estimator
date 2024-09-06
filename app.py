from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import csv
from PIL import Image

app = Flask(__name__)


model = load_model('indian_food_model.hdf5')
food_list=['Burger', 'Butter Naan', 'Chai/Coffee', 'Chapati', 'Chole Bhature', 'Dal Makhani', 'Dhokla', 'Fried Rice', 'Idly', 'Jalebi', 'Kaathi Rolls', 'Kadai Paneer', 'Kulfi', 'Masala Dosa', 'Momos', 'Paani Puri', 'Pakode', 'Pav Bhaji', 'Pizza', 'Samosa']
target_img = os.path.join(os.getcwd() , 'static/images')


@app.route('/')
def index_view():
    return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png','webp'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):

    img = load_img(filename, target_size=(224, 224))
    x  = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /=255
    #x = preprocess_input(x)
    return x

def read_nutrient_data(csv_file):
    nutrient_data = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            food_class = row['Name']
            nutrients = {
                'Name': row['Name'],
                'One Serving': int(row['One Serving']),
                'calories': float(row['Calories']),
                'protein': float(row['Proteins']),
                'carbs': float(row['Carbs']),
                'fats': float(row['Fats'])
            }
            nutrient_data[food_class] = nutrients
    return nutrient_data
nutrient_data = read_nutrient_data('calories.csv')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)


            class_prediction=model.predict(img) 
            classes_x=np.argmax(class_prediction)

            decoded_predictions = [(class_id, food_list[class_id], score) for class_id, score in enumerate(class_prediction[0])]
            decoded_predictions.sort(key=lambda x: x[2], reverse=True)
            predicted_class = decoded_predictions[0][1]
            confidence = decoded_predictions[0][2]
            li = confidence * 100
            likelihood=f"{li:.2f}%"

            fruit=food_list[classes_x]

            nutrient_values = nutrient_data.get(predicted_class)
            if nutrient_values:
                ps=nutrient_values['One Serving']
                kcal = nutrient_values['calories']
                prot = nutrient_values['protein']
                carb = nutrient_values['carbs']
                fats = nutrient_values['fats']

            if(li<90):
                return render_template('predict.html', fruit = "Not Present", user_image = file_path)
            else:
                return render_template('predict.html', fruit = predicted_class,ps=ps,kcal=kcal,prot=prot,carb=carb,fats=fats,user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension" 


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)


