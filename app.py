from flask import Flask, request, jsonify
from pymongo import MongoClient
from flask_cors import CORS
import hashlib
from bson import ObjectId
import google.generativeai as genai
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, messaging


app = Flask(__name__)
CORS(app)
cred = credentials.Certificate('C:/Users/Savio Sunny/Downloads/project-1-bca0f-firebase-adminsdk-fxjcp-de0d57f448.json')
firebase_admin.initialize_app(cred)


# MongoDB client setup
client = MongoClient('mongodb+srv://saviosunny48:2TJsNwpNwqJX2aG3@cluster0.0zmwv1l.mongodb.net/')
db = client['test']
users_collection = db['App_users']
prescriptions_collection = db['prescriptions']
appointments_collection = db['appointments']
bills_collection = db['Bills']
medicines_collection = db['Medicine']
medical_records_collection = db['medical_records']
symptom_collection = db['symptom']



GOOGLE_API_KEY = 'AIzaSyDLdxqrm1DMDnEdnX9oljbtewsRe90x2QU'
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

rf_model = None
one_hot_encoder = None

# Hash password function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Validate email function
def validate_email(Email):
    import re
    Email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(Email_regex, Email) is not None

# Train the model
def train_model():
    global rf_model, one_hot_encoder
    
    df = pd.DataFrame(list(symptom_collection.find()))
    print(df)
    
    if df.empty:
        print("Symptom collection is empty. Cannot train the model.")
        return
    
    one_hot_encoder = OneHotEncoder()
    X = one_hot_encoder.fit_transform(df[['symptom']]).toarray()
    y = df['average_time']
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    print("Model trained.")

@app.route('/send-notification', methods=['POST'])
def send_notification():
    try:
        print('Received POST request to /send-notification')

        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data received'})

        # Extract token and message from JSON data
        token = data.get('token')
        message = data.get('message')

        if not token or not message:
            return jsonify({'success': False, 'error': 'Token or message not provided'})

        print('Received token:', token)
        print('Received message:', message)

        # Construct a message to send
        notification = messaging.Notification(title='CareWise', body=message)
        data = {'message': message}  # Add data payload (optional)
        fcm_message = messaging.Message(notification=notification, token=token)
        print(fcm_message);
        
        # Send message
        response = messaging.send(fcm_message)
        print('Successfully sent message:', response)

        return jsonify({'success': True, 'message': 'Notification sent successfully'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

    
# Predict consultation time
def predict_consultation_time(descriptions):
    if rf_model is None or one_hot_encoder is None:
        raise ValueError("Model or encoder is not trained.")
    
    input_features = np.zeros(len(one_hot_encoder.categories_[0]))
    for description in descriptions:
        if description in one_hot_encoder.categories_[0]:
            index = np.where(one_hot_encoder.categories_[0] == description)[0][0]
            input_features[index] = 1

    predicted_time = rf_model.predict([input_features])[0]
    return predicted_time

@app.route('/check_appointments', methods=['POST'])
def check_appointments():
    data = request.get_json()
    Patient_ID = data['Patient_ID']

    appointment = appointments_collection.find_one({'Patient_ID': Patient_ID})

    if appointment:
        Name = appointment.get('Name', None)
        Appointment_ID = appointment.get('Appointment_ID', None)
        Patient_ID = appointment.get('Patient_ID', None)
        Token = appointment.get('Token', None)

    if Name and Appointment_ID and Patient_ID and Token:
        # Get the current smallest token value
        min_token = appointments_collection.find_one(sort=[("Token", 1)])["Token"]

        if Token is not None:
            earlier_appointments = appointments_collection.find({'Token': {'$lt': Token}})
            descriptions = [a['description'] for a in earlier_appointments]

            if not descriptions:
                # If no descriptions found, set arrival time to 9:00 AM
                arrival_time = datetime.now().replace(hour=9, minute=0)
            else:
                waiting_time = predict_consultation_time(descriptions)
                current_time = datetime.now()

                # If current time is after 4:00 PM, set arrival time to 9:00 AM of the next day
                if current_time.hour >= 16:
                    arrival_time = datetime.now().replace(hour=9, minute=0) + timedelta(days=1)
                else:
                    arrival_time = current_time

                # Add waiting time to arrival time
                arrival_time += timedelta(minutes=round(waiting_time))

            # Format arrival time to display only hours and minutes
            arrival_time_str = arrival_time.strftime('%I:%M %p')

            return jsonify({
                'status': 'success',
                'message': 'Appointment Found',
                'Name': Name,
                'Appointment_ID': Appointment_ID,
                'Patient_ID': Patient_ID,
                'Token': Token,
                'min_token': min_token,
                'arrival_time': arrival_time_str
            })
        else:
            return jsonify({'status': 'error', 'message': 'Token not found in the appointment'}), 512
    else:
        return jsonify({'status': 'error', 'message': 'Appointment not found for the patient'}), 404



# Register new user
@app.route('/register', methods=['POST'])
def register():
    data = request.json

    # Extract and validate input data
    Name = data.get('Full_name')
    Email = data.get('Email')
    Password = data.get('Password')
    Age = data.get('Age')
    Phone = data.get('Phone')
    Patient_ID = data.get('Patient_ID')

    if not all([Name, Email, Password, Age, Phone, Patient_ID]):
        return jsonify({'message': 'All fields are required'}), 400

    if not validate_email(Email):
        return jsonify({'message': 'Invalid email format'}), 417

    if len(Password) < 5:
        return jsonify({'message': 'Password must be at least 8 characters long'}), 420

    if users_collection.find_one({'Email': Email}):
        return jsonify({'message': 'Email already exists'}), 409

    if users_collection.find_one({'Patient_ID': Patient_ID}):
        return jsonify({'message': 'Patient ID already exists'}), 415

    try:
        Hashed_password = hash_password(Password)
        user_data = {
            'Name': Name,
            'Email': Email,
            'Password': Hashed_password,
            'Age': Age,
            'Phone': Phone,
            'Patient_ID': Patient_ID,
        }
        users_collection.insert_one(user_data)
        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        print(f"Error inserting user: {e}")
        return jsonify({'message': 'Internal server error'}), 500

# Get medical records
@app.route('/get_medical_records', methods=['POST'])
def get_medical_records():
    data = request.json
    Patient_ID = data.get('Patient_ID')
    
    if not Patient_ID:
        return jsonify({'status': 'error', 'message': 'Patient_ID is required'}), 400

    medical_records = medical_records_collection.find({'Patient_ID': Patient_ID})
    medical_records_list = list(medical_records)

    if medical_records_list:
        for record in medical_records_list:
            record['_id'] = str(record['_id'])
        return jsonify({'status': 'success', 'medical_records': medical_records_list}), 200
    else:
        return jsonify({'status': 'error', 'message': 'No medical records found for the patient'}), 404

# Login existing user
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    Patient_ID = data.get('Patient_ID')
    Password = data.get('Password')

    user = users_collection.find_one({'Patient_ID': Patient_ID})

    if user and user['Password'] == hash_password(Password):
        train_model()
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'message': 'Invalid Patient_ID or password'}), 401

# Get doctors by specialization
@app.route('/doctors/<Specialization>', methods=['GET'])
def get_doctors_by_Specialization(Specialization):
    doctors_collection = db['doctors']
    doctors = doctors_collection.find({'Specialization': Specialization})
    doctors_list = [{**doctor, '_id': str(doctor['_id'])} for doctor in doctors]
    return jsonify(doctors_list), 200

# Get minimum token number
@app.route('/min_token', methods=['GET'])
def get_min_token():
    tokens = appointments_collection.distinct('Token')
    min_token = min(set(range(1, max(tokens) + 2)) - set(tokens)) if tokens else 1
    return jsonify({"min_token": min_token})

# Book appointment
@app.route('/appointments', methods=['POST'])
def book_appointment():
    data = request.get_json()

    Name = data.get('Name')
    Age = data.get('Age')
    description = data.get('description')
    startDate = data.get('startDate')
    date = datetime.now().isoformat()
    Appointment_ID = random.randint(1000, 9999)
    Patient_ID = data.get('Patient_ID')
    Doctor_ID = data.get('Doctor_ID')  # Accept Doctor_ID from the request

    response = get_min_token()
    Token = response.get_json().get("min_token")

    if not all([Name, Age, description, startDate, Patient_ID, Doctor_ID]):
        return jsonify({"error": "Missing required fields"}), 400

    appointment = {
        "Name": Name,
        "Age": Age,
        "description": description,
        "StartDate": startDate,
        "date": date,
        "Appointment_ID": Appointment_ID,
        "Patient_ID": Patient_ID,
        "Doctor_ID": Doctor_ID,  # Include Doctor_ID in the appointment data
        "Token": Token
    }

    appointments_collection.insert_one(appointment)
    return jsonify({"message": "Appointment booked successfully"}), 200

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    symptoms = symptom_collection.distinct('symptom')
    return jsonify({'symptoms': symptoms}), 200

# Generate bill
@app.route('/generate_bill', methods=['POST'])
def generate_bill():
    data = request.json
    Patient_ID = data.get('Patient_ID')
    
    prescription = prescriptions_collection.find_one({'Patient_ID': Patient_ID})
    
    if not prescription:
        return jsonify({'message': 'No prescription found for this Patient_ID'}), 440

    Medicine_name = prescription['Prescription1']['Medicine']
    Quantity = prescription['Prescription1']['Quantity']

    medicine_doc = medicines_collection.find_one({"medicines.medicine": Medicine_name}, {"medicines.$": 1})
    
    if not medicine_doc or 'medicines' not in medicine_doc:
        return jsonify({'message': f'No medicine found with name {Medicine_name}'}), 404

    Cost_per_unit = medicine_doc['medicines'][0]['cost']
    Total_cost = Cost_per_unit * Quantity

    Current_date = datetime.now().isoformat()

    bill = {
        'Patient_ID': Patient_ID,
        'Medicine_name': Medicine_name,
        'Quantity': Quantity,
        'Cost_per_unit': Cost_per_unit,
        'Total_cost': Total_cost,
        'date': Current_date
    }
    
    bills_collection.insert_one(bill)

    return jsonify({
        'message': 'Bill generated successfully',
        'Patient_ID': Patient_ID,
        'Medicine_name': Medicine_name,
        'Quantity': Quantity,
        'Cost_per_unit': Cost_per_unit,
        'Total_cost': Total_cost,
        'date': Current_date
    }), 200

# Get patient profile
@app.route('/get_patient_profile', methods=['POST'])
def get_patient_profile():
    data = request.json
    Patient_ID = data.get('Patient_ID')
    
    if not Patient_ID:
        return jsonify({'status': 'error', 'message': 'Patient_ID is required'}), 400

    user = users_collection.find_one({'Patient_ID': Patient_ID})
    
    if user:
        user['_id'] = str(user['_id'])
        return jsonify({'status': 'success', 'user': user}), 200
    else:
        return jsonify({'status': 'error', 'message': 'User not found'}), 404

# Process chatbot request
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    message = data.get('message')

    response = model.generate_content(message)

    generated_message = response.candidates[0].content.parts[0].text

    return jsonify({"message": generated_message}), 200
