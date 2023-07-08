
import pandas as pd
import streamlit as st
#from pickle import dump, load

from PIL import Image
#import numpy as np

#with open('model2.pickle', 'rb') as f:
    #model = load(f)

st.write(""" 
# Happy plane :airplane: !

Choose parameters to find out if a passenger is satisfied with their journey.""")

img = Image.open("family_airplane.png")
st.image(img, width=400)

st.sidebar.header('Passenger Input Parameters')

def user_input_features():
    gender = st.sidebar.selectbox('Is a passenger a male?', [0, 1])
    age = st.sidebar.slider('How old is the passenger?', 0, 120, 47)
    customer_type = st.sidebar.selectbox('Is the passenger a disloyal customer?', [0, 1])
    type_of_travel = st.sidebar.selectbox('Is it a personal travel?', [0, 1])
    class_of_travel = st.sidebar.selectbox('Is it an Eco class travel?', [0, 1])
    ecoplus = st.sidebar.selectbox('Is it an EcoPlus class travel?', [0, 1])
    flight_distance = st.sidebar.slider('Please choose flight distance:', 0, 4000, 1700)
    departure_delay_in_minutes = st.sidebar.slider('Departure delay in minutes:', 0, 180, 100)
    arrival_delay_in_minutes = st.sidebar.slider('Arrival delay in minutes:', 0, 180, 100)
    inflight_wifi_service = st.sidebar.selectbox('Quality of inflight wifi service:', [1, 2, 3, 4, 5])
    departure_arrival_time_convenient = st.sidebar.selectbox('Time convenience:', [1, 2, 3, 4, 5])
    ease_of_online_booking = st.sidebar.selectbox('Ease of online booking:', [1, 2, 3, 4, 5])
    gate_location = st.sidebar.selectbox('Gate location:', [1, 2, 3, 4, 5])
    food_and_drink = st.sidebar.selectbox('Quality of food and drinks:', [1, 2, 3, 4, 5])
    online_boarding = st.sidebar.selectbox('Ease of online boarding:', [1, 2, 3, 4, 5])
    seat_comfort = st.sidebar.selectbox('Seat comfort:', [1, 2, 3, 4, 5])
    inflight_entertainment = st.sidebar.selectbox('Quality of inflight entertainment:', [1, 2, 3, 4, 5])
    on_board_service = st.sidebar.selectbox('Quality of onboard service:', [1, 2, 3, 4, 5])
    leg_room_service = st.sidebar.selectbox('Leg room service:', [1, 2, 3, 4, 5])
    baggage_handling = st.sidebar.selectbox('Baggage handling:', [1, 2, 3, 4, 5])
    checkin_service = st.sidebar.selectbox('Ease of checkin:', [1, 2, 3, 4, 5])
    inflight_service = st.sidebar.selectbox('Quality of inflight service:', [1, 2, 3, 4, 5])
    cleanliness = st.sidebar.selectbox('Cleanliness', [1, 2, 3, 4, 5])
    data = {'gender_male':gender,
    'age':age,
    'disloyal_customer':customer_type,
    'personal_travel':type_of_travel,
    'eco_class':class_of_travel,
    'eco_plus_class':ecoplus,
    'flight_distance':flight_distance,
    'departure_delay':departure_delay_in_minutes,
    'arrival_delay':arrival_delay_in_minutes,
    'inflight_service':inflight_service,
    'cleanliness':cleanliness,
    'wifi':inflight_wifi_service,
    'time_convenient':departure_arrival_time_convenient,
    'online_booking':ease_of_online_booking,
    'gate_location':gate_location,
    'food':food_and_drink,
    'online_boarding':online_boarding,
    'seat_comfort':seat_comfort,
    'entertainment':inflight_entertainment,
    'onboard':on_board_service,
    'leg_room':leg_room_service,
    'baggage':baggage_handling,
    'checkin_service':checkin_service}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Passenger input parameters')
st.write(df)

#clients1 = pd.read_fwf('C:/Users/Suntory1/PycharmProjects/streamlit_for_HSE/clients1.csv')
#X = clients1.drop(columns=['dissatisfied', 'satisfied'], axis=0, inplace=True)
#y = clients1[['dissatisfied', 'satisfied']]

X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')['satisfied']

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter = 1000)
model.fit(X, y)
prediction = model.predict(df)
probability = model.predict_proba(df)

#st.subheader('Corresponding index number')
#st.write(y.columns)

st.subheader('Prediction')
st.write(f'Passenger is satisfied with probability {probability[:,1].round(4).item()}')



#columns = [gender, age, customer_type, type_of_travel, class_of_travel, flight_distance, departure_delay_in_minutes,
                    #arrival_delay_in_minutes,inflight_wifi_service, departure_arrival_time_convenient, ease_of_online_booking,
                    #gate_location, food_and_drink, online_boarding, seat_comfort, inflight_entertainment, on_board_service,
                    #leg_room_service, baggage_handling, checkin_service, inflight_service, cleanliness]

#def predict():
    #row = np.array([gender, age, customer_type, type_of_travel, class_of_travel, flight_distance, departure_delay_in_minutes,
                    #arrival_delay_in_minutes,inflight_wifi_service, departure_arrival_time_convenient, ease_of_online_booking,
                    #gate_location, food_and_drink, online_boarding, seat_comfort, inflight_entertainment, on_board_service,
                    #leg_room_service, baggage_handling, checkin_service, inflight_service, cleanliness])
   # X = pd.DataFrame([row], columns = columns)
    #X.columns = X.columns.astype(str)
    #prediction = model.predict(X)[0]

    #if prediction ==1:
        #st.success('Passenger is happy :thumbsup:!')
    #else:
        #st.error('Passenger is not satisfied with the service :thumbsdown:')

#st.button('Predict', on_click=predict)