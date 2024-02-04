import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests  # Add this import for making HTTP requests

# Load the saved models
loaded_modelN = joblib.load("svr_N.joblib")  
loaded_modelP = joblib.load("svr_P.joblib")
loaded_modelK = joblib.load("svr_K.joblib")

def nutrient_status(predicted_values, ideal_values):
    ratios = {}
    for nutrient in ['N', 'P', 'K']:
        # Get the predicted value for the nutrient
        predicted_value = predicted_values[nutrient]

        # Get the ideal ratio for the nutrient for the selected crop
        ideal_ratio = ideal_values[nutrient]

        # Calculate the ratio by dividing predicted value by the minimum among them
        ratio = predicted_value / min(predicted_values.values())

        # Compare the predicted ratio with the ideal ratio
        if ratio > ideal_ratio:
            # Calculate the percentage difference
            percentage_difference = ((ratio - ideal_ratio) / ideal_ratio) * 100

            # Categorize based on percentage difference for higher levels
            if percentage_difference <= 25:
                status = f'Little Higher {nutrient}'
            elif 25 < percentage_difference <= 50:
                status = f'Moderately High {nutrient}'
            elif 50 < percentage_difference <= 75:
                status = f'Quite High {nutrient}'
            elif 75 < percentage_difference <= 100:
                status = f'Very High {nutrient}'
            else:
                status = f'Excessively High {nutrient}'
        elif ratio < ideal_ratio:
            # Calculate the percentage difference for lower levels
            percentage_difference = ((ideal_ratio - ratio) / ideal_ratio) * 100

            # Categorize based on percentage difference for lower levels
            if percentage_difference <= 25:
                status = f'Little Lower {nutrient}'
            elif 25 < percentage_difference <= 50:
                status = f'Moderately Low {nutrient}'
            elif 50 < percentage_difference <= 75:
                status = f'Quite Low {nutrient}'
            elif 75 < percentage_difference <= 100:
                status = f'Very Low {nutrient}'
            else:
                status = f'Excessively Low {nutrient}'
        else:
            status = f'Optimal {nutrient}'

        ratios[nutrient] = {'Ratio': ratio, 'Status': status}

    return ratios


def get_thingspeak_data(api_key, field_num):
    channel_id = 2416189  # Your ThingSpeak channel ID

    url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{field_num}.json?api_key={api_key}&results=1"
    api_key = "HLWOR8TTLR2V1P5D"
    response = requests.get(url)
    data = response.json()

    # Check if 'feeds' key exists and is not empty
    if 'feeds' in data and data['feeds']:
        latest_value = data['feeds'][0].get('field{}'.format(field_num), None)
        return latest_value
    else:
        st.warning("No recent data points found.")
        return None

read_api_key = "HLWOR8TTLR2V1P5D"

temperature = get_thingspeak_data(read_api_key, 1)
humidity = get_thingspeak_data(read_api_key, 2)
ph = get_thingspeak_data(read_api_key, 3)
rainfall = get_thingspeak_data(read_api_key, 4)
new_data = np.array([[temperature, humidity, ph, rainfall]])


# Check if ThingSpeak data is available, otherwise use default values
st.sidebar.header('User Input:')
temperature_user = st.sidebar.slider('Temperature:', 0.0, 40.0, 25.0)
humidity_user = st.sidebar.slider('Humidity:', 0.0, 100.0, 50.0)
ph_user = st.sidebar.slider('pH:', 0.0, 14.0, 7.0)
rainfall_user = st.sidebar.slider('Rainfall:', 0.0, 500.0, 200.0)
new_data_user = np.array([[temperature_user, humidity_user, ph_user, rainfall_user]])


prediction_source = st.radio("Choose Prediction Source:", ["ThingSpeak Data", "User Input"])

if prediction_source == "ThingSpeak Data":
    if any(value is None for value in [temperature, humidity, ph, rainfall]):
        st.warning("ThingSpeak data incomplete. Using default inputs.")
        new_data=np.array([[25,80,7,200]])
        predicted_valuesN = loaded_modelN.predict(new_data)
        predicted_valuesP = loaded_modelP.predict(new_data)
        predicted_valuesK = loaded_modelK.predict(new_data)
    else:
        predicted_valuesN = loaded_modelN.predict(new_data)
        predicted_valuesP = loaded_modelP.predict(new_data)
        predicted_valuesK = loaded_modelK.predict(new_data)
else:
    predicted_valuesN = loaded_modelN.predict(new_data_user)
    predicted_valuesP = loaded_modelP.predict(new_data_user)
    predicted_valuesK = loaded_modelK.predict(new_data_user)

print("Predicted N:", predicted_valuesN[0])
print("Predicted P:", predicted_valuesP[0])
print("Predicted K:", predicted_valuesK[0])

# Display the predicted values
st.write('**Predicted Nutrient Values:**')
st.write(f'Predicted N: {predicted_valuesN[0]}')
st.write(f'Predicted P: {predicted_valuesP[0]}')
st.write(f'Predicted K: {predicted_valuesK[0]}')

##########################################
# Displaying data on server

predicted_values = {
    "Nitrogen": predicted_valuesN[0],
    "Phosphorous": predicted_valuesP[0],
    "Potassium": predicted_valuesK[0],
}

# Make a POST request to the server
server_url = "https://blushing-sun-hat-lion.cyclic.app/products"
response = requests.post(server_url, {'Nitrogen':49.86,'Pottasium':51.95,'Phosphorus':34.28})

# Check the response status
if response.status_code == 200:
    print("Before POST request")
    st.success("Predicted nutrient values successfully sent to the server.")
    print("Success")
else:
    print("Failed")
    st.error(f"Failed to send predicted nutrient values. Status code: {response.status_code}")
#################################################




df_chart = pd.DataFrame({
    'Nutrient': ['N', 'P', 'K'],
    'Predicted Value': [predicted_valuesN[0], predicted_valuesP[0], predicted_valuesK[0]]
})

# Specify custom colors for the bar graph
bar_colors = ['#FF5733', '#33FF57', '#5733FF']

fig, ax = plt.subplots(figsize=(8, 3))
sns.barplot(x='Predicted Value', y='Nutrient', data=df_chart, ax=ax, palette=bar_colors)
ax.set_xlabel('Predicted Value')
ax.set_ylabel('Nutrient')
ax.tick_params(axis='both', labelsize=10)  # Adjust text size

# Adjust font style
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(12)  # Adjust text size

# Adjust chart lines thickness
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)

st.pyplot(fig)
# Ideal nutrient values for different crops
ideal_values = {
    'Rice': {'N': 4, 'P': 2, 'K': 1},
    'Maize': {'N': 1.5, 'P': 1, 'K': 1.5},
    'Wheat': {'N': 4, 'P': 3, 'K': 1},
    'Cotton': {'N': 4, 'P': 2, 'K': 2},
    'Sugarcane': {'N': 2.2, 'P': 1, 'K': 1},
    'Saffron': {'N': 2.87, 'P': 1, 'K': 1.71}
}

# User selects the crop
selected_crop = st.selectbox('Select Crop:', list(ideal_values.keys()))

# Display the ideal nutrient values
ideal_values_crop = ideal_values[selected_crop]

# Compare predicted ratio with ideal ratio
predicted_ratio = {'N': predicted_valuesN[0], 'P': predicted_valuesP[0], 'K': predicted_valuesK[0]}
ideal_ratio = {'N': ideal_values_crop["N"], 'P': ideal_values_crop["P"], 'K': ideal_values_crop["K"]}

# Calculate nutrient status
nutrient_ratios = nutrient_status(predicted_ratio, ideal_ratio)

# Display the nutrient status
st.write('**Nutrient Status:**')
for nutrient, values in nutrient_ratios.items():
    st.write(f'{nutrient}: {values["Status"]} (Ratio: {values["Ratio"]})')



# Display ideal nutrient ratios in a creative box at top right
st.markdown(
    f'<div style="position:absolute; top:5%; right:5%; background-color:#f1f1f1; padding: 10px; border-radius: 10px; text-align: center;">'
    f'<h4 style="margin-bottom:5px; color:#000000">Ideal Nutrient Ratios for {selected_crop}</h4>'
    f'<p style="margin: 0; color:#FF5733;">N: {ideal_values_crop["N"]}</p>'
    f'<p style="margin: 0; color:#33FF57;">P: {ideal_values_crop["P"]}</p>'
    f'<p style="margin: 0; color:#5733FF;">K: {ideal_values_crop["K"]}</p>'
    f'</div>',
    unsafe_allow_html=True
)
