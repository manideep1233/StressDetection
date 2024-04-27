import sys
import os
import glob
import re
import numpy as np
import pandas as pd

from flask import Flask, request, render_template
from PIL import Image
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask import Flask , render_template , request , url_for
import pickle

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from time import sleep, time

import tensorflow as tf
from tensorflow.keras import models, layers
import math
import matplotlib

# Use the 'agg' backend for Matplotlib,  whihc is neccesarily need for matplotlib module in flask. 
matplotlib.use('agg')
import matplotlib.pyplot as plt


from matplotlib.image import imread
import cv2
import csv
from PIL import Image

app = Flask(__name__)


################################ All pages server connection/routings ##########################################

@app.route("/", methods=["GET", "POST"]) 
def runhome():
    return render_template("index.html") 

@app.route("/server_home", methods=["GET", "POST"]) 
def server_home():
    return render_template("index.html") 

@app.route("/about", methods=["GET", "POST"]) 
def about():
    return render_template("about.html") 

@app.route("/testimonial", methods=["GET", "POST"]) 
def testimonial():
    return render_template("testimonial.html") 

@app.route("/404", methods=["GET", "POST"]) 
def _404():
    return render_template("404.html") 

@app.route("/contact", methods=["GET", "POST"]) 
def contact():
    return render_template("contact.html") 

@app.route("/stressprediction", methods=["GET", "POST"]) 
def stressprediction():
    return render_template("stressprediction.html") 

@app.route("/stressanalysis", methods=["GET", "POST"]) 
def stressanalysis():
    return render_template("stressanalysis.html") 

############################---   stress analysis ---########################################

@app.route("/stress_analyze", methods=["GET", "POST"])
def stress_analyze():
    if request.method == 'POST':

        # Load the pre-trained face and emotion models
        face_classifier = cv2.CascadeClassifier(r'model/haarcascade_frontalface_default.xml')
        stress_classifier = load_model(r'model/model.h5')

        # Define emotion labels
        stress_labels = ['Bursted', 'Irritated', 'Anxious', 'Relaxed', 'Neutral', 'Broked', 'Shocked']

        # Open the default camera (usually the built-in webcam)
        cap = cv2.VideoCapture(0)

        # Create a CSV file to log expressions
        # Rename the respective csv file
        csv_file_path = 'naveen.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Timestamp', 'PredictedEmotion'])

        while True:
            # Read a frame from the camera
            _, frame = cap.read()

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                
                # Extract the region of interest (ROI) and resize it for emotion detection
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    # Normalize and preprocess the ROI for emotion prediction
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # Predict the emotion from the ROI
                    prediction = stress_classifier.predict(roi)[0]
                    predicted_emotion = stress_labels[prediction.argmax()]

                    # Log timestamp and predicted emotion to CSV file
                    timestamp = time()
                    with open(csv_file_path, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([timestamp, predicted_emotion])

                    # Display the predicted emotion label on the frame
                    label_position = (x, y)
                    cv2.putText(frame, predicted_emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame with emotion detection information
            cv2.imshow('Stress Detector', frame)

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close the OpenCV windows
        cap.release()
        cv2.destroyAllWindows()


        return render_template("stressanalysis2.html")

################################################################################################

@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    if request.method == 'POST':



        #****************************************************************************#
        # Load the CSV file into a Pandas DataFrame
        # Here we will read the respected .csv file
        csv_file_path = 'naveen.csv'
        df = pd.read_csv(csv_file_path)

        # Convert the 'Timestamp' column to datetime format and set the timezone to UTC
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s').dt.tz_localize('UTC')
        # Convert the timezone to Indian Standard Time (IST)
        df['Timestamp'] = df['Timestamp'].dt.tz_convert('Asia/Kolkata')



        #****************************************************************************#
        # Generate Emotion Trends over Time
        plt.figure(figsize=(10, 6))
        plt.plot(df['Timestamp'], df['PredictedEmotion'], marker='o')
        plt.title('Emotion Trends over Time')
        plt.xlabel('Timestamp (IST)')
        plt.ylabel('Predicted Emotion')
        plt.xticks(rotation=45)
        #plt.show()

        # Save the plot as an image file
        emotion_trends_time_chart = 'static/plot/emotion_trends_time_chart.png'
        plt.savefig(emotion_trends_time_chart, bbox_inches='tight')
        plt.close()



        #****************************************************************************#
        # Additional Analysis: Emotion Distribution over Time
        plt.figure(figsize=(200, 10))
        df['Date'] = df['Timestamp'].dt.date
        emotion_distribution_over_time = df.groupby(['Date', 'PredictedEmotion']).size().unstack().fillna(0)
        emotion_distribution_over_time.plot(kind='bar', stacked=True)
        plt.title('Emotion Distribution over Time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend(title='Predicted Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        #plt.show()

        # Save the plot as an image file
        emotion_distribution_time_chart = 'static/plot/emotion_distribution_time_chart.png'
        plt.savefig(emotion_distribution_time_chart, bbox_inches='tight')
        plt.close()



        #****************************************************************************#
        # Additional Analysis: Average Stress Level every 20 seconds
        average_stress_level_per_20_seconds = df.groupby(df['Timestamp'].dt.floor('20S'))['PredictedEmotion'].value_counts().groupby(level=0).mean()
        plt.figure(figsize=(20, 6))
        average_stress_level_per_20_seconds.plot(kind='line', marker='o')
        plt.title('Average Stress Level every 20 seconds')
        plt.xlabel('Timestamp')
        plt.ylabel('Average Stress Level')
        plt.xticks(rotation=45)
        #plt.show()

        # Save the plot as an image file
        average_stress_level_per_minute_chart = 'static/plot/average_stress_level_per_minute_chart.png'
        plt.savefig(average_stress_level_per_minute_chart, bbox_inches='tight')
        plt.close()



        #****************************************************************************#
        # Additional Analysis: Daily Average Stress Level
        daily_average_stress_level = df.groupby('Date')['PredictedEmotion'].value_counts().groupby('Date').mean()
        plt.figure(figsize=(10, 6))
        daily_average_stress_level.plot(kind='line', marker='o')
        plt.title('Daily Average Stress Level')
        plt.xlabel('Date')
        plt.ylabel('Average Stress Level')
        plt.xticks(rotation=45)
        #plt.show()

        # Save the plot as an image file
        daily_average_stress_level_chart = 'static/plot/daily_average_stress_level_chart.png'
        plt.savefig(daily_average_stress_level_chart, bbox_inches='tight')
        plt.close()

        print(df['PredictedEmotion'].value_counts())
        print("***********************************")
        print(df['PredictedEmotion'].value_counts().mean())

        



        #****************************************************************************#
        # Create an Emotion Distribution Chart
        emotion_distribution = df['PredictedEmotion'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(emotion_distribution, labels=emotion_distribution.index, autopct='%1.1f%%', startangle=90)
        plt.title('Emotion Distribution')
        #plt.show()

        # Save the plot as an image file
        emotion_distribution_chart = 'static/plot/emotion_distribution_chart.png'
        plt.savefig(emotion_distribution_chart, bbox_inches='tight')
        plt.close()




        #****************************************************************************#
        # Average Stress Level Bar Chart
        plt.figure(figsize=(10, 6))
        df.groupby('PredictedEmotion')['Timestamp'].count().plot(kind='bar', color='skyblue')
        plt.title('Average Stress Level by Emotion')
        plt.xlabel('Predicted Emotion')
        plt.ylabel('Count')
        #plt.show()

        # Save the plot as an image file
        average_stress_level_bar_chart = 'static/plot/average_stress_level_bar_chart.png'
        plt.savefig(average_stress_level_bar_chart, bbox_inches='tight')
        plt.close()




        #****************************************************************************#
        import random

        # Generate random colors for each emotion
        colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in df['PredictedEmotion'].unique()]

        # Emotion Trends Stacked Area Chart
        df['EmotionIndex'] = df['PredictedEmotion'].astype('category').cat.codes
        df = df.sort_values(by='Timestamp')
        plt.figure(figsize=(10, 6))
        plt.stackplot(df['Timestamp'], df['EmotionIndex'], labels=df['PredictedEmotion'].unique(), colors=colors)
        plt.legend(loc='upper right')
        plt.title('Emotion Trends over Time (Stacked Area)')
        plt.xlabel('Timestamp')
        plt.ylabel('Emotion')
        #plt.show()

        # Save the plot as an image file
        emotion_trends_stacked_area_chart = 'static/plot/emotion_trends_stacked_area_chart.png'
        plt.savefig(emotion_trends_stacked_area_chart, bbox_inches='tight')
        plt.close()



        #****************************************************************************#
        # Dummy data for recommendations
        stress_recommendations = {
            'Bursted': ['Listen to calming music', 'Take a short break', 'Practice deep breathing',
                        'Engage in a physical activity', 'Express your feelings through journaling'],
            'Irritated': ['Try a mindfulness exercise', 'Take a walk outside', 'Stretch for a few minutes',
                           'Practice progressive muscle relaxation', 'Identify and challenge negative thoughts'],
            'Anxious': ['Practice guided meditation', 'Write down your thoughts', 'Focus on positive affirmations',
                        'Use visualization techniques', 'Establish a routine for better predictability'],
            'Relaxed': ['Continue your positive routine', 'Consider trying a new hobby', 'Reflect on your achievements',
                        'Connect with loved ones', 'Enjoy a favorite book or movie'],
            'Neutral': ['Maintain your balance', 'Stay hydrated', 'Check-in with your emotions',
                        'Take short breaks to refresh', 'Practice mindfulness throughout the day'],
            'Broked': ['Reach out to a friend', 'Engage in self-care activities', 'Reflect on self-improvement',
                       'Seek professional support if needed', 'Create a gratitude list'],
            'Shocked': ['Pause and assess the situation', 'Take deep breaths', 'Seek support if needed',
                        'Journal about your experience', 'Gradually reintroduce routine activities'],
        }


        #****************************************************************************#
        # Dummy data for average stress level suggestions
        average_stress_level_suggestions = [
            'Take a short break and stretch',
            'Practice mindfulness meditation',
            'Go for a walk in nature',
            'Listen to calming music',
            'Deep breathing exercises',
            'Engage in a hobby you enjoy',
            'Disconnect from digital devices for a while',
            'Spend time with loved ones',
            'Practice positive affirmations',
            'Consider seeking professional support if needed'
        ]

        # Display Motivational Messages and Recommendations for High Average Stress Level




        #****************************************************************************#
        # Calculate Average Stress Level
        average_stress_level = df['PredictedEmotion'].value_counts().mean()
        print("Based on our analysis of your emotional expressions, the average stress level is currently at ")
        print(f'Average Stress Level: {average_stress_level:.2f}')


        # Well-being Score Calculation (example scores, adjust as needed)
        print("On a scale of 1-7, your well-being score is:")
        emotion_scores = {'Relaxed': 7, 'Neutral': 6, 'Shocked': 5, 'Anxious': 4, 'Broked': 3, 'Bursted': 2, 'Irritated': 1}
        df['EmotionScore'] = df['PredictedEmotion'].map(emotion_scores)
        well_being_score = df['EmotionScore'].mean()
        print(f'Well-being Score: {well_being_score:.2f}')

        print("\nMeasures are as follows:")
        print("'Relaxed': 7, 'Neutral': 6, 'Shocked': 5, 'Anxious': 4, 'Broked': 3, 'Bursted': 2, 'Irritated': 1")




        #****************************************************************************#
        # Example: Display a message if overall stress level is high
        average_stress_level_result = "Healthy, But you can improve with few suggestions."
        if average_stress_level > 3.0:
            print("High overall stress level observed. Take a moment to relax and practice mindfulness.")
            print("Here are some recommendations:")
            average_stress_level_result = f"High overall stress level observed. Take a moment to relax and practice mindfulness. "

            '''for recommendation in average_stress_level_suggestions:
                print(f"- {recommendation}")
                average_stress_level_result = average_stress_level_result + ". " + recommendation'''
                



        #****************************************************************************#
        # Example: Display a message if there is a sudden change in emotion
        sudden_shock_change_result = "Healthy, You didn't shocked suddenly. but, you can take some suggestions."
        if df['PredictedEmotion'].nunique() > 1 and df['PredictedEmotion'].value_counts().idxmax() == 'Shocked':
            print("Sudden shift to 'Shocked' emotion. Check-in with yourself and address any concerns.")
            print("Here are some recommendations:")
            sudden_shock_change_result = "Sudden shift to 'Shocked' emotion. Check-in with yourself and address any concerns." + "Here are some recommendations:"

            for recommendation in stress_recommendations['Shocked']:
                print(f"- {recommendation}")
                sudden_shock_change_result = sudden_shock_change_result + ". " + recommendation




        # Display Motivational Messages and Recommendations Based on Trends

        #****************************************************************************#
        # Example: Display recommendations if 'Shocked' emotion is frequent
        shocked_result = "Healthy, But you can improve with few suggestions."
        shocked_count = emotion_distribution.get('Shocked', 0)
        if shocked_count > 5:
            print(f"Frequent 'Shocked' expressions detected. Consider taking a break and relaxing.")
            print("Here are some recommendations:")
            shocked_result = f"Frequent 'Shocked' expressions detected ({shocked_count} times). Consider taking a break and relaxing."

            '''for recommendation in stress_recommendations['Shocked']:
                print(f"- {recommendation}")
                shocked_result = shocked_result + ". " + recommendation'''

                


        #****************************************************************************#
        # Example: Display a message if 'Anxious' emotion is consistently high
        anxious_result = "Healthy, But you can improve with few suggestions."
        anxious_count = emotion_distribution.get('Anxious', 0)
        if anxious_count > 3:
            print(f"Consistent high levels of 'Anxious' emotion. Consider activities to reduce anxiety.")
            print("Here are some recommendations:")
            anxious_result = f"Consistent high levels of 'Anxious' emotion ({anxious_count} times). Consider activities to reduce anxiety."
            '''for recommendation in stress_recommendations['Anxious']:
                print(f"- {recommendation}")
                anxious_result = anxious_result + ". " + recommendation'''



                
        #****************************************************************************#
        # Example: Display a positive message if 'Relaxed' emotion is predominant
        relaxed_result = "Healthy, But you can improve with few suggestions."
        relaxed_count = emotion_distribution.get('Relaxed', 0)
        if relaxed_count > 10:
            print(f"Frequent 'Relaxed' expressions detected. You're doing great! Keep up the positive vibes.")
            print("Here are some recommendations:")
            relaxed_result = f"Frequent 'Relaxed' expressions detected ({relaxed_count} times). You're doing great! Keep up the positive vibes."
            '''for recommendation in stress_recommendations['Relaxed']:
                print(f"- {recommendation}")
                relaxed_result = relaxed_result + ". " + recommendation'''

                


        #****************************************************************************#
        # Example: Display recommendations if 'Irritated' emotion is frequent
        broked_result = "Healthy, But you can improve with few suggestions."
        broked_count = emotion_distribution.get('Broked', 0)
        if broked_count > 5:
            print(f"Frequent 'Broked' expressions detected. Try these recommendations:")
            broked_result = f"Frequent 'Broked' expressions detected ({broked_count} times)."
            '''for recommendation in stress_recommendations['Irritated']:
                print(f"- {recommendation}")
                broked_result = broked_result + ". " + recommendation'''
                



        #****************************************************************************#
        # Example: Display recommendations if 'Bursted' emotion is frequent
        bursted_result = "Healthy, But you can improve with few suggestions."
        bursted_count = emotion_distribution.get('Bursted', 0)
        if bursted_count > 5:
            print(f"Frequent 'Bursted' expressions detected. Consider these recommendations:")
            bursted_result = f"Frequent 'Bursted' expressions detected ({bursted_count} times). "
            '''for recommendation in stress_recommendations['Bursted']:
                print(f"- {recommendation}")
                bursted_result = bursted_result + ". " + recommendation'''



                
        #****************************************************************************#
        # Example: Display recommendations if 'Neutral' emotion is frequent
        neutral_result = "Healthy, But you can improve with few suggestions."
        neutral_count = emotion_distribution.get('Neutral', 0)
        if neutral_count > 10:  
            print(f"Frequent 'Neutral' expressions detected. Here are some recommendations:")
            neutral_result = f"Frequent 'Neutral' expressions detected ({neutral_count} times)."
            '''for recommendation in stress_recommendations['Neutral']:
                print(f"- {recommendation}")
                neutral_result = neutral_result + ". " + recommendation'''



        return render_template("stress_analysis_result.html", 
            emotion_trends_time_chart=emotion_trends_time_chart,
            emotion_distribution_time_chart=emotion_distribution_time_chart,
            average_stress_level_per_minute_chart=average_stress_level_per_minute_chart,
            daily_average_stress_level_chart=daily_average_stress_level_chart,
            emotion_distribution_chart=emotion_distribution_chart,
            average_stress_level_bar_chart=average_stress_level_bar_chart,
            emotion_trends_stacked_area_chart=emotion_trends_stacked_area_chart,

            average_stress_level=average_stress_level,
            well_being_score=well_being_score,

            average_stress_level_result=average_stress_level_result,
            sudden_shock_change_result=sudden_shock_change_result,

            shocked_result=shocked_result,
            anxious_result=anxious_result,
            relaxed_result=relaxed_result,
            broked_result=broked_result,
            bursted_result=bursted_result,
            neutral_result=neutral_result,

            shocked_count=shocked_count,
            anxious_count=anxious_count,
            relaxed_count=relaxed_count,
            broked_count=broked_count,
            bursted_count=bursted_count,
            neutral_count=neutral_count,
            )


################################################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
