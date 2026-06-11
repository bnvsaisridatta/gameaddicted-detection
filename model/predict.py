import pandas as pd
import joblib
import time

encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("stacking_model.pkl")

input_data = {
    "age": 22,
    "gender": "Female",
    "daily_gaming_hours": 1.5,
    "game_genre": "Puzzle",
    "primary_game": "Candy Crush",
    "gaming_platform": "Mobile",
    "sleep_hours": 7.8,
    "sleep_quality": "Good",
    "sleep_disruption_frequency": "Rarely",
    "academic_work_performance": "Excellent",
    "grades_gpa": 3.8,
    "work_productivity_score": 4,
    "mood_state": "Happy",
    "mood_swing_frequency": "Never",
    "withdrawal_symptoms": False,
    "loss_of_other_interests": False,
    "continued_despite_problems": False,
    "eye_strain": False,
    "back_neck_pain": False,
    "weight_change_kg": 0.1,
    "exercise_hours_weekly": 6,
    "social_isolation_score": 1,
    "face_to_face_social_hours_weekly": 12,
    "monthly_game_spending_usd": 5,
    "years_gaming": 2
}

df_input = pd.DataFrame([input_data])

categorical_cols = encoder.feature_names_in_

encoded = encoder.transform(df_input[categorical_cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_cols)
)

numeric_df = df_input.drop(columns=categorical_cols)

final_input = pd.concat(
    [numeric_df.reset_index(drop=True),
     encoded_df.reset_index(drop=True)],
    axis=1
)

final_input_scaled = scaler.transform(final_input)
start_time = time.time()
prediction = model.predict(final_input_scaled)
probabilities = model.predict_proba(final_input_scaled)
end_time = time.time()
predicted_class = label_encoder.inverse_transform(prediction)[0]
confidence = probabilities.max()

print("\nPrediction Result")
print("----------------------------")

print("Predicted Class :", predicted_class)
print("Confidence      :", round(confidence * 100, 2), "%")

print("Time Taken      :", round(end_time - start_time, 4), "seconds")