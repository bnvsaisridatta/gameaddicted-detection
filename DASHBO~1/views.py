from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Prediction
from django.db.models import Count

import pandas as pd
import joblib
import time

encoder = joblib.load("model/onehot_encoder.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
scaler = joblib.load("model/scaler.pkl")
model = joblib.load("model/stacking_model.pkl")

@login_required
def dashboard_view(request):
    return render(request, 'dashboard/dashboard.html')

@login_required
def predict_view(request):

    result = None
    confidence = None

    if request.method == "POST":

        input_data = {
            "age": float(request.POST.get("age")),
            "gender": request.POST.get("gender"),
            "daily_gaming_hours": float(request.POST.get("daily_gaming_hours")),
            "game_genre": request.POST.get("game_genre"),
            "primary_game": request.POST.get("primary_game"),
            "gaming_platform": request.POST.get("gaming_platform"),
            "sleep_hours": float(request.POST.get("sleep_hours")),
            "sleep_quality": request.POST.get("sleep_quality"),
            "sleep_disruption_frequency": request.POST.get("sleep_disruption_frequency"),
            "academic_work_performance": request.POST.get("academic_work_performance"),
            "grades_gpa": float(request.POST.get("grades_gpa")),
            "work_productivity_score": float(request.POST.get("work_productivity_score")),
            "mood_state": request.POST.get("mood_state"),
            "mood_swing_frequency": request.POST.get("mood_swing_frequency"),
            "withdrawal_symptoms": request.POST.get("withdrawal_symptoms") == "true",
            "loss_of_other_interests": request.POST.get("loss_of_other_interests") == "true",
            "continued_despite_problems": request.POST.get("continued_despite_problems") == "true",
            "eye_strain": request.POST.get("eye_strain") == "true",
            "back_neck_pain": request.POST.get("back_neck_pain") == "true",
            "weight_change_kg": float(request.POST.get("weight_change_kg")),
            "exercise_hours_weekly": float(request.POST.get("exercise_hours_weekly")),
            "social_isolation_score": float(request.POST.get("social_isolation_score")),
            "face_to_face_social_hours_weekly": float(request.POST.get("face_to_face_social_hours_weekly")),
            "monthly_game_spending_usd": float(request.POST.get("monthly_game_spending_usd")),
            "years_gaming": float(request.POST.get("years_gaming")),
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
        confidence = float(probabilities.max())

        result = predicted_class

        # Save prediction
        Prediction.objects.create(
            user=request.user,
            input_data=input_data,
            predicted_class=predicted_class,
            confidence=confidence
        )

    return render(request, "dashboard/predict.html", {
        "result": result,
        "confidence": confidence
    })

@login_required
def history_view(request):
    qs = Prediction.objects.filter(user=request.user)

    # Count per class
    class_counts = qs.values('predicted_class').annotate(count=Count('id'))

    labels = [item['predicted_class'] for item in class_counts]
    data = [item['count'] for item in class_counts]

    return render(request, 'dashboard/history.html', {'labels': labels,'data': data,})

@login_required
def profile_page(request):
    profile = request.user.profile
    return render(request, 'dashboard/profile.html', {'profile': profile})

@login_required
def my_predictions(request):
    predictions = Prediction.objects.filter(user=request.user)
    return render(request, 'dashboard/my_predictions.html', {'predictions': predictions})
