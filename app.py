import math
from datetime import datetime
from flask import Flask, render_template, request

app = Flask(__name__)

# Global constant: recommended base price
BASE_PRICE_DEFAULT = 106

def predict_final_price(
    check_in_date,
    booking_date,
    star_numeric,
    customer_rating_num,
    reviews,
    free_wifi_bin,
    free_parking_bin,
    stay_month,
    cluster,
    base_price=BASE_PRICE_DEFAULT
):
    # Convert strings -> datetime
    if isinstance(check_in_date, str):
        check_in_date = datetime.strptime(check_in_date, "%Y-%m-%d")
    if isinstance(booking_date, str):
        booking_date = datetime.strptime(booking_date, "%Y-%m-%d")

    # lead_time
    lead_time = (check_in_date - booking_date).days

    # is_weekend
    is_weekend = 1 if check_in_date.weekday() in [5, 6] else 0

    # Static location_baseline_price
    location_baseline_price = 106

    # log(reviews) safe
    log_reviews = math.log(max(reviews, 1))

    # Interaction terms
    lead_time_x_weekend = lead_time * is_weekend
    rating_x_reviews = customer_rating_num * log_reviews

    # Month effect (x)
    month_effect = {
        "january": 0.1128,
        "february": 0.1284,
        "march": 0.1654,
        "april": 0.2094,
        "may": 0.2560,
        "june": 0.3454,
        "july": 0.4491,
        "august": 0.4245,
        "september": 0.4548
    }

    stay_month = stay_month.strip().lower()
    x = month_effect.get(stay_month, 0)

    # Regression prediction (y_pred)
    y_pred = (
        2.55
        - 0.0008 * lead_time
        + 0.2047 * star_numeric
        + 0.1467 * customer_rating_num
        - 0.1184 * log_reviews
        + 0.0059 * location_baseline_price
        - 0.1462 * is_weekend
        + 0.0871 * free_wifi_bin
        - 0.0272 * free_parking_bin
        + 0.0222 * rating_x_reviews
        + 0.0007 * lead_time_x_weekend
        + x
    )

    # Alpha based on cluster
    alpha_map = {
        "dynamic": 1.5,
        "moderate": 1.0,
        "stable": 0.5
    }

    cluster_clean = cluster.strip().lower()
    alpha = alpha_map.get(cluster_clean, 1.0)

    # Final Price: base_price + (exp(y_pred) - base_price) * alpha
    final_price = round(base_price + (math.exp(y_pred) - base_price) * alpha, 2)

    return {
        "lead_time": lead_time,
        "is_weekend": is_weekend,
        "lead_time_x_weekend": lead_time_x_weekend,
        "log_reviews": log_reviews,
        "rating_x_reviews": rating_x_reviews,
        "y_pred": y_pred,
        "stay_month": stay_month,
        "alpha": alpha,
        "base_price": base_price,
        "final_price": final_price
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            check_in_date = request.form.get("check_in_date", "").strip()
            booking_date = request.form.get("booking_date", "").strip()
            cluster = request.form.get("cluster", "").strip()

            star_numeric = float(request.form.get("star_numeric", 0))
            customer_rating_num = float(request.form.get("customer_rating_num", 0))
            reviews = int(request.form.get("reviews", 0))
            free_wifi_bin = int(request.form.get("free_wifi_bin", 0))
            free_parking_bin = int(request.form.get("free_parking_bin", 0))

            # validations
            if not check_in_date or not booking_date:
                raise ValueError("Please provide both Check-in date and Booking date.")

            check_in_dt = datetime.strptime(check_in_date, "%Y-%m-%d")
            booking_dt = datetime.strptime(booking_date, "%Y-%m-%d")
            if booking_dt > check_in_dt:
                raise ValueError("Booking date cannot be after check-in date.")

            if reviews < 0:
                raise ValueError("Reviews cannot be negative.")
            if free_wifi_bin not in (0, 1) or free_parking_bin not in (0, 1):
                raise ValueError("Free WiFi / Free Parking must be 0 or 1.")

            # Auto-calculate stay_month from check_in_date
            stay_month = check_in_dt.strftime("%B").lower()

            result = predict_final_price(
                check_in_date=check_in_date,
                booking_date=booking_date,
                star_numeric=star_numeric,
                customer_rating_num=customer_rating_num,
                reviews=reviews,
                free_wifi_bin=free_wifi_bin,
                free_parking_bin=free_parking_bin,
                stay_month=stay_month,
                cluster=cluster,
                base_price=BASE_PRICE_DEFAULT
            )

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


