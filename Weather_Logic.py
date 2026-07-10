"""Derived weather features computed from Prophet model outputs."""

def feels_like(temp_c, humidity_pct):
    if temp_c < 27 or humidity_pct < 40:
        result = temp_c - 0.4 * (temp_c - 10) * (1 - humidity_pct / 100)
    else:
        T = temp_c * 9 / 5 + 32
        R = humidity_pct
        HI = (-42.379 + 2.04901523*T + 10.14333127*R
              - 0.22475541*T*R - 0.00683783*T*T
              - 0.05481717*R*R + 0.00122874*T*T*R
              + 0.00085282*T*R*R - 0.00000199*T*T*R*R)
        result = (HI - 32) * 5 / 9
    # Cap the delta from actual temp to a realistic range (heat index rarely
    # exceeds actual temp by more than ~12C in most real-world reporting)
    result = min(result, temp_c + 12)
    return round(result, 1)

def cloud_coverage(humidity):
    if humidity < 40:   return "Clear", 5
    elif humidity < 55: return "Partly Cloudy", 25
    elif humidity < 70: return "Mostly Cloudy", 55
    else:               return "Overcast", 85

def rain_chance(humidity, pressure):
    score = 0
    if humidity >= 90:   score += 55
    elif humidity >= 80: score += 35
    elif humidity >= 70: score += 18
    elif humidity >= 60: score += 8
    if pressure < 1000:   score += 35
    elif pressure < 1005: score += 22
    elif pressure < 1010: score += 12
    elif pressure < 1015: score += 4
    return min(int(score), 95)

def rain_perception(chance):
    if chance < 15:   return "No rain"
    elif chance < 35: return "Light drizzle"
    elif chance < 55: return "Moderate rain"
    elif chance < 75: return "Heavy rain"
    else:             return "Very heavy rain"

def weather_condition(temp, humidity, pressure, rain_pct):
    if rain_pct >= 70:
        return ("Thunderstorm", "storm") if temp > 30 else ("Heavy Rain", "rain")
    elif rain_pct >= 45:
        return "Rain", "rain"
    elif rain_pct >= 25:
        return "Drizzle", "drizzle"
    elif humidity >= 82:
        return "Haze", "haze"
    elif humidity >= 70:
        _, cov = cloud_coverage(humidity)
        return ("Cloudy", "cloudy") if cov >= 55 else ("Partly Cloudy", "partly_cloudy")
    elif temp >= 38:
        return "Hot & Sunny", "sunny"
    elif temp >= 30:
        return "Sunny", "sunny"
    else:
        return "Clear", "clear"

def wind_gust(wind_speed):
    return round(max(wind_speed * 1.4, wind_speed + 5), 1)

def uv_index(temp, cloud_pct):
    base = max(0, (temp - 20) / 3)
    reduction = cloud_pct / 100 * base * 0.7
    return max(0, min(11, round(base - reduction, 0)))

def weather_description(condition, temp, feels, humidity, rain_pct, wind):
    descs = {
        "Thunderstorm":  "Thunderstorms likely, stay indoors.",
        "Heavy Rain":    "Heavy rainfall expected today.",
        "Rain":          "Rainy conditions throughout the day.",
        "Drizzle":       "Light drizzle on and off.",
        "Haze":          "Hazy skies with high humidity.",
        "Cloudy":        "Overcast skies for most of the day.",
        "Partly Cloudy": "A mix of sun and cloud.",
        "Hot & Sunny":   "Very hot — stay hydrated.",
        "Sunny":         "Clear and sunny conditions.",
        "Clear":         "Clear skies throughout.",
    }
    base = descs.get(condition, "")
    return f"{base} Feels like {feels:.0f}°C, wind {wind:.0f} km/h, humidity {humidity:.0f}%."

def sunrise_sunset(city):
    defaults = {
        "Kolkata": ("05:02","18:15"), "Delhi": ("05:28","19:22"), "Mumbai": ("06:02","19:18"),
        "Chennai": ("05:58","18:32"), "Bengaluru": ("06:05","18:40"), "Hyderabad": ("05:52","18:44"),
        "Ahmedabad": ("06:07","19:30"), "Pune": ("06:04","18:47"), "Jaipur": ("05:38","19:22"),
        "Lucknow": ("05:16","19:04"), "Kanpur": ("05:18","19:03"), "Nagpur": ("05:42","18:55"),
        "Bhopal": ("05:47","19:03"), "Patna": ("05:04","18:45"), "Surat": ("06:09","19:26"),
        "Kochi": ("06:11","18:26"), "Varanasi": ("05:13","18:52"), "Mysuru": ("06:07","18:37"),
        "Goa": ("06:14","18:52"), "Visakhapatnam": ("05:36","18:21"),
    }
    return defaults.get(city, ("05:30", "18:30"))