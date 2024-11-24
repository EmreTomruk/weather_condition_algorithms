class RuleBasedWeatherForecast:

    def predict(self, temperature, humidity):
        if temperature > 30 and humidity < 50:
            return "Sunny"
        elif temperature <= 30 and humidity >= 70:
            return "Rainy"
        elif 20 <= temperature <= 30 and 50 <= humidity < 70:
            return "Cloudy"
        else:
            return "Uncertain"