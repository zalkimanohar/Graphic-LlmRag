import re

class AutoLabeler:
    """
    Parses natural language prompts to auto-detect target variables 
    and configure machine learning tasks for the Gold star schema.
    """
    def __init__(self):
        # Expanded intent mapping covering wins, podiums, points, regression, anomaly, and forecasts
        self.intent_mapping = {
            "maximum score": {"target": "points", "task": "aggregation", "metric": "sum"},
            "most points": {"target": "points", "task": "aggregation", "metric": "sum"},
            "score": {"target": "points", "task": "aggregation", "metric": "sum"},
            "standings": {"target": "points", "task": "aggregation", "metric": "sum"},
            "win": {"target": "is_win", "task": "classification", "metric": "logloss"},
            "winner": {"target": "is_win", "task": "classification", "metric": "logloss"},
            "podium": {"target": "is_podium", "task": "classification", "metric": "logloss"},
            "points": {"target": "has_points", "task": "classification", "metric": "logloss"},
            "position": {"target": "final_position", "task": "regression", "metric": "rmse"},
            "anomaly": {"target": "status_anomaly", "task": "anomaly_detection", "metric": "anomaly_score"},
            "crash": {"target": "status_anomaly", "task": "anomaly_detection", "metric": "anomaly_score"},
            "forecast": {"target": "points", "task": "temporal_forecast", "metric": "rmse"}
        }

    def detect_target_from_prompt(self, prompt: str) -> dict:
        prompt_lower = prompt.lower()
        
        # Default fallback target
        detected = {"target": "is_podium", "task": "classification", "metric": "logloss"}
        
        # Check longer/specific keyword phrases first, then single words
        sorted_keywords = sorted(self.intent_mapping.keys(), key=lambda x: len(x), reverse=True)
        
        for keyword in sorted_keywords:
            if keyword in prompt_lower:
                config = self.intent_mapping[keyword]
                detected["target"] = config["target"]
                detected["task"] = config["task"]
                detected["metric"] = config["metric"]
                break
                
        print(f"[AutoLabeler] Prompt parsed successfully -> Target: {detected['target']} | Task: {detected['task']} | Metric: {detected['metric']}")
        return detected