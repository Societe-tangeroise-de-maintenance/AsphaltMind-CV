
class Settings():
    weapons_detection_thresh = 0.4
    person_detection_thresh = 0.4
    threat_detection_thresh = 0.5
    pothole_detection_thresh = 0.1

    # Path to the YOLO weights and model configuration
    threat_detection_weight = "src/weights/threat_detection.h5"
    person_detection_weight = "src/weights/yolov8s.pt"
    weapon_detection_weight = "src/weights/weapons_100_s.pt"
    pose_estimation_weight = "src/weights/yolov8s-pose.pt"
    pothole_detection_weight = "src/weights/pothole.pt"  # Replace with your pothole model path
    person_bbox_margin = 0.2
    server_api_token = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbkBhZG1pbi5jb20iLCJleHAiOjE3MzczMDc0MjV9.DWha_BbA5OUxX4ozDkZJPDNJwGfMG7mrZyRisKrG99g"
    server_url_incidents = "http://192.168.0.61:8000/incidents/incidents/"
    server_url_users = "http://192.168.0.61:8000/users/users/"


settings = Settings()