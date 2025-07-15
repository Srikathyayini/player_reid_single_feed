import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
def load_model():
    # Replace 'path_to_downloaded_model.pt' with the actual path to the downloaded .pt file
    print("Loading Ultralytics YOLOv11 model...")
    model = YOLO('best.pt')  # Place the model file here
    return model

# Detect players using the model
def detect_players(frame, model):
    # Run inference on the frame
    results = model(frame)
    detections = []
    for result in results:
        for box in result.boxes:
            x, y, w, h = map(int, box.xywh[0])  # Extract center x, y, width, height
            detections.append({'x': x - w//2, 'y': y - h//2, 'w': w, 'h': h})
    return detections

# Assign and maintain player IDs
def assign_ids(detections, player_ids, frame_count):
    if frame_count < 60:  
        for det in detections:
            key = (det['x'], det['y'], det['w'], det['h'])  
            if key not in player_ids:
                player_ids[key] = len(player_ids) + 1
    else:
        for det in detections:
            key = (det['x'], det['y'], det['w'], det['h'])  
            if key not in player_ids:
                min_dist = float('inf')
                closest_key = None
                for old_key in player_ids:
                    old_det = {'x': old_key[0], 'y': old_key[1], 'w': old_key[2], 'h': old_key[3]}
                    dist = ((det['x'] - old_det['x'])**2 + (det['y'] - old_det['y'])**2)**0.5
                    if dist < min_dist and dist < 100:  
                        min_dist = dist
                        closest_key = old_key
                if closest_key:
                    player_ids[key] = player_ids[closest_key]
                else:
                    player_ids[key] = len(player_ids) + 1
    return player_ids


def draw_detections(frame, detections, player_ids):
    for det in detections:
        x, y, w, h = det['x'], det['y'], det['w'], det['h']
        key = (x, y, w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(player_ids.get(key, 'N/A')), (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


def main():
    cap = cv2.VideoCapture('15sec_input_720p.mp4')  
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    model = load_model()  
    player_ids = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_players(frame, model)  
        player_ids = assign_ids(detections, player_ids, frame_count)
        frame = draw_detections(frame, detections, player_ids)

        cv2.imshow('Player Re-Identification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()