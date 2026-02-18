

from pathlib import Path
import cv2
import numpy as np

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0+cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, -6)

def overlap(box1, box2, threshold):
    x1_min, y1_min = box1["top_left"]
    x1_max, y1_max = box1["bottom_right"]
    x2_min, y2_min = box2["top_left"]
    x2_max, y2_max = box2["bottom_right"]

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return False
    
    inter_area = (inter_x_max-inter_x_min)*(inter_y_max-inter_y_min)
    box1_area = (x1_max - x1_min)*(y1_max-y1_min)
    box2_area = (x2_max-x2_min)*(y2_max-y2_min)

    iou = inter_area/(box1_area + box2_area - inter_area)
    return iou > threshold

def non_max_supression(boxes, overlap_threshold=0.3):
    if len(boxes)==0:
        return []
    es = sorted(boxes, key=lambda item: item["confidence"], reverse=True)

    picked = []

    while boxes:
        current = boxes.pop(0)
        picked.append(current)
        boxes = [box for box in boxes if not overlap(current, box, overlap_threshold)]
    
    return picked

def match(image, template, scales=np.arange(0.4, 2.5, 0.05), threshold = 0.8):
    matches = []

    for scale in scales:
        reszes_template = cv2.resize(template, 
                                     (int(template.shape[0] * scale), 
                                      int(template.shape[1]*scale)))
        result = cv2.matchTemplate(image, reszes_template, cv2.TM_CCOEFF_NORMED)

        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            matches.append({
                "top_left": pt,
                "bottom_right": (pt[0]+int(template.shape[1]*scale), pt[1]+int(template.shape[0]*scale)),
                "confidence": result[pt[1], pt[0]],
                "scale": scale
            })
    return matches
    
template = None
while capture.isOpened():
    ret, frame = capture.read()    
    key = chr(cv2.waitKey(1) & 0xFF)

    if key == 'q':
        break

    elif key == 't':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.selectROI("ROI", gray)
        x, y, w, h = roi
        cv2.destroyWindow("ROI")
        template = gray[y:y+h, x:x+w]

    if template is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Template", template)
        matches = match(gray, template)
        print(matches)
        matches = non_max_supression(matches)

        for m in matches:
            cv2.rectangle(frame, m["top_left"], m["bottom_right"], (0, 255, 0), 1)
            cv2.putText(frame, f"{m['confidence']:.2f}", m["top_left"], cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))

    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()
