

from pathlib import Path
import cv2
import numpy as np

images_path = Path(__file__).parent / "dataset"
images = [item for item in images_path.iterdir() if item.is_file()]

def match(image, template, scales = np.arange(0.4, 1.7,0.1), threshol=0.8):
    matches = []
    for scale in scales:
        resized_template = cv2.resize(template, 
                                      (int(template.shape[0]*scale), 
                                       int(template.shape[1]*scale)))
        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshol)
        for pt in zip(*loc[::-1]):
            matches.append({"top_left": pt,
                            "bottom_right": (pt[0] + int(template.shape[1]*scale),
                                             pt[1]+int(template.shape[0]*scale)),
                            "confidence": result[pt[1],pt[0]],
                            "scale":scale
                            })
    return matches

image = cv2.imread(images[0])

cv2.namedWindow("Image",cv2.WINDOW_GUI_NORMAL)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
x, y, w, h = cv2.selectROI("Template",gray)
template = gray[y:y+h, x:x+w]

matches = match(gray, template)

for match in matches:
    cv2.rectangle(image, match["top_left"], match["bottom_right"],(0,255,0),1)
    cv2.putText(image, f"{match["confidence"]:.2f}",match["top_left"], cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

print(matches)
cv2.imshow("Image",image)
cv2.waitKey()
