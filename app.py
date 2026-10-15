import os
import cv2
import numpy as np
import time 
from ultralytics import YOLO

model = YOLO('/home/bitpi/Desktop/PiApp/Small_Model_640_Best.onnx', task='segment')
cam_idx = 0
interval = 1.25
last_inference_time = 0
current_results = None

bin_map = {
    'Plastic-Tray': 'Recycle',
    'Styrofoam-Tray': 'Trash',
    'Utensil': 'Trash',
    'Plastic-Wrapper': 'Trash',
    'Can': 'Recycle',
    'Napkin': 'Trash',
    'Generic-Wrapper': 'Trash',
    'Condiments': 'Trash',
    'Snack-Cup': 'Recycle',
    'Water-Bottle': 'Recycle',
    'Hamburger': 'Compost',
    'Fries': 'Compost',
    'Chicken': 'Compost',
    'Waffle': 'Compost',
    'Salad': 'Compost',
    'Fruit': 'Compost',
    'Quesadilla': 'Compost',
    'Bread': 'Compost',
    'Styrofoam-Cup': 'Trash',
    'Vegetables': 'Compost',
    'Burrito': 'Compost',
    'Meat': 'Compost',
    'Cheese': 'Compost',
    'Chip-Bag': 'Recycle',
    'Spaghetti': 'Compost',
    'Pizza': 'Compost',
    'Corndog': 'Compost',
    'Human': 'Human',
    'Gravy': 'Trash',
    'Mashed Potatoes': 'Compost',
    'Lasagna': 'Compost',
    'Macaroni': 'Compost',
    'Sauce': 'Trash',
    'Tortilla': 'Compost',
    'Rice': 'Compost',
    'Carton': 'Recycle'
}

bin_colors = {
    'Compost': (0, 255, 0),
    'Recycle': (255, 100, 100),
    'Trash': (0, 0, 255),
    'Human': (0, 0, 0),
    'Needs Review': (0, 0, 0)
}

text_colors = {
    'Compost': (0, 255, 0),
    'Recycle': (255, 100, 100),
    'Trash': (0, 0, 255),
    'Human': (0, 0, 0),
    'Needs Review': (255, 255, 255)
}

ITEM_TEXT_COLOR = (255, 255, 255)

def draw_translucent_box(frame, x1, y1, x2, y2, color=(40, 40, 40), alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def Lunch_Tray_Detection():
    global last_inference_time, current_results
    
    cap = cv2.VideoCapture(cam_idx)
    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Failed to grab frame')
            break

        curr_time = time.time()

        if curr_time - last_inference_time >= interval:
            results = model.track(frame, verbose=False, conf=0.47, imgsz= 640, persist = True, tracker = 'bytetrack.yaml')
            current_results = results[0]
            last_inference_time = curr_time

        if current_results is not None and hasattr(current_results, 'boxes'):
            bin_summary = {'Trash': [], 'Recycle': [], 'Compost': []}

            for box in current_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                bin_cat = bin_map.get(class_name, 'Needs Review')

                box_color = bin_colors.get(bin_cat, (0, 190, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                if bin_cat in bin_summary:
                    bin_summary[bin_cat].append((class_name, conf))

            summary_lines = []
            for category in ['Trash', 'Recycle', 'Compost']:
                items = bin_summary[category]
                if items:
                    items_str = ', '.join([f'{item}: {conf*100:.1f}%' for item, conf in items])
                else:
                    items_str = 'None'
                summary_lines.append((category, items_str, text_colors[category]))

            start_x = 15
            start_y = 30
            
            total_text_width = 0
            for category, items_str, _ in summary_lines:
                full_line = f"{category}: {items_str}"
                (w, _), _ = cv2.getTextSize(full_line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                if w > total_text_width:
                    total_text_width = w

            padding = 10
            box_x1 = start_x - padding
            box_y1 = start_y - 20
            box_x2 = start_x + total_text_width + padding
            box_y2 = start_y + (len(summary_lines) - 1) * 25 + padding

            draw_translucent_box(frame, box_x1, box_y1, box_x2, box_y2, color=(40, 40, 40), alpha=0.6)

            y_offset = start_y
            for category, items_str, cat_color in summary_lines:
                cat_text = f"{category}: "
                cv2.putText(frame, cat_text, (start_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cat_color, 1)
                (cat_w, _), _ = cv2.getTextSize(cat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.putText(frame, items_str, (start_x + cat_w, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ITEM_TEXT_COLOR, 1)
                y_offset += 25

        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        frame_h, frame_w = frame.shape[:2]
        fps_label = f'FPS: {int(fps)}'

        (fps_w, fps_h), _ = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        fps_x = frame_w - fps_w - 15
        fps_y = fps_h + 15

        draw_translucent_box(frame, fps_x - 8, fps_y - fps_h - 8, fps_x + fps_w + 8, fps_y + 8, color=(40, 40, 40), alpha=0.6)
        cv2.putText(frame, fps_label, (fps_x, fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        exit_label = 'Press "Q" to exit program'
        (exit_w, exit_h), _ = cv2.getTextSize(exit_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        exit_x = 15
        exit_y = frame_h - 15

        draw_translucent_box(frame, exit_x - 8, exit_y - exit_h - 8, exit_x + exit_w + 8, exit_y + 8, color=(40, 40, 40), alpha=0.6)
        cv2.putText(frame, exit_label, (exit_x, exit_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)     
        
        cv2.imshow('Lunch Tray Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

Lunch_Tray_Detection()
