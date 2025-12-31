import cv2
from matplotlib import colormaps


color_map = lambda x: [int(d*255) for d in colormaps['Set3'](x)[:3]]


def fast_plot(frame, yolo_boxes, names_map):

    src_frame = frame.copy()
    
    for box in yolo_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls_idx = int(box.cls[0])

        label = f"{names_map[cls_idx]}: {conf:.2f}"
        color = color_map(cls_idx / len(names_map))

        # plot with fill and alpha
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)
        
    # Blend original frame with the colored boxes
    alpha = 0.4
    frame = cv2.addWeighted(frame, alpha, src_frame, 1 - alpha, 0)

    return frame