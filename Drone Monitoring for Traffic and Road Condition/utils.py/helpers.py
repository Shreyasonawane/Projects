import cv2  # Import OpenCV for image processing

# Function to draw bounding boxes and tracking IDs on the video frame
def draw_boxes(frame, box, track_id):
    # Unpack the bounding box coordinates
    x, y, w, h = box

    # Draw a green rectangle around the object
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Put the object ID above the bounding box
    cv2.putText(frame, f"ID: {track_id}", (x, y - 10),          # Position the text above the rectangle
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,                  # Font and size
                (0, 255, 0), 2)                                 # Green text color with thickness 2
