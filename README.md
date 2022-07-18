# vehicle-detection
Project to detect types of vehicles to pass a stationary camera, using computer vision.

## Folder Structure
Contains the following detectors/ trackers:

    - simple_detector: Very compact but innefective detector which uses a cascade classifier to detect cars.
                       Detected cars are highlighted by a rectangle in the out_vid. No tracking capability.

    - yolov5_detector: More advanced detector, utilising yolov5 to detect various object classes.
                       Detected objects are highlighted by a rectangle, with type label in out_vid. Still no tracking capability.

    - yolov5_strongsort: Object tracker utilising both yolo5 and strong_sort libraries to give advanced tracking capabilities
                         Detected objects are highlighted by rectangle in out_vid along with confidence, type and id labels.
                         Will develop further to complete initial challenge 4 assignement.