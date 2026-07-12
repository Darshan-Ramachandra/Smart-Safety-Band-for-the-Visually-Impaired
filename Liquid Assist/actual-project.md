For the liquid detection and fill-level estimation module, describe a wearable edge-AI assistive system for visually impaired users that performs real-time liquid identification and container fill-level estimation using on-device computer vision and deep learning.

The system should use a wrist-mounted Raspberry Pi camera with event-driven image capture to reduce computational overhead and improve power efficiency.

The liquid analysis pipeline should include:

* Object detection for identifying containers such as cups and bottles
* Region of Interest (ROI) extraction for isolating the container area
* A multi-task deep learning model for simultaneous beverage classification and fill-level estimation
* Offline Text-to-Speech audio feedback generation

Mention that YOLOv5-Nano is used for lightweight object detection on Raspberry Pi edge hardware, while MobileNetV2 is used as the shared backbone for multi-task learning because of its computational efficiency and suitability for embedded inference.

The system should classify beverages such as water, milk, tea, soft drinks, and cooking oil while estimating fill levels into discrete categories:

* Low
* Medium
* High

State that the dataset does not contain manually annotated fill-level labels; therefore, pseudo-labeling is performed using classical computer vision techniques including:

* Edge detection
* Boundary detection
* Contour analysis
* Geometric liquid-height estimation

Explain that relative liquid-height ratios are computed and converted into ordinal fill-level categories for supervised training.

Include the rationale for using multi-task learning:

* Shared feature extraction
* Reduced model size
* Lower inference latency
* Improved generalization
* Efficient deployment on embedded edge devices

Mention major technical challenges handled by the system:

* Transparent liquids
* Reflections
* Lighting variation
* Container shape variability
* Noise sensitivity during boundary estimation

Specify that all inference is executed locally on Raspberry Pi 4 using optimized edge-AI inference pipelines such as TensorFlow Lite without cloud dependency to ensure:

* Low latency
* Privacy preservation
* Offline usability
* Real-time assistive feedback

The conceptual workflow should follow:
Image capture → preprocessing → YOLO-based container detection → ROI extraction → MobileNetV2 beverage and fill-level prediction → response generation → Text-to-Speech audio output.
