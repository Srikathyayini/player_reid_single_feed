# Player Re-identification in Sports Footage Report

## Your Approach and Methodology
This project addresses the "Option 2: Re-identification in a Single Feed" task, aiming to identify players in a 15-second video clip and maintain consistent IDs when they re-enter the frame. The solution leverages a fine-tuned Ultralytics YOLOv11 model for player detection, integrated with OpenCV for video processing. The methodology involves:
- Initial ID assignment based on detections in the first 2 seconds (60 frames at 30 FPS).
- Re-identification using spatial proximity (distance < 100 pixels) for players re-entering the frame.
- Real-time simulation with frame-by-frame processing, saving outputs to an "outputs" folder for review.

## Techniques You Tried and Their Outcomes
- **Distance-Based Re-identification**: Implemented a 100-pixel threshold to match re-entering players to their original IDs. This approach successfully maintained consistency across the video, with minimal ID swaps observed during testing.
- **Frame Saving**: Added functionality to save each processed frame as a JPEG in the "outputs" folder, enabling post-analysis. This resulted in 450 frames for a 15-second clip, confirming the pipeline's operation.
- **Model Integration**: Used the `ultralytics` library to load the provided YOLOv11 model, adapting its output to fit the ID assignment logic.

## Challenges Encountered
- **TypeError Resolution**: Encountered a `TypeError: unhashable type: 'dict'` when using dictionaries as keys in the `player_ids` dictionary. Resolved by switching to tuples of `(x, y, w, h)` as hashable keys.
- **Model Loading Issues**: Initial errors due to incorrect file paths were fixed by ensuring the `model.pt` file was correctly placed in the project directory.
- **Performance Limitations**: The current implementation processes frames sequentially, leading to noticeable latency. Real-time performance was not fully achieved due to frame saving overhead.

## If Incomplete, Describe What Remains and How You Would Proceed with More Time/Resources
The solution is functional but incomplete in terms of real-time optimization and full accuracy. With more time and resources:
- **Optimization**: Implement multi-threading or GPU acceleration to improve frame rate and reduce latency.
- **Accuracy Enhancement**: Tune the re-identification threshold and incorporate temporal features (e.g., motion vectors) to improve ID consistency.
- **Additional Features**: Add ball tracking and cross-frame validation to enhance the system's robustness, pending access to the full model capabilities.

## Conclusion
This submission provides a working prototype for player re-identification, meeting the assignment's core objectives. The code is documented in `README.md`, and outputs are saved for evaluation. Further development could address performance and accuracy, aligning with real-world constraints.

## Submission Details
- **Author**: [Your Name]
- **Date**: July 15, 2025
- **Submitted to**: `arshdeep@liat.ai` and `rishit@liat.ai`
- **Repository Link**: [Insert GitHub or Google Drive link here]
