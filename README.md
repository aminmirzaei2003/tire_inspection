# Tire Inspection Action Detection and Time Measurement

This repository presents a computer vision pipeline for **detecting tire inspection actions performed by workers**, **counting inspected tires**, and **measuring the inspection time for each tire** in industrial environments.

The system integrates object tracking, human pose estimation, and depth-based spatial reasoning to robustly infer inspection events from video data.

---

## 🚀 Project Overview

Manual tire inspection is a critical quality control step in manufacturing. This project aims to automatically:

- Detect when a worker is inspecting a tire  
- Count the number of inspected tires  
- Measure the inspection duration for each tire  

The proposed pipeline combines:
- **Visual tracking** to follow tires across frames  
- **Human pose estimation** to identify worker actions  
- **Depth estimation** to reason about spatial proximity between workers and tires  

---

## 🧠 Methodology

The pipeline consists of the following main components:

### 1. Tire Detection and Tracking
- Tires are detected and tracked using the **SuperGlue** feature matching model.
- Persistent tracking IDs allow each tire to be uniquely identified across frames.
- This enables accurate tire counting and temporal association.

### 2. Worker Pose Estimation
- Worker body keypoints are extracted using a **YOLO Pose** model.
- Upper-body joints (e.g., hands, torso) are used to infer inspection-related motion patterns.

### 3. Spatial Distance Measurement
- **Depth Anything v2** is used to estimate depth maps from monocular video.
- 3D spatial distances between worker keypoints and tire locations are computed.
- Inspection is inferred when the worker remains within a defined spatial threshold of a tire.

### 4. Inspection Time Estimation
- For each tracked tire:
  - Inspection start time is detected when spatial and pose conditions are met.
  - Inspection end time is detected when the worker moves away.
- Total inspection duration is computed per tire.

---

## 🏗️ Pipeline Overview

