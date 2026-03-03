# Real-Time-Lip-Movement-Detection-and-Command-Execution-in-Robotics
## Abstract
The main purpose of the project focuses on leveraging modern technology to enable human-robot or human-computer interaction. Specifically, the project utilizes Media Pipe to detect lip movements (comparing information of mean difference lips gap in the actual frame vs the last frame ). This mechanism triggers the Whisper system to convert spoken instructions into text. Subsequently, a large language model (LLM) powered by LangChain and Ollama performing the process of the text to interpret and execute commands within a simulated drone environment in Gazebo on Ubuntu 22.04 LTS

## Objectives:
### General:
To design, implement, and evaluate a multimodal human–robot interaction system that integrates computer vision, automatic speech recognition, and large language models to interpret natural language commands and execute real-time drone control within a simulated robotic environment.

### Specific:
1. To implement a facial landmark-based algorithm that detects lip movement in real time by analyzing the mean difference between upper and lower lip coordinates to trigger speech recording. <br>
2. To incorporate OpenAI Whisper for converting recorded audio signals into structured textual representations suitable for natural language processing. <br>
3. To implement a classification framework using LangChain and locally deployed LLMs (LLaMA via Ollama) that:  <br>
    - Identifies the user’s command intent (movement, landing, stop, etc.) <br>
    - Determines the directional axis of motion (X/Y/Z ±) <br>

4. To deploy large language models locally and evaluate their performance (LLaMA 2 vs LLaMA 3) in terms of inference latency and classification effectiveness under hardware constraints. <br>
5. To design and tune a PID controller capable of executing movement commands in a Gazebo-based drone simulation environment. <br>
6. To evaluate multithreading, multiprocessing, and concurrent futures approaches in order to mitigate latency introduced by speech transcription and LLM inference. <br>
7. To analyze hardware bottlenecks, ASR latency, LLM response time, and environmental biases (lighting, occlusion, noise) affecting real-time multimodal interaction. <br>


## Research Questions
1. How feasible and effective is the integration of computer vision-based lip movement detection, automatic speech recognition (Whisper), and locally deployed large language models for real-time human–robot interaction? <br>
2. What are the primary computational and architectural bottlenecks that affect real-time execution when deploying Whisper and LLaMA models locally in a robotics control loop? <br>
3. To what extent can large language models (via LangChain and Ollama) reliably classify and interpret natural language commands for structured robotic control tasks in simulated environments? <br>


## Related Work
The related work demonstrates advances in automatic speech recognition integration in robotics, LLM-based pipeline orchestration via LangChain, natural language database interfaces, and real-time computer vision using MediaPipe. While prior research explores these technologies independently—particularly in ROS environments, medical information retrieval, chatbot systems, and gesture recognition—none combine lip-based activation, ASR transcription, local LLM reasoning, and PID-based robotic control into a unified multimodal real-time interaction system.


## Pipeline structure
1. Lip Movement Detection
  MediaPipe detects facial landmarks.
  Upper and lower lip Y-coordinates are averaged.
  A mean difference threshold triggers speech recording.

2. Speech Recognition
  Audio is recorded using sounddevice.
  Whisper transcribes speech into text.
  CSV logs are generated for tracking.

3. Natural Language Classification
  LLaMA (via Ollama) classifies commands into:
    Movement
    Position recovery
    Orientation recovery
    Velocity queries
    Takeoff / Landing / Stop
  A second classification determines axis direction (X/Y/Z ±).

4. Drone Control 
  Commands are interpreted and sent to a Gazebo drone simulation.
  A manually tuned PID controller (Kp=1, Ki=0.0001, Kd=0.01) manages movement.
  Position and orientation data are logged and plotted.


## Experimental Framework
To capture video it is necessary to consider not having the following biases:
 * Viewing angle to the camera.
 * Separation distance from face to camera.
 * Strong body movements.
 * Strong movements of the face or head.
 * Covering the lips, the media-pipe system makes the prediction but is not able to detect movement with covered lips.
 * Environment with too much or too little light.
 * Ram memory saturation affects the performance of the camera.


## Latency & Bottlenecks
This project targets a real-time interaction loop (Webcam → Lip trigger → ASR → LLM → Drone action). During experimentation, the main limitation was latency, caused by CPU-intensive stages competing for resources and by Python runtime constraints.

### 1) Whisper (ASR) blocks the video loop (OpenCV freeze/lag)
#### What happens: 
When Whisper is triggered, the OpenCV stream experiences a visible interruption (frame freeze/lag) until transcription completes.  <br>

#### Why it happens:
 * Whisper transcription is computationally heavy (CPU-bound in local execution).
 * If transcription runs in the main thread (or starves it), the video capture/render loop cannot maintain its frame rate.
 * Whisper inference competes with OpenCV + MediaPipe for CPU and RAM.  <br>

#### Observed impact:
Typical transcription time: ~40–70 seconds per activation (hardware/load dependent).
In practice, ~1 minute average to produce the final text locally.  <br>

### 2) Concurrency helps responsiveness, but doesn’t make Whisper real-time
#### What was tested: 
Multithreading, multiprocessing, and concurrent.futures.

#### Why results are limited:
 * Multithreading: limited benefit for CPU-bound workloads in Python due to the GIL.
 * Multiprocessing: introduces overhead (process spawn, IPC, memory duplication), which can reduce the net gain.
 * concurrent.futures: offers the cleanest control over asynchronous tasks and improved loop responsiveness, but cannot reduce Whisper’s raw inference time.
 * Key takeaway: Even with the best concurrency approach, the system cannot exceed the limits imposed by local ASR compute cost.

### 3) Audio encoding/decoding & pipeline choices add extra overhead
#### What happens: The choice of audio format/codec and file handling affects how fast audio becomes “ready” for transcription.

#### Why it happens:
 * Some formats require additional decoding steps.
 * File I/O and conversion introduce extra latency before Whisper inference even begins.
 * Longer audio segments increase total inference time proportionally.

### 4) Local LLM inference (LLaMA via Ollama) increases response time vs cloud
#### What happens: 
LLaMA 2/3 classification takes noticeably longer than cloud-based models.

#### Why it happens:
 * The model runs locally (Ollama), so inference is limited by the host machine’s CPU/RAM (and GPU if not enabled).
 * Larger models and longer context windows increase computation.
 * The local setup trades speed for offline execution and privacy.


### 5) Resource contention across the whole multimodal loop
#### What happens: 
Even when each module works independently, the full pipeline suffers from competition for the same compute resources.

#### Why it happens:
 * OpenCV frame capture + MediaPipe landmark extraction require steady CPU cycles.
 * Whisper and LLM inference can saturate CPU/RAM and reduce the frame rate.
 * Memory pressure (RAM saturation) can cause stutter, swapping, or inconsistent timing.



## Performance of the local comparison between processing sentences
### Sentences used to measure the latency between LLama v2 and LLama v3:
1. ”I want to go to a position”.
2. ”Move the dron 2 meters to the right”.
3. ”What it’s the actual position of the Dron”
4. "How far it's the dron from the origin?"
5. "How fast the dron it moving?"
6. "How fast the dron it's rotating?"
7. "How many revolutions per second it's performing the dron"
8. "I want to put a tag in this point"
9. "Recover x,y,z"
10. "Recover Vx,Vy,Vz"
11. "Recover Wx,Wy,Wz"
12. "Where is facing the Dron?"
13. "Land"
14. "Take off the Dron"
15. "Can you stop Dron?"
16. "I don't going to use the dron know, can you please stop the dron?
17. "If the Dron is stopped, please land the dron"
18. "If the Dron has Movement, please stop the dron"
19. "Is the dron moving right now?"
20. "The Dron it's rotating?"

<p align="center">
  <img src="assets/LLAMA_EN.png" width="600"/>
</p>
<p align="center"><em>Figure 1. LLama v2 vs LLama v3 english questions latency comparison (local inference).</em></p>

