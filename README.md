# Failure-Aware Natural-Language Robot Agent

A persistent natural-language robot manipulation system built with NVIDIA Isaac Sim, Franka, Streamlit, and a local Qwen3 model through Ollama.

## Features

- Natural-language robot commands
- Persistent Isaac Sim scene
- Pick, hold, release, and place actions
- Cube-on-cube stacking
- Scene-graph-based blocker detection
- Automatic blocker removal
- Occupancy-aware rearrangement
- Procedural unseen letter generation
- Procedural digit generation
- Workspace and spacing validation
- Final placement-error verification

## Example Commands

```text
Pick and hold the white cube.
Place the white cube on the red cube.
Pick the cube under the blue cube.
Make a capital letter R.
Make the number seven.
System Pipeline
Natural-language instruction
→ intent recognition
→ deterministic or geometric planning
→ scene-state analysis
→ blocker and occupancy handling
→ Isaac Sim execution
→ placement verification

Clear letter and digit commands are converted procedurally from rendered font geometry into eight validated cube target positions. No fixed A–Z or 0–9 robot-coordinate templates are used.

Technology
Python
NVIDIA Isaac Sim
Franka Panda
Ollama
Qwen3
Streamlit
NumPy
Pillow
Scene-graph reasoning
Project Status

This is a research prototype. Current work focuses on execution reliability, visual task-quality validation, failure diagnosis, and automatic recovery.

Demo

Add the LinkedIn or YouTube demo-video link here.

Author

M A Hafiz
M.Sc. Medical Engineering, FAU Erlangen-Nürnberg
