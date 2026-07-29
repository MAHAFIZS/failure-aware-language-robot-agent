# Isaac Sim Simulation Workflow

## Purpose

This project provides Python-driven Franka Panda simulations for manipulation-task development, validation and ML experimentation.

## Supported tasks

- Multi-object color sorting
- Multi-object tower stacking
- Single-object pick and place

## Workflow

1. Receive a natural-language instruction.
2. Route the instruction to a supported robot skill.
3. Launch NVIDIA Isaac Sim.
4. Construct the Franka, objects and environment.
5. Execute the manipulation sequence.
6. Validate object completion and final placement.
7. Export a normalized ContactTrace JSON result.

## Simulation validation

Current checks include:

- number of requested and completed objects;
- final object positions;
- XY placement error;
- success threshold;
- simulation steps;
- task duration;
- constraint-violation count.

## Physics parameters

The scene currently uses:

- stage units: 1 metre;
- gravity: -9.81 m/s²;
- cube dimensions: 51.5 mm.

Mass, friction and restitution currently use Isaac Sim defaults. Future validation will compare these parameters against real-object measurements.

## Assets

- Franka Panda USD from the NVIDIA Isaac Sim asset library
- Procedural cube and environment primitives
- Simulation configurations stored under `configs/`
- Result artifacts stored under `results/`

## Reproducibility

Each task accepts:

- instruction;
- random seed;
- output JSON path.

Scripts and configurations are intended to be stored under version control.
