# Hand_Exlporation

![Exploration Demo 1](Results/gif/200step.gif)
![Exploration Demo 2](Results/gif/stage6.gif)
#### This project focuses on **tactile exploration** and **surface reconstruction** using a **dexterous robotic hand**.  


## Project Structure

```
Hand_Exploration/
├── Results/
├── Env/
│   ├── allegro_hand.py               # Allegro Hand model definitions
│   ├── Explloration_env.py            # Tactile exploration environment setup (Stage 1)
│   ├── Exploration_env_stage2.py      # Advanced exploration environment (Stage 2)
│   ├── ....
├── utils/
│   ├── GPIS.py                        # Basic GPIS modeling
│   ├── HE_GPIS.py                     # Hand Exploration-specific GPIS extensions
│   ├── Pid_Controller.py              # PID control for hand pose
│   ├── ....
├── README.md                          
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/FerryRain/Hand_Exlporation.git
   cd Hand_Exploration
   ```

2. **Create the Conda environment**:

   Create a new environment using the provided `environment.yaml`:

   ```bash
   conda env create -f environment.yaml
   conda activate hand_exploration
   ```

3. **Install Isaac Sim and Isaac Lab dependencies manually**:

   - Follow the official installation guides:

     - [Isaac Sim Installation Guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install-workstation.html)
     - [Isaac Lab Setup Instructions](https://isaac-lab.readthedocs.io/en/latest/getting_started/installation.html)

   - Typically, you need to:
     - Download and install **Isaac Sim**.
     - Set up environment variables (e.g., `source setup_env.sh` inside Isaac Sim folder).
     - Install **Isaac Lab** Python packages inside your environment:

       ```bash
       cd /path/to/isaaclab
       pip install -e .
       ```
    ⚠️ **Note:**  
    Isaac Sim and Isaac Lab require specific versions of CUDA, GPU drivers, and system dependencies.  
    Make sure your machine meets their requirements.
## Main Components

- **Env/**
  - `allegro_hand.py`: Robot hand model and kinematics definition.
  - `Explloration_env.py`: Basic tactile exploration simulation environment.
  - `Exploration_env_stage2.py`: Extended environment for Stage 2 exploration tasks, incorporating surface reconstruction and path planning.

- **utils/**
  - `GPIS.py`: Standard Gaussian Process Implicit Surface (GPIS) modeling.
  - `HE_GPIS.py`: Hand Exploration-enhanced GPIS with uncertainty-guided exploration strategies.
  - `Pid_Controller.py`: Proportional-Integral-Derivative (PID) controller for controlling hand position and orientation.

- **TEST/**
  - Reserved for test scripts and experimental trials.

## Getting Started

1. Run the simple exploration demo:

   ```bash
   conda activate hand_exploration
   ```

   ```bash
   cd Env
   python Exploration_env_stage{x}.py
   ```

---

## Experiment Stages Overview

The project progresses through **nine stages** of tactile exploration and surface reconstruction:

1. **Stage 1: Teleportation + Contact**

   * The hand is directly teleported to object surface to collect initial contact points.

2. **Stage 2: `move_to` + Uncertainty**

   * The hand moves toward regions of higher surface uncertainty using basic motion primitives.

3. **Stage 3: Surface Constraint + Smoothness**

   * Hand motion is constrained on the object surface.
   * Guided by both uncertainty gradient and contact smoothness (possibly using δ constraints).

4. **Stage 4: Global Point Sampling as AS Input**

   * Actively samples next exploration targets from global surface uncertainty map.

5. **Stage 5: Position Control Version**

   * Replaces high-level `move_to` with low-level position control for trajectory execution.

6. **Stage 6: Continuous PID Control**

   * Employs PID-based continuous control to smoothly follow desired paths along surface.

7. **Stage 7: New GPIS Initialization Function**

   * Improves initialization of the GPIS model with different spatial priors (e.g. spherical or conic).

8. **Stage 8: Hybrid Force-Position Control**

   * Combines force feedback with position control to ensure compliant, stable contact.

---

## Results Organization

Experimental data for each stage is stored in the `Data/` directory with folders named accordingly. Some examples:

```
Results/
├── stage6_1/                     # Continuous control (PID) for sphere (radius=0.5) (falied)
├── stage6_2/                     # Continuous control (PID) for sphere (radius=0.15)
├── stage6_3(CONE)/               # Continuous control (PID) for cone (radius=0.15, height=0.3)
├── stage7_1_1(sphere)/           # Stage 7 for GPIS which initialized by double bb for sphere (radius=0.15)
├── stage7_1_2/                   # Stage 7 for GPIS which initialized by double bb for sphere (radius=0.5) (success)
├── stage7_2_1/                   # Stage 7 for GPIS which initialized by double bb for cone (radius=0.15, height=0.3)
├── stage7_2_2/                   # Stage 7 for GPIS which initialized by double bb for cone (radius=0.15, height=0.3) with another type of the gaussian data preprocess 
├── stage8_1/                     # Stage 8 experiment with hybrid control for sphere
├── stage8_2_1/                   # Stage 8 experiment with hybrid control for cone

```

To visualize saved 3D reconstruction results:

```bash
cd Results
python Vis_points.py
```

---

