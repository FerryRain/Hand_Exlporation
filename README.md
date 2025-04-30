# Hand_Exlporation
#### This project focuses on **tactile exploration** and **surface reconstruction** using a **dexterous robotic hand**.  


## Project Structure

```
Hand_Exploration/
├── Env/
│   ├── allegro_hand.py               # Allegro Hand model definitions
│   ├── Explloration_env.py            # Tactile exploration environment setup (Stage 1)
│   ├── Exploration_env_stage2.py      # Advanced exploration environment (Stage 2)
├── TEST/                              # (Some testing files)
├── utils/
│   ├── GPIS.py                        # Basic GPIS modeling
│   ├── HE_GPIS.py                     # Hand Exploration-specific GPIS extensions
│   ├── Pid_Controller.py              # PID control for hand pose
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
   python Exploration_env_stage2.py
   ```

