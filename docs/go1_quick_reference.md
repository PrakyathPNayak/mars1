# Go1 Quick Reference Guide for Researchers
## Comparison with Go2, Spot, and ANYmal C

---

## 1. GO1 VS GO2 - QUICK COMPARISON

### When to Choose Go1:
✅ **Budget-conscious research** ($3,000-$5,000)
✅ **RL/learning research** (framework is more accessible)
✅ **Educational use** (good balance of capability and cost)
✅ **Indoor research environments** (lighter, more portable)
✅ **Sim-to-Real transfer studies** (good success rate: ~80-90%)

### When to Choose Go2:
✅ **Better performance needed** (higher torque: 45 N·m vs 35 N·m)
✅ **Outdoor research** (better sensors, weatherproofing)
✅ **Larger payload experiments** (12kg vs 3-5kg)
✅ **LiDAR-based navigation** (4D LiDAR standard on Go2)
✅ **Longer runtime operations** (2-4 hours vs 1-2.5 hours)

### Side-by-Side Specs:
```
Parameter               Go1              Go2         Winner for Research
─────────────────────────────────────────────────────────────────────
Weight                  12kg             15kg        Go1 (portable)
Max Speed              4.7 m/s          5.0 m/s     Go2 (better perf)
Max Torque             35.55 N·m        45 N·m      Go2 (more power)
Payload                3-5 kg           12 kg       Go2 (flexibility)
Battery Life           1-2.5 hrs        1-2 hrs     Go1 (slightly longer)
LiDAR                  Optional         Standard    Go2 (SLAM-ready)
4D LiDAR L1            No               Yes         Go2 (better perception)
Cost                   ~$3-5k           ~$8-12k     Go1 (affordable)
Learning Curve         Moderate         Moderate    Tie
Community Support      Excellent        Good        Go1 (more examples)
```

---

## 2. GO1 TECHNICAL SPECIFICATIONS SUMMARY

### Physical Specifications
```
Weight (with battery):     12 kg
Weight (without battery):  ~10.5 kg
Standing dimensions:       58-65cm (L) × 28cm (W) × 22-40cm (H)
Folded size:              54 × 29 × 13 cm
Payload capacity:         3-5 kg (varies with gait)
Ground clearance:         ~5-7 cm
```

### Motor & Actuator Specifications
```
Total Joints:             12 (3 per leg)
Actuator Type:            Electric servo motors (brushless)
Thigh Motor Torque:       23.7 N·m (instantaneous)
Knee Motor Torque:        35.55 N·m (instantaneous)
Motor Weight:             ~0.8 kg per motor
Control Interface:        CAN bus (1 Mbps)
Position Encoder:         Magnetic absolute encoder (12-bit)
```

### Motion Capabilities
```
Max Speed (trot):         3.5-4.7 m/s (12.6-17 km/h)
Max Acceleration:         ~4 m/s² (est.)
Max Climb Angle:          35°
Stair Climbing:           ~8-10cm step height
Turning Radius:           ~0.5m at full speed
Stride Length:            ~0.35m (varies with speed)
```

### Battery & Power System
```
Battery Type:             Li-ion (4S2P configuration)
Capacity:                 6000 mAh
Nominal Voltage:          14.8V (per cell: 3.7V)
System Voltage:           25.2V nominal
Energy:                   ~172.8 Wh
Operating Time:           1-2.5 hours (varies with gait, terrain)
Charge Time:              ~2-3 hours (with standard charger)
Operating Temp:           -10°C to 50°C
Charge Temp:              0°C to 45°C
```

### Sensing Suite
```
IMU:                      9 DOF (3-axis accelerometer, gyroscope, magnetometer)
                          Update rate: 100 Hz
Cameras:                  5 fisheye stereo depth cameras
                          - 3 pairs for obstacle detection
                          - Resolution: varies by variant
                          - FOV: ~150° per camera
Ultrasonic Sensors:       3 sets (front, sides)
                          - Range: 10cm - 2m
                          - Update rate: 50 Hz
Foot Pressure:            Not standard (optional via payload)
```

### Control Interface
```
Onboard Compute:          3× Nano processors (Jetson)
                          - BCM2711 (ARM controller)
                          - 3× NVIDIA Jetson Nano
Communication:            Ethernet, WiFi (802.11ac)
                          Wireless range: ~30-50m
SDK:                      C++, Python (unitree_legged_sdk)
Control Frequency:        500-1000 Hz
Latency:                  ~10-20ms (wireless)
```

---

## 3. GO1 TRAINING QUICK START

### Hardware Requirements
```
GPU:                      NVIDIA RTX 3090, A100, or equivalent (10GB+ VRAM)
CPU:                      Intel i7/i9 or AMD Ryzen 9 (8+ cores)
RAM:                      64GB recommended
Storage:                  500GB SSD (for datasets + checkpoints)
Training Time:            2-4 hours for basic locomotion
```

### Software Stack
```
OS:                       Ubuntu 20.04 or 22.04
Python:                   3.8-3.10
PyTorch:                  1.10+
Isaac Gym:                2023.04 or later
CUDA:                     11.8 or 12.x
cuDNN:                    8.6+
ROS:                      ROS2 Humble (optional, for hardware)
```

### Basic Training Pipeline
```
Step 1: Environment Setup
├── Install Isaac Gym
├── Create robot model (provided)
├── Define observation/action space
└── Set up physics parameters

Step 2: Reward Design
├── Define velocity tracking reward
├── Add energy efficiency penalty
├── Include stability constraints
└── Tune reward weights

Step 3: Training
├── Initialize policy network (256-512 neurons)
├── Run PPO algorithm
├── Monitor loss curves
└── Save best checkpoint (~2-4 hours)

Step 4: Evaluation
├── Test in diverse simulated terrains
├── Apply domain randomization validation
├── Check robustness metrics
└── Deploy to hardware

Step 5: Hardware Transfer
├── Update controller gains if needed
├── Validate contact detection
├── Start with slow speeds
└── Gradually increase speed commands
```

### Key Hyperparameters for Go1
```python
# PPO Configuration
learning_rate = 5e-5
batch_size = 32
num_epochs = 4
gamma = 0.99  # discount factor
gae_lambda = 0.95

# Network Architecture
actor_hidden_sizes = [256, 256]
critic_hidden_sizes = [256, 256]
activation = "relu"

# Training
num_envs = 4096  # parallel environments
num_steps_per_episode = 24  # 0.24 seconds of simulation
total_steps = 10_000_000  # ~2-3 hours training

# Domain Randomization
mass_randomization = 0.3  # ±30%
friction_randomization = [0.2, 1.2]
motor_delay = 0.05  # seconds
sensor_noise = 0.05  # ±5%
```

---

## 4. GO1 DEPLOYMENT CHECKLIST

### Pre-Deployment Validation
```
☐ Simulation Success
  ├── Episode length > 500 steps
  ├── Cumulative reward > threshold
  ├── Smooth velocity tracking
  └── Low energy consumption

☐ Robustness Testing
  ├── ±30% mass randomization
  ├── ±50% friction variation
  ├── Sensor noise injection
  └── Motor delay simulation

☐ Transfer Validation
  ├── Test on unseen terrains (in sim)
  ├── Verify gait patterns
  ├── Check stability margins
  └── Confirm foot contact timing

☐ Hardware Preparation
  ├── Charge batteries to 100%
  ├── Calibrate IMU
  ├── Update firmware to latest
  ├── Test wireless connection
  └── Verify SDK installation
```

### First Run on Hardware
```
☐ Initial Checks
  ├── Check motor ranges (full rotation)
  ├── Verify foot sensor readings
  ├── Confirm WiFi connection (stable)
  └── Monitor CPU/GPU usage

☐ Slow Speed Testing
  ├── Start with 0.5 m/s max speed
  ├── Test on flat surface only
  ├── Record IMU data for analysis
  ├── Check for jerky motions
  └── Monitor joint temperatures

☐ Speed Increase
  ├── 1 m/s on flat (5-10 trials)
  ├── 2 m/s on flat (5-10 trials)
  ├── 3 m/s on flat (5-10 trials)
  └── Full speed on test course

☐ Terrain Testing
  ├── Slight slope (10°)
  ├── Stairs (5-8 cm steps)
  ├── Obstacles (5 cm height)
  └── Mixed terrain
```

---

## 5. COMPARATIVE ADVANTAGES & DISADVANTAGES

### Go1 Advantages
| Aspect | Advantage | Details |
|--------|-----------|---------|
| **Cost** | Very Affordable | $3-5k vs $30-50k for competitors |
| **Size** | Portable | 12 kg, fits in backpack, easy transport |
| **Speed** | Fast for class | 4.7 m/s competitive performance |
| **Learning** | Research-Friendly | Active community, many training examples |
| **Customization** | Flexible | Open SDK, modular design, extensible |
| **Training** | Fast Transfer | 80-90% sim-to-real success without fine-tuning |

### Go1 Disadvantages
| Aspect | Disadvantage | Details |
|--------|-------------|---------|
| **Payload** | Limited | Only 3-5 kg vs 12-14 kg on competitors |
| **Sensors** | Fewer Standard | LiDAR is optional, no thermal camera |
| **Robustness** | Not IP-Rated | Better for indoor research, not industrial |
| **Durability** | Less Rugged | Not designed for heavy abuse like Spot/ANYmal |
| **Weather** | Limited | Not suitable for harsh environments |
| **Actuators** | Standard Motors | Non-compliant, no force feedback |

### Go1 vs Competitors Summary
```
╔════════════════════╦═════════╦═════════╦═════════════╦══════════════╗
║ Category           ║  Go1    ║  Go2    ║    Spot     ║   ANYmal C   ║
╠════════════════════╬═════════╬═════════╬═════════════╬══════════════╣
║ Price (USD)        ║ 3-5k    ║ 8-12k   ║  100-150k   ║  100-150k    ║
║ Portability        ║ ★★★★★   ║ ★★★★   ║  ★★        ║  ★★         ║
║ Max Payload        ║ 3-5 kg  ║ 12 kg   ║  14 kg      ║  10 kg       ║
║ Outdoor Capable    ║ ★★      ║ ★★★    ║  ★★★★★     ║  ★★★★★      ║
║ Research Friendly  ║ ★★★★★   ║ ★★★★   ║  ★★★       ║  ★★★        ║
║ Learning Support   ║ Excellent║ Good    ║  Limited    ║  Limited     ║
║ Field Proven       ║ Emerging ║ Emerging║  Proven     ║  Proven      ║
║ Battery Life       ║ 1-2.5h  ║ 1-2h    ║  1.5-3h     ║  2-4h        ║
╚════════════════════╩═════════╩═════════╩═════════════╩══════════════╝
```

---

## 6. RESEARCH PROJECT IDEAS FOR GO1

### Beginner Projects (1-2 weeks)
1. **Basic Locomotion Learning**
   - Train PPO policy for trotting
   - Test on varied speeds
   - Analyze gait patterns

2. **Terrain Classification**
   - Classify terrain from IMU signals
   - Learn to adapt gait
   - Implement terrain-specific controllers

3. **Obstacle Avoidance**
   - Use camera input for obstacle detection
   - Train reactive navigation policy
   - Test in lab environment

### Intermediate Projects (3-6 weeks)
1. **Sim-to-Real Transfer Study**
   - Compare different domain randomization strategies
   - Measure transfer accuracy
   - Optimize for minimal fine-tuning

2. **Multi-Gait Learning**
   - Train single policy for multiple gaits
   - Analyze gait switching mechanisms
   - Test on different terrains

3. **Vision-Based Navigation**
   - Integrate camera input into control policy
   - Train for autonomous waypoint navigation
   - Test in cluttered environments

### Advanced Projects (6+ weeks)
1. **Meta-Learning for Robot Adaptation**
   - Implement MetaLoco approach
   - Train universal policy
   - Test on unknown robot variants

2. **Fault-Tolerant Locomotion**
   - Train policies for joint failure scenarios
   - Implement automatic gait switching
   - Validate on real hardware

3. **Bipedal Locomotion**
   - Implement TumblerNet controller
   - Train for two-leg walking
   - Test on stairs and uneven terrain

4. **Multi-Robot Coordination**
   - Train multiple Go1s to work together
   - Implement formation control
   - Study emergent behaviors

---

## 7. COMMON ISSUES & SOLUTIONS

### Training Issues
```
Problem: Policy not converging
├── Solution 1: Reduce learning rate (5e-5 → 1e-5)
├── Solution 2: Increase reward weight for primary objective
├── Solution 3: Check for NaN values in observations
└── Solution 4: Verify simulation physics parameters

Problem: Large sim-to-real gap
├── Solution 1: Increase domain randomization ranges
├── Solution 2: Add more terrain variety
├── Solution 3: Validate against real hardware early
└── Solution 4: Use motion imitation as auxiliary task

Problem: High energy consumption
├── Solution 1: Increase energy penalty in reward
├── Solution 2: Reduce maximum speed requirement
├── Solution 3: Tune PID gains on hardware
└── Solution 4: Use momentum-based optimization
```

### Hardware Issues
```
Problem: Robot not responding to commands
├── Solution: Check WiFi connection (ping latency < 50ms)
├── Solution: Verify battery level (>30%)
├── Solution: Update controller firmware
└── Solution: Check SDK compatibility

Problem: Shaky/unstable motion
├── Solution: Reduce control frequency (1000 → 500 Hz)
├── Solution: Increase damping in controller gains
├── Solution: Check for loose mechanical parts
└── Solution: Validate motor calibration

Problem: Battery drains quickly
├── Solution: Profile motor currents
├── Solution: Reduce speed commands
├── Solution: Check for motor encoder issues
└── Solution: Replace battery if > 2 years old
```

---

## 8. RECOMMENDED RESOURCES

### Official Documentation
- **Unitree Go1 Datasheet**: Technical specifications
- **Unitree SDK GitHub**: `unitree_legged_sdk`
- **ROS Support**: `go1_gazebo` simulation package

### Research Papers
1. "Walk These Ways" - PPO training + Domain Randomization
2. "MetaLoco" - Universal policies across morphologies
3. "TumblerNet" - Bipedal locomotion controller
4. "AcL" - Fault-tolerant gait learning

### Community Resources
- **Unitree Forums**: Official user community
- **GitHub Issues**: SDK problems and solutions
- **Research Repositories**: University implementations
- **YouTube Tutorials**: Hardware setup and programming

### Simulation Tools
- **Isaac Gym**: GPU-accelerated physics (recommended)
- **PyBullet**: Free alternative (slower)
- **Gazebo + ROS**: ROS integration (heavier but more flexible)
- **MuJoCo**: Contact-rich simulation (good for force analysis)

---

## 9. QUICK REFERENCE: PARAMETER VALUES

### Motor Control Values
```
Recommended PD Gains (Hardware):
- Hip   (P): 30-50 N·m/rad
- Hip   (D): 0.8-1.5 N·m·s/rad
- Thigh (P): 25-40 N·m/rad
- Thigh (D): 0.6-1.2 N·m·s/rad
- Knee  (P): 25-40 N·m/rad
- Knee  (D): 0.6-1.2 N·m·s/rad

Max Joint Torques:
- Hip:   ±55 N·m (absolute max)
- Thigh: ±55 N·m (absolute max)
- Knee:  ±55 N·m (absolute max)

Recommended Limits:
- Hip:   ±40 N·m (safe operating)
- Thigh: ±40 N·m (safe operating)
- Knee:  ±45 N·m (safe operating)
```

### Speed Ranges
```
Velocity Commands:
- Slow:   0.5-1.0 m/s (stable, for fine control)
- Normal: 1.0-2.5 m/s (standard operation)
- Fast:   2.5-4.0 m/s (high performance)
- Sprint: 4.0-4.7 m/s (maximum, short duration)

Angular Velocity:
- Yaw rate: ±1.5 rad/s (typical)
- Roll: ±20° (maximum safe)
- Pitch: ±15° (maximum safe)
```

### Power Consumption
```
Idle (standing):        ~30-50 W
Slow walking (1 m/s):   ~100-150 W
Normal trot (2 m/s):    ~200-300 W
Fast run (4 m/s):       ~400-500 W
Maximum sprint:         ~600+ W

Battery Life Estimates:
- Idle: 4+ hours
- Continuous trot: 1.5-2 hours
- Mixed gait: 2-2.5 hours
- Maximum sprint: 30-60 minutes
```

---

**Document Version:** 1.0  
**Last Updated:** April 2026  
**For Research:** Unitree Go1  
**Companion Files:** 
- Quadruped_Robots_Comparison.xlsx
- Training_Algorithms_Detailed.xlsx
- Training_Algorithms_and_Control.md
