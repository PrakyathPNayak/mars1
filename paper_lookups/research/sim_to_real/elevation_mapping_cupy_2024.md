# Elevation Mapping CuPy: GPU-Accelerated Multi-Modal Elevation Mapping Framework

**Authors:** Takahiro Miki, Lorenz Wellhausen, Ruben Grandia, Fabian Jenelten, Timon Homberger, Marco Hutter (ETH Zurich / Robotic Systems Lab)
**Year:** 2024 | **Venue:** GitHub / Technical Report (leggedrobotics)
**Links:** [GitHub](https://github.com/leggedrobotics/elevation_mapping_cupy)

---

## Abstract Summary
Elevation Mapping CuPy is a GPU-accelerated framework for building and maintaining real-time multi-modal elevation maps suitable for legged robot locomotion. The framework fuses geometric data (depth cameras, LiDAR), semantic information (segmentation masks), and visual features (learned embeddings) into a unified 2.5D grid representation that captures terrain geometry and traversability. By leveraging CuPy (GPU-accelerated NumPy) for all core computations—raycasting, map fusion, artifact removal, and layer operations—the system achieves real-time update rates (>30 Hz) on embedded GPU hardware (NVIDIA Jetson) while maintaining high-resolution maps.

The system addresses several practical challenges in elevation mapping for legged robots: sensor noise and outlier rejection through statistical filtering, dynamic obstacle handling through temporal decay, body self-collision removal through known robot geometry masking, and multi-resolution map management for balancing local detail with global coverage. The elevation map is stored as a multi-layer grid where each cell contains elevation, variance, semantic class, traversability score, and optional learned features—enabling downstream planners to query both geometric and semantic terrain properties.

A critical feature for learning-based locomotion is the extensibility with learned traversability filters. The framework provides a plugin architecture where neural network-based terrain classifiers can be attached as additional map layers, converting raw geometric/visual data into traversability scores that directly interface with RL-based locomotion controllers. This bridges the gap between traditional geometric mapping and learning-based terrain-aware locomotion.

## Core Contributions
- GPU-accelerated elevation mapping achieving >30 Hz on embedded hardware (Jetson AGX)
- Multi-modal fusion of geometric (depth/LiDAR), semantic, and visual data into unified grid representation
- Raycasting-based map update with statistical outlier rejection and variance tracking
- Body self-collision removal using known robot URDF geometry
- Dynamic obstacle handling through temporal elevation decay
- Plugin architecture for learning-based traversability filters (neural network layers)
- Open-source ROS2-compatible implementation with extensive documentation and examples

## Methodology Deep-Dive
The core data structure is a 2.5D grid map M with cell resolution r (typically 2-5 cm) covering an L×L area (typically 4-8 meters) centered on the robot. Each cell (i,j) stores a multi-layer representation: elevation h_{i,j}, variance σ²_{i,j}, update count n_{i,j}, semantic class c_{i,j}, traversability score τ_{i,j}, and K optional feature channels f_{i,j}^k. The map is implemented as a set of CuPy arrays (GPU tensors) enabling massively parallel cell-wise operations.

Map updates from depth cameras and LiDAR follow a raycasting pipeline. For each sensor measurement, the 3D point is computed in the world frame using the sensor-to-robot and robot-to-world transforms. The point is then projected onto the 2D grid to determine the affected cell (i,j). The elevation update uses a Kalman filter-style fusion: h_{i,j}^{new} = (σ²_{meas} · h_{i,j}^{old} + σ²_{i,j}^{old} · h_{meas}) / (σ²_{i,j}^{old} + σ²_{meas}), with measurement variance σ²_{meas} computed from the sensor noise model (increasing with range squared for depth cameras). This produces smooth, low-noise elevation estimates that converge as more measurements accumulate.

Raycasting for visibility and artifact removal is performed on the GPU. For each sensor measurement, a ray is traced from the sensor origin through the measurement point. Grid cells along the ray that have elevations above the ray height are identified as potential artifacts (caused by dynamic objects that have moved) and their elevations are decayed: h_{i,j} ← h_{i,j} · (1 - α_decay). This raycasting is parallelized across all measurements in a single GPU kernel, processing a full depth image (640×480 points) in <1 ms. The artifact removal is crucial for legged robots operating near people or in dynamic environments.

Body self-collision removal masks map cells that are occluded by the robot's own body. The robot's collision geometry (loaded from the URDF) is projected onto the grid at the current robot pose, and cells inside the projection are excluded from updates. This prevents the robot's own legs or body from corrupting the elevation map—a common issue with downward-facing depth cameras on legged robots. The collision geometry is simplified to convex hulls for efficient GPU-based point-in-polygon testing.

The traversability layer computes per-cell traversability scores from geometric features: local slope (computed from elevation gradients), roughness (standard deviation of elevation in a local window), step height (maximum elevation difference to neighbors), and curvature. Each feature is converted to a cost using sigmoid functions with learned or tuned thresholds: τ_{i,j} = σ(-(slope_{i,j} - slope_thresh) / slope_scale) · σ(-(rough_{i,j} - rough_thresh) / rough_scale) · ... The product of individual traversability factors gives the final score τ ∈ [0,1], where 1 is perfectly traversable and 0 is impassable.

The plugin architecture enables neural network traversability filters to replace or augment the geometric heuristics. A plugin receives the multi-layer map as input and writes to a designated output layer. For example, a CNN can process a local patch of elevation + RGB data and output a learned traversability score that accounts for deformable terrain, vegetation, and surface material—factors invisible to geometric analysis alone. The plugin interface uses PyTorch with CuPy interop for zero-copy GPU data transfer between the mapping pipeline and the neural network.

## Key Results & Numbers
- Update rate: >30 Hz on NVIDIA Jetson AGX Xavier, >100 Hz on desktop GPU
- Map resolution: 2-5 cm cells covering 4-8 meter area
- Raycasting throughput: 640×480 depth image processed in <1 ms on GPU
- Sensor support: Intel RealSense D435/D455, Velodyne VLP-16, Ouster OS1, ZED2
- Memory footprint: ~50-200 MB GPU memory depending on resolution and number of layers
- Deployed on ANYmal, Spot, and custom legged platforms at ETH Zurich
- Open-source with ROS2 integration and Rviz visualization
- Supports up to 20 custom map layers simultaneously

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**

Elevation Mapping CuPy provides the terrain perception pipeline for the Mini Cheetah's outdoor navigation capabilities. The GPU-accelerated framework can run on the Mini Cheetah's onboard compute (Jetson Xavier/Orin) at >30 Hz, providing real-time terrain geometry and traversability estimates to the locomotion controller. The multi-layer grid representation—with elevation, slope, roughness, and traversability—directly interfaces with terrain-aware RL policies.

For the Mini Cheetah's sim-to-real pipeline, the elevation map provides structured terrain observations that are easier to simulate accurately than raw sensor data. In MuJoCo, height field terrains can be directly converted to synthetic elevation maps for training, ensuring representation consistency between simulation and reality. The traversability layer can be used as input to the RL policy's value function, enabling terrain-cost-aware foothold planning.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**

For Cassie's hierarchical system, Elevation Mapping CuPy provides the input pipeline for the Contrastive Pre-trained Terrain Encoder (CPTE) at the Planner level. The elevation map's multi-layer grid is the raw terrain observation that the CPTE processes into a compact terrain embedding for world model conditioning. The framework's plugin architecture allows the CPTE neural network to be integrated directly into the mapping pipeline, computing terrain embeddings at map update rate for low-latency planning.

The traversability layer also informs Cassie's Safety level: terrain regions with low traversability scores can trigger conservative safety constraints in the CBF-QP filter, reducing step length and increasing stance time when approaching difficult terrain. The real-time update rate (>30 Hz) ensures the Safety layer has current terrain information for constraint computation.

## What to Borrow / Implement
- Deploy Elevation Mapping CuPy on Mini Cheetah's Jetson for real-time terrain perception
- Use the elevation + traversability layers as RL policy observations for terrain-aware locomotion
- Integrate the CPTE as a plugin layer in the mapping framework for Cassie's Planner terrain encoding
- Generate synthetic elevation maps from MuJoCo height fields during sim training for representation consistency
- Use the traversability scores as inputs to Cassie's CBF-QP safety filter for terrain-adaptive constraints

## Limitations & Open Questions
- 2.5D representation cannot capture overhanging structures (bridges, tunnels) or multi-level terrain
- Elevation accuracy degrades at map boundaries and areas with sparse sensor coverage
- Traversability heuristics require manual threshold tuning; learning-based plugins need labeled terrain data
- ROS2 dependency may add latency for non-ROS control architectures; direct CuPy API usage is possible but less documented
