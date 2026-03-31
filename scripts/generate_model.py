#!/usr/bin/env python3
"""
Generate a MuJoCo MJCF model of the MIT Mini Cheetah.
Physical parameters from Kim et al. 2019 and the MIT Mini Cheetah open-source repo.
"""
import os
import json

PARAMS = {
    "body_mass": 6.0,
    "body_length": 0.40,
    "body_width": 0.10,
    "body_height": 0.05,
    "thigh_length": 0.209,
    "calf_length": 0.175,
    "foot_radius": 0.02,
    "hip_offset_lateral": 0.062,
    "hip_offset_sagittal": 0.19,
    "clearance_height": 0.30,
    "hip_abduction_range": 0.523,
    "hip_rotation_range": 1.047,
    "knee_min": -2.697,
    "knee_max": -0.524,
    "actuator_kp": 80.0,
    "actuator_kd": 1.0,
    "max_torque": 17.0,
}


def make_leg(prefix, sag_sign, lat_sign, p):
    """Generate MJCF for one leg (FR/FL/RR/RL)."""
    hip_x = p["hip_offset_sagittal"] * sag_sign
    hip_y = p["hip_offset_lateral"] * lat_sign
    tl = p["thigh_length"]
    cl = p["calf_length"]
    fr = p["foot_radius"]
    # Lateral offset for thigh (small)
    thigh_y = 0.035 * lat_sign

    return f"""
      <!-- {prefix} leg -->
      <body name="{prefix}_hip" pos="{hip_x:.4f} {hip_y:.4f} 0">
        <inertial pos="0 0 0" mass="0.54" diaginertia="0.001 0.001 0.001"/>
        <joint name="{prefix}_hip_abduct" axis="1 0 0" range="-{p['hip_abduction_range']:.3f} {p['hip_abduction_range']:.3f}" damping="0.5" armature="0.01"/>
        <geom name="{prefix}_hip_geom" type="capsule" size="0.025 0.02" quat="0.707 0 0.707 0" material="leg_mat" contype="0" conaffinity="0"/>

        <body name="{prefix}_thigh" pos="0 {thigh_y:.4f} 0">
          <inertial pos="0 0 {-tl/2:.4f}" mass="0.634" diaginertia="0.002 0.002 0.0003"/>
          <joint name="{prefix}_hip_rotate" axis="0 1 0" range="-{p['hip_rotation_range']:.3f} {p['hip_rotation_range']:.3f}" damping="0.5" armature="0.01"/>
          <geom name="{prefix}_thigh_geom" type="capsule" fromto="0 0 0 0 0 {-tl:.4f}" size="0.018" material="leg_mat" contype="0" conaffinity="0"/>

          <body name="{prefix}_shin" pos="0 0 {-tl:.4f}">
            <inertial pos="0 0 {-cl/2:.4f}" mass="0.280" diaginertia="0.001 0.001 0.00005"/>
            <joint name="{prefix}_knee" axis="0 1 0" range="{p['knee_min']:.3f} {p['knee_max']:.3f}" damping="0.5" armature="0.01"/>
            <geom name="{prefix}_shin_geom" type="capsule" fromto="0 0 0 0 0 {-cl:.4f}" size="0.013" material="leg_mat" contype="0" conaffinity="0"/>

            <body name="{prefix}_foot" pos="0 0 {-cl:.4f}">
              <inertial pos="0 0 0" mass="0.06" diaginertia="0.00001 0.00001 0.00001"/>
              <geom name="{prefix}_foot_geom" type="sphere" size="{fr:.4f}" material="foot_mat" condim="6" friction="1.5 0.005 0.0001" solimp="0.9 0.95 0.001" solref="0.02 1"/>
              <site name="{prefix}_foot_site" pos="0 0 0" size="0.01"/>
            </body>
          </body>
        </body>
      </body>"""


def generate_mjcf(p):
    bhl = p["body_length"] / 2
    bhw = p["body_width"] / 2
    bhh = p["body_height"] / 2
    mt = p["max_torque"]

    fr_leg = make_leg("FR", 1, 1, p)
    fl_leg = make_leg("FL", 1, -1, p)
    rr_leg = make_leg("RR", -1, 1, p)
    rl_leg = make_leg("RL", -1, -1, p)

    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="mini_cheetah">
  <compiler angle="radian" autolimits="true"/>

  <option timestep="0.002" gravity="0 0 -9.81" iterations="50" solver="Newton" cone="elliptic"/>

  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <map force="0.1" zfar="30"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.05 0.05 0.2" width="100" height="100"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.6 0.6 0.6" rgb2="0.4 0.4 0.4" width="300" height="300"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="10 10" specular="0.5" shininess="0.3" reflectance="0.1"/>
    <material name="body_mat" rgba="0.2 0.3 0.8 1" specular="0.5" shininess="0.8"/>
    <material name="leg_mat" rgba="0.25 0.25 0.25 1" specular="0.3" shininess="0.5"/>
    <material name="foot_mat" rgba="0.1 0.1 0.1 1" specular="0.2" shininess="0.3"/>
  </asset>

  <default>
    <joint limited="true" armature="0.01" damping="0.1"/>
    <geom condim="6" friction="0.8 0.02 0.01" margin="0.001"/>
    <motor ctrllimited="true" ctrlrange="-{mt:.1f} {mt:.1f}"/>
  </default>

  <worldbody>
    <light name="top" pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <light name="front" pos="2 -2 2" dir="-1 1 -1" diffuse="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="50 50 0.1" material="floor_mat" condim="6" friction="1.0 0.005 0.0001"/>

    <!-- Robot body -->
    <body name="base" pos="0 0 {p['clearance_height']:.3f}">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{p['body_mass']:.1f}" diaginertia="0.05 0.15 0.15"/>
      <geom name="body_box" type="box" size="{bhl:.4f} {bhw:.4f} {bhh:.4f}" material="body_mat" contype="0" conaffinity="0"/>
      <site name="imu_site" pos="0 0 0" size="0.01"/>
      {fr_leg}
      {fl_leg}
      {rr_leg}
      {rl_leg}
    </body>
  </worldbody>

  <actuator>
    <motor name="FR_hip_abduct_motor" joint="FR_hip_abduct" gear="1"/>
    <motor name="FR_hip_rotate_motor" joint="FR_hip_rotate" gear="1"/>
    <motor name="FR_knee_motor"       joint="FR_knee"        gear="1"/>
    <motor name="FL_hip_abduct_motor" joint="FL_hip_abduct" gear="1"/>
    <motor name="FL_hip_rotate_motor" joint="FL_hip_rotate" gear="1"/>
    <motor name="FL_knee_motor"       joint="FL_knee"        gear="1"/>
    <motor name="RR_hip_abduct_motor" joint="RR_hip_abduct" gear="1"/>
    <motor name="RR_hip_rotate_motor" joint="RR_hip_rotate" gear="1"/>
    <motor name="RR_knee_motor"       joint="RR_knee"        gear="1"/>
    <motor name="RL_hip_abduct_motor" joint="RL_hip_abduct" gear="1"/>
    <motor name="RL_hip_rotate_motor" joint="RL_hip_rotate" gear="1"/>
    <motor name="RL_knee_motor"       joint="RL_knee"        gear="1"/>
  </actuator>

  <sensor>
    <framelinvel name="base_linvel" objtype="site" objname="imu_site"/>
    <frameangvel name="base_angvel" objtype="site" objname="imu_site"/>
    <framequat name="base_quat" objtype="site" objname="imu_site"/>
    <accelerometer name="base_accel" site="imu_site"/>
    <gyro name="base_gyro" site="imu_site"/>
  </sensor>
</mujoco>"""
    return xml


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    xml = generate_mjcf(PARAMS)
    with open("assets/mini_cheetah.xml", "w") as f:
        f.write(xml)
    print("Generated: assets/mini_cheetah.xml")

    with open("assets/mini_cheetah_params.json", "w") as f:
        json.dump(PARAMS, f, indent=2)
    print("Generated: assets/mini_cheetah_params.json")
