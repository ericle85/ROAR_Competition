"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""
from collections import defaultdict
import math
from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np
import time

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
    
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 0
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        

    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        inf = float('inf')
        if self.current_waypoint_idx < 20:
            throttle_control = inf
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        def getLookAheadDistance():
            currentSpeed =np.linalg.norm(self.velocity_sensor.get_last_gym_observation()) * 3.6
            if currentSpeed >= 180:
                num = 35
            else:
                num = 11
            
            return num
        
        waypointClose = self.maneuverable_waypoints[(self.current_waypoint_idx + getLookAheadDistance()) % len(self.maneuverable_waypoints)]

        waypointMedium = self.maneuverable_waypoints[(self.current_waypoint_idx + getLookAheadDistance()+ 15) % len(self.maneuverable_waypoints)]

        waypointFar = self.maneuverable_waypoints[(self.current_waypoint_idx + getLookAheadDistance()+ 20) % len(self.maneuverable_waypoints)]
        
        # waypoint_to_follow = avg_location
        #x and y components of each waypoint
        point1 = (waypointMedium.location[0], waypointMedium.location[1])
        point2 = (waypointClose.location[0], waypointClose.location[1])
        point3 = (waypointFar.location[0], waypointFar.location[1])
      
        l1 = round(math.dist(point1, point2), 5)
        l2 = round(math.dist(point2, point3), 5)
        l3 = round(math.dist(point1, point3),5)
        zeroChecker = 0.01
        sp = (l1 + l2 + l3) / 2
        area_squared =  sp * (sp - l1) * (sp - l2) * (sp - l3)
        
        # Check if any side lengths are very small
        
        if l1 < zeroChecker or l2 < zeroChecker or l3 < zeroChecker:
            targetSpeed = inf  # Set a high value as a default or handle differently
        else:
            # Check if area_squared is non-zero before calculating radius
            if area_squared > 0:
                def default_value():
                         return 2.8
                frictionCoefficents = defaultdict(default_value,  {
                            0 : inf,
                            1 : 2,
                            2 : 2,
                            4 : 3.7,
                            5 : 4.4,
                            7 : inf,
                            8 : inf,
                            9 : 2.3
                } )
                                
                #coFriction = 1.8  
                coFriction = frictionCoefficents[int((self.current_waypoint_idx % 2775) / 277.5)]
                print(str(coFriction))
                radius = (l1 * l2 * l3) / (4 * math.sqrt(area_squared))
                targetSpeed = math.sqrt(9.81 * coFriction * radius)
            else:
                targetSpeed = inf
        # targetSpeed = max(25, targetSpeed)
        steer = -35
        
        if(self.current_waypoint_idx % 2775 < 1500):
            x= 21
        elif 1800 < self.current_waypoint_idx % 2775 < 2000:
            x = 17
        else:
            x = 38
            
        next_x_waypoints = [
        self.maneuverable_waypoints[(self.current_waypoint_idx + i - 2 ) % len(self.maneuverable_waypoints)]
        for i in range(1, x)
        ]

        
        if (self.current_waypoint_idx % 2775)> 2690:
            waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + 9) % len(self.maneuverable_waypoints)].location
            steer = -25
        else:
            waypoint_to_follow = np.mean([waypoint.location for waypoint in next_x_waypoints], axis=0)  

        vector_to_waypoint = (waypoint_to_follow - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Proportional controller to steer the vehicle towards the target waypoint
        
        
        steer_control = (
            steer / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)
       
       

            # Determine if reversing is necessary
        if targetSpeed < 0.75 * vehicle_velocity_norm: 
            throttle_control = -1  # Reverse with full throttle
            
        else:
            # Apply proportional controller for throttle
            speed_error = targetSpeed - vehicle_velocity_norm  # Convert vehicle speed to kph
            Kp = 500
            throttle_control = Kp * speed_error


        
        if (1290 < (self.current_waypoint_idx % 2775) <1314):
            throttle_control = -.8

        # if 1595 < (self.current_waypoint_idx % 2775) < 1615:
        #     throttle_control = .7

        if 2635 < (self.current_waypoint_idx % 2775) < 2700:
           throttle_control = -.3
        
        
        gear = max(1, (int)((vehicle_velocity_norm * 3.6) / 55))

        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": np.clip(-throttle_control, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0 if throttle_control < 0 else 0,
            "target_gear": gear
        }
        await self.vehicle.apply_action(control)
        return control

    