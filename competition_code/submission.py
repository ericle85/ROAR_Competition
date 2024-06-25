"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""
from collections import defaultdict
import math
from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np

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



        self.current_waypoint_idx = 13
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
       
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        def getLookAheadDistance():
            currentSpeed =np.linalg.norm(self.velocity_sensor.get_last_gym_observation()) * 3.6
            if currentSpeed >= 160:
                num = 35
            else:
                num = 11
            return num
        waypointClose = self.maneuverable_waypoints[(self.current_waypoint_idx +  getLookAheadDistance()) % len(self.maneuverable_waypoints)]

        waypointMedium = self.maneuverable_waypoints[(self.current_waypoint_idx + int(getLookAheadDistance() + 15) ) % len(self.maneuverable_waypoints)]

        waypointFar = self.maneuverable_waypoints[(self.current_waypoint_idx + int( getLookAheadDistance() + 30) ) % len(self.maneuverable_waypoints)]

        waypoint_to_follow = waypointClose
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
        targetSpeed = 20
        # Check if any side lengths are very small
        if l1 < zeroChecker or l2 < zeroChecker or l3 < zeroChecker:
            targetSpeed = 350  # Set a high value as a default or handle differently
        else:
            # Check if area_squared is non-zero before calculating radius
            
            if area_squared > 0:

                def default_value():
                         return 1.8
                frictionCoefficents = defaultdict(default_value,  {
                            0 : 1.6,
                            #4 : .8,
                            6 : 1.1,
                            7 : 2.2,
                            9 : 1.7
                } )
                                
                coFriction = 1.8  
                if 0 <self.current_waypoint_idx <= 2775:
                    coFriction = frictionCoefficents[int(self.current_waypoint_idx / 277.5)]
                if (2775 <self.current_waypoint_idx <= 2775 * 2):
                    coFriction = frictionCoefficents[int((self.current_waypoint_idx - 2775 )/ 277.5)]
                if (2775 * 2 <self.current_waypoint_idx <= 2775 * 3):
                    coFriction = frictionCoefficents[int((self.current_waypoint_idx - 2775 * 2 )/ 277.5)]
                
                radius = (l1 * l2 * l3) / (4 * math.sqrt(area_squared))
                globalRadius = radius
                targetSpeed = math.sqrt(9.81 * coFriction * radius)
               # print("radius: " + str(radius))
            else:
                # Handle case where area_squared is zero
                # This could involve setting targetSpeed to a default value or handling it differently
                targetSpeed = float('inf')  # Example of setting targetSpeed to a high value


        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Proportional controller to steer the vehicle towards the target waypoint
        steer_control = (
            -8.0 / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)
       
        speed_error = targetSpeed - vehicle_velocity_norm * 3.6  # Convert vehicle speed to kph
        Kp = 8

            # Determine if reversing is necessary
        if targetSpeed < 0.75 * vehicle_velocity_norm * 3.6: 
            throttle_control = -1  # Reverse with full throttle
            
        else:
            # Apply proportional controller for throttle
            speed_error = targetSpeed - vehicle_velocity_norm * 3.6  # Convert vehicle speed to kph
            Kp = 5
            throttle_control = Kp * speed_error

        
        # Apply brake if speed error is small
        if abs(speed_error) < 3:
            brake_control = .75
            
        else:
            brake_control = 0.0
            

       # print("target: " + str(round(targetSpeed,3)) + "| curr kph: " + str(int(vehicle_velocity_norm * 3.6)))
        

        

        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": np.clip(-throttle_control, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0 if throttle_control < 0 else 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return control

    