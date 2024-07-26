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
import os
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


        self.maneuverable_waypoints = (
            roar_py_interface.RoarPyWaypoint.load_waypoint_list(
                np.load(f"{os.path.dirname(__file__)}\\racinglines\\MonzaFinal.npz")
            )
        )
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

        self.startTime = time.time()
        self.prev_steering_error = 0
        self.steering_integral = 0
        self.prev_speed_error = 0
        self.throttle_integral = 0
        
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
        
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        def getLookAheadDistance():
            currentSpeed =np.linalg.norm(self.velocity_sensor.get_last_gym_observation()) * 3.6
            if currentSpeed >= 180:
                num = 36
            else:
                num = 12
            
            return num
    
        
        waypointClose = self.maneuverable_waypoints[(self.current_waypoint_idx + getLookAheadDistance() - 2) % len(self.maneuverable_waypoints)]

        waypointMedium = self.maneuverable_waypoints[(self.current_waypoint_idx + getLookAheadDistance()+ 14) % len(self.maneuverable_waypoints)]

        waypointFar = self.maneuverable_waypoints[(self.current_waypoint_idx + getLookAheadDistance()+ 19) % len(self.maneuverable_waypoints)]
        
        #x and y components of each waypoint
        point1 = (waypointMedium.location[0], waypointMedium.location[1])
        point2 = (waypointClose.location[0], waypointClose.location[1])
        point3 = (waypointFar.location[0], waypointFar.location[1])

        l1 = round(math.dist(point1, point2), 3)
        l2 = round(math.dist(point2, point3), 3)
        l3 = round(math.dist(point1, point3),3)
        zeroChecker = 0.01
        sp = (l1 + l2 + l3) / 2
        area_squared =  sp * (sp - l1) * (sp - l2) * (sp - l3)
        
        # Check if any side lengths are very small
        
        if l1 < zeroChecker or l2 < zeroChecker or l3 < zeroChecker:
            targetSpeed = inf 
        else:
            # Check if area_squared is non-zero before calculating radius
            if area_squared > 0:
                
                #default CoFriction
                def default_value():
                         return 2.2
                frictionCoefficents = defaultdict(default_value,  {
                            0 : inf,
                            1 : 3.2, # 2.15 before
                            2 : 3.42,
                            3 : 3.1,
                            4 : 3, #*
                            5 : 3,
                            6 : 3,
                            7 : inf,
                            8 : 3.3
                } )
        
                coFriction = frictionCoefficents[int((self.current_waypoint_idx % 2775) / 308.33)]
                #print(str(coFriction))

                radius = (l1 * l2 * l3) / (4 * math.sqrt(area_squared))
                targetSpeed = math.sqrt(9.81 * coFriction * radius)
            else:
                targetSpeed = inf
        
        # X is how many waypoitns it looks ahead
        # So X = 20 means look 20 waypoint ahead and averages all waypoints from current location to the waypoint that is 20 ahead
        x = 30      
        if 300 < (self.current_waypoint_idx % 2775) < 570:
            x= 15
        elif 570 <= (self.current_waypoint_idx % 2775) < 780:
            x= 32
        elif 780 <= self.current_waypoint_idx % 2775 < 1700:
            x = 29 #29
        elif 1700 <= self.current_waypoint_idx % 2775 < 2300:
            x = 17
        elif 2600 < self.current_waypoint_idx % 2725:
            x = 15


        if 350< self.current_waypoint_idx % 2775 < 400:
            targetSpeed = 61
        # #averages waypoints in order to get a smooth path
        if (self.current_waypoint_idx % 2775) >= 2725:
            waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + 11) % len(self.maneuverable_waypoints)].location 
        elif (300 < self.current_waypoint_idx % 2775 < 570):
            waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + 6) % len(self.maneuverable_waypoints)].location
        # elif  1600 < (self.current_waypoint_idx % 2775) <= 2300:
        #     waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + 20) % len(self.maneuverable_waypoints)].location
        else:
            next_x_waypoints = [
            self.maneuverable_waypoints[(self.current_waypoint_idx + i - 2 ) % len(self.maneuverable_waypoints)]
            for i in range(1, x)
            ]
            waypoint_to_follow = np.mean([waypoint.location for waypoint in next_x_waypoints], axis=0)  
        
        vector_to_waypoint = (waypoint_to_follow - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        steering_error = delta_heading / np.pi
        steering_derivative = steering_error - self.prev_steering_error
        self.steering_integral += steering_error
        self.prevSteerError = steering_error
        
        # Kp = 2.6
        Kp = 4
        Ki = 0.1
        Kd = 5
        if 1600<  self.current_waypoint_idx % 2775 < 2300:
            Kp = 8
            Kd = 8.4
        elif self.current_waypoint_idx % 2775 >= 2600:
            Kp = 2.6
            Kd = 8
        # Proportional controller to steer the vehicle towards the target waypoint
        steer_control = (
            Kp * steering_error +
            Ki * self.steering_integral +
            Kd * steering_derivative
        )

        self.prev_steering_error = steering_error

        if vehicle_velocity_norm > 1e-2:
            steer_control = -8.0 / np.sqrt(vehicle_velocity_norm) * steer_control
        else:
            steer_control = -np.sign(steer_control)
        
        steer_control = np.clip(steer_control, -1.0, 1.0)
     
     #braking lightly if within a close range of target speed
        if targetSpeed < 0.75  * vehicle_velocity_norm:  #.75
            throttle_control = -1 # -1.8
        else:
            # Apply proportional controller for throttle
            speed_error = targetSpeed - vehicle_velocity_norm
            tKp = 200 #acceleration constant
              
            throttle_control = (tKp * speed_error)
 
    #slowing down for specific points
        if (1290 < (self.current_waypoint_idx % 2775) <1345):
            throttle_control = -.1

        if 2635 < (self.current_waypoint_idx % 2775) < 2700:
           throttle_control = -.05
        
        #always full throttle at the start
        if self.current_waypoint_idx < 20:
            throttle_control = inf
        
        gear = max(1, (int)((vehicle_velocity_norm * 3.6) / 60))
        print(
            str(
                round(vehicle_velocity_norm)
            )
        )
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

    