"""
Bayesian Parameter Optimization for ROAR Competition
Uses Gaussian Process optimization to find optimal hardcoded parameters using sequential evaluations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import json
import time
from dataclasses import dataclass
import asyncio
import subprocess
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

import roar_py_interface
import roar_py_carla
from submission import RoarCompetitionSolution
from infrastructure import RoarCompetitionAgentWrapper
import carla


@dataclass
class BayesianOptimizationConfig:
    """Bayesian optimization configuration"""
    # Parameter space
    target_speed_min: float = 10.0
    target_speed_max: float = 40.0
    
    # Optimization settings
    n_calls: int = 50  # Number of optimization iterations
    n_initial_points: int = 8  # Random points before GP optimization starts
    n_jobs: int = 1  # Number of parallel evaluations (set to 1 for simplicity)
    
    # CARLA server settings
    carla_exe_path: str = r"C:\Users\shrek\Downloads\Monza_V1.1\Monza\CarlaUE4.exe"
    base_port: int = 2000  # CARLA server port
    server_startup_delay: int = 10  # Seconds to wait for server startup
    
    # Evaluation settings
    num_laps_per_eval: int = 1
    eval_timeout: int = 300  # seconds per evaluation
    
    # Logging settings
    results_path: str = "bayesian_results"
    experiment_name: str = "speed_bayesian_v1"


class BayesianParameterOptimizer:
    """
    Bayesian parameter optimizer using Gaussian Process optimization.
    Runs sequential evaluations on a single CARLA server for parameter search.
    """
    
    def __init__(self, config: BayesianOptimizationConfig):
        self.config = config
        
        # Create results directory
        os.makedirs(config.results_path, exist_ok=True)
        
        # Define search space
        self.search_space = [
            Real(config.target_speed_min, config.target_speed_max, name='target_speed')
        ]
        
        # Optimization tracking
        self.evaluation_count = 0
        self.results_history = []
        self.best_parameters = None
        self.best_fitness = float('inf')
        
        # Checkpoint file path
        self.checkpoint_file = os.path.join(
            config.results_path,
            f"{config.experiment_name}_checkpoint.json"
        )
        
        # Server management
        self.server_process = None
        
        print(f"✓ Bayesian parameter optimizer initialized")
        print(f"  - Parameter space: target_speed [{config.target_speed_min}, {config.target_speed_max}]")
        print(f"  - Sequential evaluations: {config.n_calls}")
        print(f"  - Laps per evaluation: {config.num_laps_per_eval}")
        print(f"  - Checkpoint file: {self.checkpoint_file}")
    
    def _stop_carla_server(self):
        """Stop the CARLA server"""
        if self.server_process:
            print("🛑 Stopping CARLA server...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            except Exception as e:
                print(f"Warning: Failed to stop server: {e}")
            
            self.server_process = None
            print("✓ CARLA server stopped")
    
    def _save_checkpoint(self):
        """Save current best results to checkpoint file"""
        checkpoint_data = {
            'experiment_name': self.config.experiment_name,
            'evaluation_count': self.evaluation_count,
            'best_parameters': self.best_parameters,
            'best_fitness': self.best_fitness,
            'timestamp': time.time(),
            'config': {
                'target_speed_min': self.config.target_speed_min,
                'target_speed_max': self.config.target_speed_max,
                'n_calls': self.config.n_calls,
                'n_initial_points': self.config.n_initial_points,
                'num_laps_per_eval': self.config.num_laps_per_eval
            },
            'results_history': self.results_history[-10:]  # Keep last 10 results for context
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"💾 Checkpoint saved: {self.checkpoint_file}")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
    
    def _start_single_carla_server(self):
        """Start single CARLA server for sequential evaluation"""
        port = self.config.base_port
        
        print(f"🚀 Starting CARLA server on port {port}...")
        
        # Start CARLA server
        cmd = [
            self.config.carla_exe_path,
            f"-carla-port={port}"
            # "-RenderOffScreen"  # Run without display for better performance
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1)
        
        # Configure server with no rendering
        # subprocess.run(["python", "utils/config.py", "--no-rendering", "-p", str(port)])
        
        self.server_process = process
        
        # Wait for server to start up
        print(f"⏳ Waiting {self.config.server_startup_delay}s for CARLA server to start...")
        time.sleep(self.config.server_startup_delay)
        
        print(f"✓ CARLA server started successfully on port {port}")
    
    
    async def _evaluate_parameters_single(self, target_speed: float, port: int) -> float:
        """
        Evaluate parameters using the competition runner directly for exact simulation consistency.
        
        Args:
            target_speed: Target speed parameter to evaluate
            port: CARLA server port to use
            
        Returns:
            Fitness score (lap time)
        """
        try:
            # Connect to CARLA server like competition runner
            carla_client = carla.Client('127.0.0.1', port)
            carla_client.set_timeout(5.0)
            roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
            world = roar_py_instance.world
            world.set_control_steps(0.05, 0.005)
            world.set_asynchronous(False)
            
            total_lap_time = 0.0
            successful_laps = 0
            
            print(f"    [Port {port}] Evaluating target_speed={target_speed:.1f}")
            
            for lap in range(self.config.num_laps_per_eval):
                try:
                    # Create a lambda to pass target_speed to RoarCompetitionSolution
                    def ParameterizedSolution(*args, **kwargs):
                        return RoarCompetitionSolution(*args, **kwargs, target_speed=target_speed)
                    
                    # Use the competition runner's evaluation function directly
                    evaluation_result = await self._run_competition_evaluation(
                        world, ParameterizedSolution, self.config.eval_timeout
                    )
                    
                    if evaluation_result is not None:
                        episode_time = evaluation_result["elapsed_time"]
                        successful_laps += 1
                    else:
                        episode_time = 999.0  # Failed evaluation penalty
                    
                    total_lap_time += episode_time
                    print(f"    [Port {port}] Lap {lap + 1}: {episode_time:.2f}s")
                    
                except Exception as lap_error:
                    print(f"    [Port {port}] Lap {lap + 1} failed: {lap_error}")
                    total_lap_time += 999.0  # Penalty for failed lap
            
            # Calculate fitness
            if successful_laps > 0:
                avg_lap_time = total_lap_time / self.config.num_laps_per_eval
                success_rate = successful_laps / self.config.num_laps_per_eval
                # Penalize low success rates
                fitness = avg_lap_time * (2.0 - success_rate)
            else:
                fitness = 999.0  # High penalty for no successful laps
            
            print(f"    [Port {port}] Result: avg_time={avg_lap_time:.2f}s, success_rate={success_rate:.2f}, fitness={fitness:.2f}")
            return fitness
            
        except Exception as e:
            print(f"    [Port {port}] Error during evaluation: {e}")
            return 999.0  # High penalty for errors
    
    async def _run_competition_evaluation(self, world, solution_constructor, max_seconds):
        """
        Run competition evaluation exactly like competition_runner.py evaluate_solution function.
        This ensures identical simulation behavior.
        """
        # Import competition rule class  
        from competition_runner import RoarCompetitionRule
        
        # Spawn vehicle and sensors exactly like competition runner
        waypoints = world.maneuverable_waypoints
        vehicle = world.spawn_vehicle(
            "vehicle.tesla.model3",
            waypoints[0].location + np.array([0,0,1]),
            waypoints[0].roll_pitch_yaw,
            True,
        )
        assert vehicle is not None
        
        camera = vehicle.attach_camera_sensor(
            roar_py_interface.RoarPyCameraSensorDataRGB,
            np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]),
            np.array([0, 10/180.0*np.pi, 0]),
            image_width=1024,
            image_height=768
        )
        location_sensor = vehicle.attach_location_in_world_sensor()
        velocity_sensor = vehicle.attach_velocimeter_sensor()
        rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor()
        occupancy_map_sensor = vehicle.attach_occupancy_map_sensor(50, 50, 2.0, 2.0)
        collision_sensor = vehicle.attach_collision_sensor(np.zeros(3), np.zeros(3))

        assert camera is not None
        assert location_sensor is not None
        assert velocity_sensor is not None
        assert rpy_sensor is not None
        assert occupancy_map_sensor is not None
        assert collision_sensor is not None

        # Start solution exactly like competition runner
        solution = solution_constructor(
            waypoints,
            RoarCompetitionAgentWrapper(vehicle),
            camera,
            location_sensor,
            velocity_sensor,
            rpy_sensor,
            occupancy_map_sensor,
            collision_sensor
        )
        rule = RoarCompetitionRule(waypoints * 1, vehicle, world)  # 1 lap for evaluation

        for _ in range(20):
            await world.step()
        
        rule.initialize_race()

        # Timer starts here - exactly like competition runner
        start_time = world.last_tick_elapsed_seconds
        current_time = start_time
        await vehicle.receive_observation()
        await solution.initialize()

        try:
            crashed = False
            while True:
                # Terminate if time out
                current_time = world.last_tick_elapsed_seconds
                if current_time - start_time > max_seconds:
                    vehicle.close()
                    crashed = True
                    return None
                
                # Receive sensors' data
                await vehicle.receive_observation()
                await rule.tick()

                # Terminate if there is major collision
                collision_impulse_norm = np.linalg.norm(collision_sensor.get_last_observation().impulse_normal)
                if collision_impulse_norm > 100.0:
                    print(f"major collision of intensity {collision_impulse_norm}")
                    crashed = True
                    await rule.respawn()
                    break
                
                if rule.lap_finished():
                    # if rule.progress_bar is not None:
                    #     rule.progress_bar.close()
                    break

                await solution.step()
                await world.step()
            
            end_time = world.last_tick_elapsed_seconds
            vehicle.close()
            
            if crashed:
                return None  # Handled upstream as 999.0
            else:
                return {
                    "elapsed_time": end_time - start_time,
                }
        except Exception as e:
            print(f"Error during competition evaluation: {e}")
            vehicle.close()
            return None
    
    def _evaluate_parameters_single_sync(self, target_speed: float) -> float:
        """
        Evaluate parameters synchronously using single CARLA server.
        This is the function called by scikit-optimize.
        """
        port = self.config.base_port
        
        try:
            # Create a new event loop for this evaluation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                fitness = loop.run_until_complete(
                    self._evaluate_parameters_single(target_speed, port)
                )
            finally:
                loop.close()
            
            # Update global tracking
            self.evaluation_count += 1
            
            # Track best result
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_parameters = {'target_speed': target_speed}
                print(f"🎉 New best parameters: target_speed={target_speed:.1f}, fitness={fitness:.2f}")
                
                # Save checkpoint with new best result
                self._save_checkpoint()
            
            # Store result
            result = {
                'evaluation': self.evaluation_count,
                'target_speed': target_speed,
                'fitness': fitness,
                'port': port,
                'timestamp': time.time()
            }
            self.results_history.append(result)
            
            return fitness
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 999.0
    
    def optimize(self):
        """Main optimization loop using Bayesian optimization"""
        print(f"🎯 Starting Bayesian parameter optimization")
        print(f"   Experiment: {self.config.experiment_name}")
        
        # Initialize checkpoint with starting state
        self._save_checkpoint()
        
        try:
            # Start single CARLA server
            self._start_single_carla_server()
            
            # Define objective function for scikit-optimize
            @use_named_args(self.search_space)
            def objective(**params):
                return self._evaluate_parameters_single_sync(params['target_speed'])
            
            # Run Bayesian optimization
            start_time = time.time()
            
            print(f"🔍 Running Bayesian optimization with {self.config.n_calls} evaluations...")
            
            result = gp_minimize(
                func=objective,
                dimensions=self.search_space,
                n_calls=self.config.n_calls,
                n_initial_points=self.config.n_initial_points,
                n_jobs=1,  # Single evaluation at a time
                acq_func='EI',  # Expected Improvement
                random_state=42
            )
            
            total_time = time.time() - start_time
            
            # Extract results
            best_target_speed = result.x[0]
            best_fitness = result.fun
            
            print(f"\n🎯 BAYESIAN OPTIMIZATION COMPLETE!")
            print(f"   Total time: {total_time:.2f} seconds")
            print(f"   Best target speed: {best_target_speed:.2f}")
            print(f"   Best fitness (lap time): {best_fitness:.2f}s")
            print(f"   Total evaluations: {len(result.func_vals)}")
            
            # Save results
            self._save_results(result)
            
            return {
                'best_parameters': {'target_speed': best_target_speed},
                'best_fitness': best_fitness,
                'n_evaluations': len(result.func_vals),
                'optimization_result': result
            }
            
        except KeyboardInterrupt:
            print("\n⚠️  Optimization interrupted by user")
            if self.best_parameters:
                print(f"Best parameters so far: {self.best_parameters}")
                print(f"Best fitness so far: {self.best_fitness:.2f}")
            return None
            
        except Exception as e:
            print(f"\n❌ Error during optimization: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Always stop server
            self._stop_carla_server()
    
    def _save_results(self, skopt_result):
        """Save optimization results to file"""
        results_file = os.path.join(
            self.config.results_path,
            f"{self.config.experiment_name}_results.json"
        )
        
        # Prepare results data
        results_data = {
            'config': {
                'target_speed_min': self.config.target_speed_min,
                'target_speed_max': self.config.target_speed_max,
                'n_calls': self.config.n_calls,
                'n_initial_points': self.config.n_initial_points,
                'n_jobs': self.config.n_jobs,
                'num_laps_per_eval': self.config.num_laps_per_eval,
                'experiment_name': self.config.experiment_name
            },
            'best_parameters': {
                'target_speed': float(skopt_result.x[0])
            },
            'best_fitness': float(skopt_result.fun),
            'n_evaluations': len(skopt_result.func_vals),
            'all_evaluations': [float(val) for val in skopt_result.func_vals],
            'all_parameters': [[float(x) for x in point] for point in skopt_result.x_iters],
            'results_history': self.results_history
        }
        
        # Save to JSON
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"📁 Results saved to {results_file}")


def main():
    """Example usage"""
    config = BayesianOptimizationConfig(
        n_calls=50,           
        n_initial_points=4, 
        n_jobs=1,             
        num_laps_per_eval=1, 
        target_speed_min=34.25,
        target_speed_max=34.26,
        server_startup_delay=3,
        experiment_name="bayesian_test_v2"
    )
    
    optimizer = BayesianParameterOptimizer(config)
    result = optimizer.optimize()
    
    if result:
        print(f"\n✅ Final Results:")
        print(f"   Best parameters: {result['best_parameters']}")
        print(f"   Best lap time: {result['best_fitness']:.2f}s")


if __name__ == "__main__":
    main()