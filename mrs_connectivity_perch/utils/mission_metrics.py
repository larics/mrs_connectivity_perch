
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class MissionMetrics:
    def __init__(self, log_directory: str = "logs", mission_name: str = None, config_name: str = None, trial_number: int = 1):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        if mission_name is None:
            mission_name = f"mission_{int(time.time())}"
        self.mission_name = mission_name
        self.config_name = config_name or "unknown_config"
        self.trial_number = trial_number
        
        # Core metrics
        self.mission_start_time = time.time()
        self.controller_steps = 0
        self.network_connected_steps = 0
        self.robot_base_connected_steps = 0
        
        self.fiedler_history = []  # Network connectivity values
        # Only track robot-base connectivity (separate from network Fiedler)
        self.robot_base_connectivity_history = []  # Robot-base connectivity
        
        # UAV and battery tracking over time
        self.uav_count_history = []  # Number of UAVs at each timestep
        self.average_battery_history = []  # Average battery level at each timestep
        
        # State tracking
        self.last_fiedler_value = 0.0
        self.last_robot_base_connected = False
        self.last_uav_count = 0
        self.last_average_battery = 0.0
        
        # LLM metrics tracking
        self.llm_prompts = []  # Store all prompts with timestamps
        self.llm_response_times = []  # Store response times
        self.llm_total_prompts = 0
        self.llm_total_time = 0.0
        self.llm_errors = 0
        
        # UAV loss tracking
        self.initial_uav_count = None  # Will be set on first update_step
        
        print(f"📊 Mission Metrics initialized: {self.mission_name}")
        print(f"📊 Tracking: Connectivity + LLM metrics")
    
    def update_step(self, swarm, fiedler_value: float = None, robot_base_connected: bool = None):
        """Update metrics for current controller step."""
        self.controller_steps += 1
        
        # Set initial UAV count on first call
        if self.initial_uav_count is None:
            self.initial_uav_count = len([a for a in swarm.agents if a.type == 'UAV'])
        
        # Auto-calculate Fiedler value if not provided
        if fiedler_value is None:
            fiedler_value = self._calculate_fiedler_value(swarm)
        
        # Auto-calculate robot-base connectivity if not provided
        if robot_base_connected is None:
            robot_base_connected = self._check_robot_base_connectivity(swarm)
        
        # Calculate UAV metrics
        uav_count = len([a for a in swarm.agents if a.type == 'UAV'])
        average_battery = self._calculate_average_battery(swarm)
        
        # Update connectivity counters
        if fiedler_value > 0:
            self.network_connected_steps += 1
        
        if robot_base_connected:
            self.robot_base_connected_steps += 1
        
        # Store essential data for JSON output
        self.fiedler_history.append(fiedler_value)
        self.robot_base_connectivity_history.append(robot_base_connected)
        self.uav_count_history.append(uav_count)
        self.average_battery_history.append(average_battery)
        
        # Update state
        self.last_fiedler_value = fiedler_value
        self.last_robot_base_connected = robot_base_connected
        self.last_uav_count = uav_count
        self.last_average_battery = average_battery
    
    def log_llm_prompt(self, prompt: str, response_time: float = None, success: bool = True, error: str = None, function_calls: List[Dict] = None):
        """
        Log an LLM interaction (minimally invasive).
        
        Args:
            prompt: The prompt sent to LLM
            response_time: Time taken for LLM response (seconds)
            success: Whether the LLM call was successful
            error: Error message if failed (responses not logged to avoid bloat)
            function_calls: List of function calls made by LLM (name + arguments only)
        """
        timestamp = time.time()
        
        # Track summary stats
        self.llm_total_prompts += 1
        if response_time is not None:
            self.llm_response_times.append(response_time)
            self.llm_total_time += response_time
        if not success:
            self.llm_errors += 1
        
        # Store prompt details
        prompt_log = {
            'timestamp': timestamp,
            'relative_time': timestamp - self.mission_start_time,
            'prompt': prompt[:500] + '...' if len(prompt) > 500 else prompt,  # Truncate long prompts
            'response_time': response_time,
            'success': success,
            'controller_step': self.controller_steps  # Link to simulation step
        }
        
        # Store function calls (LLM intent) without results - compact and useful!
        if function_calls:
            prompt_log['function_calls'] = []
            for call in function_calls:
                # Extract function name and arguments, preserve phase info
                call_summary = {
                    'function': call.get('function', call.get('name', 'unknown')),  # Handle both field names
                    'arguments': call.get('arguments', {})
                }
                
                # Preserve phase information if available
                if 'phase' in call:
                    call_summary['phase'] = call['phase']
                
                # Truncate large argument values to prevent bloat
                if isinstance(call_summary['arguments'], dict):
                    for key, value in call_summary['arguments'].items():
                        if isinstance(value, str) and len(value) > 200:
                            call_summary['arguments'][key] = value[:200] + '...'
                prompt_log['function_calls'].append(call_summary)
        
        # Only store error messages, not successful responses (to avoid bloat)
        if error:
            prompt_log['error'] = str(error)[:200]  # Truncate long error messages too
            
        self.llm_prompts.append(prompt_log)
    
    def get_llm_stats(self) -> Dict[str, Any]:
        """Get current LLM performance statistics."""
        if not self.llm_response_times:
            return {
                'total_prompts': self.llm_total_prompts,
                'errors': self.llm_errors,
                'avg_response_time': 0,
                'total_time': self.llm_total_time
            }
        
        return {
            'total_prompts': self.llm_total_prompts,
            'successful_prompts': len(self.llm_response_times),
            'errors': self.llm_errors,
            'avg_response_time': sum(self.llm_response_times) / len(self.llm_response_times),
            'min_response_time': min(self.llm_response_times),
            'max_response_time': max(self.llm_response_times),
            'total_time': self.llm_total_time,
            'success_rate': (len(self.llm_response_times) / self.llm_total_prompts * 100) if self.llm_total_prompts > 0 else 0
        }
    
    def _calculate_fiedler_value(self, swarm) -> float:
        """Calculate Fiedler algebraic connectivity value from swarm - EXCLUDING perching positions."""
        try:
            # Exclude perching positions from Fiedler calculation
            allowed_edge_types = {('UAV', 'UAV'), ('UAV', 'robot'), ('UAV', 'base'), 
                                 ('robot', 'robot'), ('robot', 'base'), ('base', 'base')}
            A = swarm.compute_adjacency_matrix(allowed_edge_types=allowed_edge_types)
            
            fiedler_value = swarm.algebraic_connectivity(A)
            return fiedler_value
        except Exception as e:
            print(f"  Warning: Could not calculate Fiedler value: {e}")
            return 0.0
    
    def _calculate_average_battery(self, swarm) -> float:
        """Calculate average battery level of all UAVs."""
        try:
            uavs = [a for a in swarm.agents if a.type == 'UAV']
            if not uavs:
                return 0.0
            
            total_battery = sum(getattr(uav, 'battery', 0.0) for uav in uavs)
            return total_battery / len(uavs)
        except Exception as e:
            print(f"  Warning: Could not calculate average battery: {e}")
            return 0.0
    
    def _check_robot_base_connectivity(self, swarm) -> bool:
        try:
            # Find robot and base agents
            robot_id = None
            base_id = None
            
            for i, agent in enumerate(swarm.agents):
                # Handle different agent type attribute names
                agent_type = getattr(agent, 'agent_type', None) or getattr(agent, 'type', None) or type(agent).__name__.lower()
                
                if 'robot' in str(agent_type).lower():
                    robot_id = i  # Use index in agents list
                elif 'base' in str(agent_type).lower():
                    base_id = i   # Use index in agents list
            
            if robot_id is None or base_id is None:
                return False
            
            # Use swarm's adjacency matrix calculation (EXCLUDING perching positions like Fiedler)
            allowed_edge_types = {('UAV', 'UAV'), ('UAV', 'robot'), ('UAV', 'base'), 
                                 ('robot', 'robot'), ('robot', 'base'), ('base', 'base')}
            adjacency_matrix = swarm.compute_adjacency_matrix(allowed_edge_types=allowed_edge_types)
            
            if adjacency_matrix.size == 0:
                return False
                
            # Map original agent indices to filtered matrix indices
            filtered_agents = [agent for agent in swarm.agents if agent.type in {'UAV', 'robot', 'base'}]
            robot_filtered_id = None
            base_filtered_id = None
            
            for i, agent in enumerate(filtered_agents):
                agent_type = getattr(agent, 'agent_type', None) or getattr(agent, 'type', None) or type(agent).__name__.lower()
                if 'robot' in str(agent_type).lower():
                    robot_filtered_id = i
                elif 'base' in str(agent_type).lower():
                    base_filtered_id = i
            
            if robot_filtered_id is None or base_filtered_id is None:
                return False
                
            return self._is_path_connected(adjacency_matrix, robot_filtered_id, base_filtered_id)
            
        except Exception as e:
            print(f"  Warning: Could not check robot-base connectivity: {e}")
            return False
    
    def _is_path_connected(self, adjacency_matrix, start_id, end_id) -> bool:
        try:
            n_nodes = adjacency_matrix.shape[0]
            if start_id >= n_nodes or end_id >= n_nodes:
                return False
            
            visited = set()
            queue = [start_id]
            
            while queue:
                current = queue.pop(0)
                if current == end_id:
                    return True
                
                if current in visited:
                    continue
                
                visited.add(current)
                
                # Add neighbors to queue
                for neighbor in range(n_nodes):
                    if adjacency_matrix[current, neighbor] > 0 and neighbor not in visited:
                        queue.append(neighbor)
            
            return False
        except Exception:
            return False
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary (without saving to file)."""
        current_time = time.time()
        
        return {
            'mission_info': {
                'name': self.mission_name,
                'start_time': self.mission_start_time,
                'current_duration_steps': self.controller_steps,
                'real_time_duration': current_time - self.mission_start_time
            },
            'connectivity_metrics': {
                'network_connected_steps': self.network_connected_steps,
                'robot_base_connected_steps': self.robot_base_connected_steps,
                'network_uptime_percentage': (self.network_connected_steps / self.controller_steps * 100) if self.controller_steps > 0 else 0,
                'robot_base_uptime_percentage': (self.robot_base_connected_steps / self.controller_steps * 100) if self.controller_steps > 0 else 0
            },
            'current_state': {
                'current_fiedler_value': self.last_fiedler_value,
                'current_robot_base_connected': self.last_robot_base_connected,
                'current_uav_count': self.last_uav_count,
                'current_average_battery': self.last_average_battery,
                'total_data_points': len(self.fiedler_history)
            },
            'llm_metrics': self.get_llm_stats()
        }
    
    def save_metrics(self, swarm=None, filename: str = None) -> str:
        # Log UAV losses automatically if swarm is provided
        if swarm is not None and self.initial_uav_count is not None:
            current_uav_count = len([a for a in swarm.agents if a.type == 'UAV'])
            uavs_lost = self.initial_uav_count - current_uav_count
            print(f"📊 Mission UAV Summary: Started with {self.initial_uav_count}, ended with {current_uav_count}, lost {uavs_lost} UAVs")
        
        if filename is None:
            # Generate filename in format: {date}_{time}_configfile_trialx
            now = datetime.now()
            date_str = now.strftime("%Y%m%d")  # YYYYMMDD
            time_str = now.strftime("%H%M%S")  # HHMMSS
            
            # Clean config name (remove .yaml extension and path)
            config_clean = Path(self.config_name).stem if self.config_name != "unknown_config" else self.config_name
            
            filename = f"{date_str}_{time_str}_{config_clean}_trial{self.trial_number}_metrics.json"
        
        filepath = self.log_directory / filename
        
        # Calculate UAV metrics for JSON
        uav_metrics = {}
        if swarm is not None and self.initial_uav_count is not None:
            current_uav_count = len([a for a in swarm.agents if a.type == 'UAV'])
            uav_metrics = {
                'initial_uav_count': self.initial_uav_count,
                'final_uav_count': current_uav_count,
                'uavs_lost': self.initial_uav_count - current_uav_count,
                'survival_rate_percentage': (current_uav_count / self.initial_uav_count * 100) if self.initial_uav_count > 0 else 0
            }
        
        # Prepare MINIMAL metrics data - just the essentials!
        metrics_data = {
            'mission_info': {
                'name': self.mission_name,
                'start_time': self.mission_start_time,
                'total_duration_steps': self.controller_steps,
                'real_time_duration': time.time() - self.mission_start_time
            },
            'connectivity_metrics': {
                'network_connected_steps': self.network_connected_steps,
                'robot_base_connected_steps': self.robot_base_connected_steps,
                'network_uptime_percentage': (self.network_connected_steps / self.controller_steps * 100) if self.controller_steps > 0 else 0,
                'robot_base_uptime_percentage': (self.robot_base_connected_steps / self.controller_steps * 100) if self.controller_steps > 0 else 0
            },
            'uav_metrics': uav_metrics,
            'history': {
                'fiedler_values': self.fiedler_history,
                'robot_base_connectivity': self.robot_base_connectivity_history,
                'uav_count': self.uav_count_history,
                'average_battery': self.average_battery_history
            },
            'llm_metrics': self.get_llm_stats(),
            'llm_interactions': self.llm_prompts,
            'final_state': {
                'final_fiedler_value': self.last_fiedler_value,
                'final_robot_base_connected': self.last_robot_base_connected,
                'final_uav_count': self.last_uav_count,
                'final_average_battery': self.last_average_battery
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        print(f" Metrics saved to: {filepath}")
        return str(filepath)


# Convenience function for easy integration
def create_mission_metrics(log_directory: str = "logs", mission_name: str = None, config_name: str = None, trial_number: int = 1) -> MissionMetrics:
    return MissionMetrics(log_directory, mission_name, config_name, trial_number)


# Global instance for optional non-invasive LLM tracking
_global_metrics_instance = None

def set_global_metrics_instance(metrics_instance: MissionMetrics):
    """Set a global metrics instance for automatic LLM tracking."""
    global _global_metrics_instance
    _global_metrics_instance = metrics_instance

def track_llm_call(prompt: str, response_time: float = None, success: bool = True, error: str = None, function_calls: List[Dict] = None):
    """Track an LLM call using the global metrics instance (if available)."""
    if _global_metrics_instance:
        _global_metrics_instance.log_llm_prompt(prompt, response_time, success, error, function_calls)

class LLMTracker:
    """Context manager for tracking LLM calls automatically."""
    
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.start_time = None
        self.success = True
        self.response = None
        self.error = None
        self.function_calls = []
    
    def add_function_call(self, function_name: str, arguments: Dict[str, Any]):
        """Add a function call made by the LLM."""
        self.function_calls.append({
            'name': function_name,
            'arguments': arguments
        })
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        response_time = time.time() - self.start_time if self.start_time else None
        
        if exc_type is not None:
            self.success = False
            self.error = str(exc_val)
        
        track_llm_call(
            prompt=self.prompt,
            response_time=response_time,
            success=self.success,
            error=self.error,
            function_calls=self.function_calls if self.function_calls else None
        )
        return False  # Don't suppress exceptions
