import matplotlib
matplotlib.use("TkAgg")  # Or "QtAgg" depending on your environment

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


class SwarmRenderer:
    def __init__(self, render_type, env, swarm, occupancy_grid, origin, resolution, vis_radius=None, plot_limits=None, show_battery_plot=False, rotate_90=False):
        self.env = env
        self.render_type = render_type
        self.swarm = swarm
        self.occupancy_grid = occupancy_grid
        self.origin = origin
        self.resolution = resolution
        self.vis_radius = vis_radius
        self.rotate_90 = False  # CHANGED BACK TO FALSE
        self.fig = None
        self.ax = None
        self.agent_markers = []  # Stores matplotlib Line2D objects (not scatter)
        self.sensor_circles = []
        self.paths = []  # List to store agent paths
        self.old_paths = []  # List to store old agent paths
        self.adjacency_lines = []
        self.battery_circles = []  # List to store battery status circles
        self.type_styles = {
            "UAV": {'cmap': 'Blues', 'marker': 'o'},
            "perching_pos": {'cmap': 'Greens', 'marker': '^'},
            "robot": {'cmap': 'YlOrBr', 'marker': 's'},
            "base": {'cmap': 'Greens', 'marker': 's'} 
        }
        self.plot_limits = plot_limits
        self.show_battery_plot = show_battery_plot
        
        # Battery plot variables
        self.battery_fig = None
        self.battery_ax = None
        self.battery_lines = {}
        self.battery_data = {}  # Store battery history for each UAV
        self.time_data = []
        self.current_time = 0

        # HARDCODED HIGHLIGHT CIRCLES - Multiple agents with different colors
        self.highlight_circles = {}  # Dictionary to store multiple circles by agent_id
        
        # HARDCODE SPECIFIC AGENT IDS AND COLORS
        self.highlight_circles[1] = {
            'radius': 0.15,
            'color': 'blue',
            'circle': None
        }
        
        # self.highlight_circles[0] = {
        #     'radius': 0.15,
        #     'color': 'green',
        #     'circle': None
        # }
        
        # You can add more agents as needed:
        # self.highlight_circles[7] = {
        #     'radius': 0.20,
        #     'color': 'orange',
        #     'circle': None
        # }

    def set_highlight_agent(self, agent_id, radius=0.5, color='blue'):
        """Set which agent to highlight with a colored dashed circle."""
        self.highlight_circles[agent_id] = {
            'radius': radius,
            'color': color,
            'circle': None
        }

    def clear_highlight_agent(self, agent_id):
        """Remove the highlight circle for a specific agent."""
        if agent_id in self.highlight_circles:
            if self.highlight_circles[agent_id]['circle'] is not None:
                self.highlight_circles[agent_id]['circle'].remove()
            del self.highlight_circles[agent_id]

    def clear_all_highlights(self):
        """Remove all highlight circles."""
        for agent_id in list(self.highlight_circles.keys()):
            self.clear_highlight_agent(agent_id)

    def initialize(self):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 12))
        self.ax = self.fig.add_subplot(111)

        extent = (
            self.origin['x'],
            self.origin['x'] + self.occupancy_grid.shape[1] * self.resolution,
            self.origin['y'],
            self.origin['y'] + self.occupancy_grid.shape[0] * self.resolution,
        )

        occupancy_to_show = self.occupancy_grid
        extent_to_use = extent

        if self.render_type == 'nav':
            # Show original map first (base layer)
            self.ax.imshow(1.0 - occupancy_to_show, cmap='gray', origin='lower', extent=extent_to_use, zorder=0)
            
            # Overlay dilated map with transparency
            # if hasattr(self.env, 'occupancy_grid_dilated') and self.env.occupancy_grid_dilated is not None:
                # Create a colored overlay for dilated areas
                # dilated_overlay = self.env.occupancy_grid_dilated - occupancy_to_show
                # dilated_overlay = np.clip(dilated_overlay, 0, 1)  # Ensure values are in [0,1]
                
                # # Create RGBA overlay: Red for dilated areas, transparent elsewhere
                # overlay_rgba = np.zeros((*dilated_overlay.shape, 4))
                # overlay_rgba[:, :, 0] = dilated_overlay  # Red channel
                # overlay_rgba[:, :, 3] = dilated_overlay * 0.3  # Alpha channel (30% transparency)
                
                # self.ax.imshow(overlay_rgba, origin='lower', extent=extent_to_use, zorder=0.5)
                
                # # Add a legend to explain the overlay
                # from matplotlib.patches import Patch
                # legend_elements = [
                #     Patch(facecolor='black', label='Obstacles'),
                #     Patch(facecolor='white', label='Free Space'),
                #     Patch(facecolor='red', alpha=0.3, label=f'Safety Zone (+{self.env.dilation_distance}m)')
                # ]
                # self.ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
            
        elif self.render_type == 'explore':
            exploration_to_show = self.env.exploration_map
            inverted_map = np.where(exploration_to_show == -1, 0.5, 1 - exploration_to_show)
            self.map_display = self.ax.imshow(
                inverted_map,
                cmap='gray',
                origin='lower',
                extent=extent_to_use,
                vmin=0,
                vmax=1,
                zorder=0
            )
            
            # Also show dilated overlay for exploration mode
            if hasattr(self.env, 'occupancy_grid_dilated') and self.env.occupancy_grid_dilated is not None:
                dilated_overlay = self.env.occupancy_grid_dilated - self.env.occupancy_grid
                dilated_overlay = np.clip(dilated_overlay, 0, 1)
                
                overlay_rgba = np.zeros((*dilated_overlay.shape, 4))
                overlay_rgba[:, :, 0] = dilated_overlay
                overlay_rgba[:, :, 3] = dilated_overlay * 0.2  # Lower transparency for exploration
                
                self.ax.imshow(overlay_rgba, origin='lower', extent=extent_to_use, zorder=0.5)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axis('off')

        if self.plot_limits is not None:
            self.ax.set_xlim(self.plot_limits[0], self.plot_limits[1])
            self.ax.set_ylim(self.plot_limits[2], self.plot_limits[3])
        else:
            self.ax.set_xlim(extent_to_use[0], extent_to_use[1])
            self.ax.set_ylim(extent_to_use[2], extent_to_use[3])

        self.ax.set_aspect('equal', 'box')

        # Initialize paths and old paths
        self.paths = [None] * len(self.swarm.agents)
        self.old_paths = [None] * len(self.swarm.agents)
        
        # Initialize danger regions
        self.danger_region_patches = []
        
        # Initialize sector grid lines and patches
        self.sector_lines = []
        self.sector_patches = []
        self.sector_texts = []  # Add this for sector name labels
        
        # Initialize sector grid visualization immediately (only once since it's static)
        self.initialize_sector_grid()
        
        # Force a refresh to ensure sectors are visible
        if hasattr(self.env, 'sector_grid') and self.env.sector_grid and self.sector_patches:
            plt.draw()

        # Initialize battery plot if enabled
        if self.show_battery_plot:
            self.initialize_battery_plot()

    def initialize_battery_plot(self):
        """Initialize the battery monitoring plot window."""
        self.battery_fig = plt.figure(figsize=(12, 8))
        self.battery_ax = self.battery_fig.add_subplot(111)
        
        self.battery_ax.set_title('UAV Battery Levels Over Time', fontsize=16)
        self.battery_ax.set_xlabel('Time (steps)', fontsize=12)
        self.battery_ax.set_ylabel('Battery Level', fontsize=12)
        self.battery_ax.set_ylim(0, 1)
        self.battery_ax.grid(True, alpha=0.3)
        
        # Initialize battery data storage for each UAV (only discoverable ones)
        uavs = [agent for agent in self.swarm.agents 
               if agent.type == 'UAV' and getattr(agent, 'discoverability', True)]
        for uav in uavs:
            self.battery_data[uav.agent_id] = []
            line, = self.battery_ax.plot([], [], 'o-', linewidth=2, markersize=4, 
                                       label=f'UAV {uav.agent_id}')
            self.battery_lines[uav.agent_id] = line
        
        self.battery_ax.legend()
        
        # Position the battery plot window
        mngr = self.battery_fig.canvas.manager
        mngr.window.wm_geometry("+100+100")  # Position at (100, 100)

    def update_markers(self):
        # Create a mapping from agent_id to agent for easy lookup
        agent_dict = {agent.agent_id: agent for agent in self.swarm.agents}
        
        # Remove markers for agents that no longer exist
        existing_agent_ids = set(agent_dict.keys())
        markers_to_remove = []
        
        if not hasattr(self, 'agent_marker_dict'):
            self.agent_marker_dict = {}
            self.agent_paths_dict = {}
            self.agent_old_paths_dict = {}
            self.agent_velocity_arrows_dict = {}
            self.agent_id_texts_dict = {}
            self.agent_battery_circles_dict = {}
            self.agent_comm_radius_circles_dict = {}
        
        # Remove markers for deleted agents
        for agent_id in list(self.agent_marker_dict.keys()):
            if agent_id not in existing_agent_ids:
                # Remove marker
                if self.agent_marker_dict[agent_id] is not None:
                    self.agent_marker_dict[agent_id].remove()
                del self.agent_marker_dict[agent_id]
                
                # Remove path
                if agent_id in self.agent_paths_dict and self.agent_paths_dict[agent_id] is not None:
                    self.agent_paths_dict[agent_id].remove()
                    del self.agent_paths_dict[agent_id]
                
                # Remove old path
                if agent_id in self.agent_old_paths_dict and self.agent_old_paths_dict[agent_id] is not None:
                    self.agent_old_paths_dict[agent_id].remove()
                    del self.agent_old_paths_dict[agent_id]
                
                # Remove velocity arrow
                if agent_id in self.agent_velocity_arrows_dict and self.agent_velocity_arrows_dict[agent_id] is not None:
                    self.agent_velocity_arrows_dict[agent_id].remove()
                    del self.agent_velocity_arrows_dict[agent_id]
                
                # Remove ID text
                if agent_id in self.agent_id_texts_dict and self.agent_id_texts_dict[agent_id] is not None:
                    self.agent_id_texts_dict[agent_id].remove()
                    del self.agent_id_texts_dict[agent_id]
                
                # Remove battery circle
                if agent_id in self.agent_battery_circles_dict and self.agent_battery_circles_dict[agent_id] is not None:
                    self.agent_battery_circles_dict[agent_id].remove()
                    del self.agent_battery_circles_dict[agent_id]
                
                # Remove communication radius circle
                if agent_id in self.agent_comm_radius_circles_dict and self.agent_comm_radius_circles_dict[agent_id] is not None:
                    self.agent_comm_radius_circles_dict[agent_id].remove()
                    del self.agent_comm_radius_circles_dict[agent_id]

        # Update/create markers for existing agents (only if discoverable)
        for agent in self.swarm.agents:
            # Skip agents that are not discoverable
            if not getattr(agent, 'discoverability', True):  # Default to True if attribute doesn't exist
                continue
                
            agent_id = agent.agent_id
            style = self.type_styles.get(agent.type, {'cmap': 'Greys', 'marker': 'o'})
            cmap = plt.cm.get_cmap(style['cmap'])

            # Transform agent position
            x, y = agent.state[0], agent.state[1]

            # Determine color
            if agent.type == 'UAV':
                if agent.controller_mode == 'perched':
                    color = 'red'  
                elif agent.controller_mode == 'perch_seek':
                    color = 'yellow'
                else:
                    color = cmap(agent.battery) if agent.battery is not None else mcolors.to_rgb(style['cmap'])
            
            elif agent.type == 'perching_pos':
                # Color perching positions: red if occupied, yellow if reserved, green if free
                if getattr(agent, 'occupied_by', None) is not None:
                    color = 'red'  # Occupied
                elif getattr(agent, 'reserved_by', None) is not None:
                    color = 'yellow'  # Reserved but not occupied yet
                else:
                    color = 'green'  # Free
            else:
                color = cmap(0.5)

            # Determine marker size and z-order
            if agent.type == 'perching_pos':
                markersize = 20.0 
            # elif agent.type == 'base':
            #     markersize = 22.0
            else:
                markersize = 16

            zorder = 4 if agent.type == 'perching_pos' else 5  # Higher than before

            # Update or create marker
            if agent_id not in self.agent_marker_dict or self.agent_marker_dict[agent_id] is None:
                marker, = self.ax.plot(
                    [x], [y], style['marker'],
                    mfc=color, mec='black', markersize=markersize, zorder=zorder
                )
                self.agent_marker_dict[agent_id] = marker
            else:
                self.agent_marker_dict[agent_id].set_xdata(x)
                self.agent_marker_dict[agent_id].set_ydata(y)
                self.agent_marker_dict[agent_id].set_markerfacecolor(color)

            # Handle UAV ID text
            if agent.type == 'UAV' or agent.type == 'robot' or agent.type == 'perching_pos':
                # Remove previous text if it exists
                if agent_id in self.agent_id_texts_dict and self.agent_id_texts_dict[agent_id] is not None:
                    self.agent_id_texts_dict[agent_id].remove()
                
                # NO MORE COORDINATE TRANSFORMATION
                text_x, text_y = agent.state[0] + 0.085, agent.state[1] + 0.085
                
                # Add new text
                self.agent_id_texts_dict[agent_id] = self.ax.text(
                    text_x, text_y, str(agent.agent_id),
                    fontsize=20, color='black', zorder=20, 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1')
                )
            elif agent_id in self.agent_id_texts_dict and self.agent_id_texts_dict[agent_id] is not None:
                self.agent_id_texts_dict[agent_id].remove()
                self.agent_id_texts_dict[agent_id] = None

            # Update communication/visibility radius circle
            self.update_comm_radius_circle(agent, agent_id, x, y)

            # Draw velocity arrow for each agent
            # if hasattr(agent, 'state') and len(agent.state) >= 4:
            #     x, y, vx, vy = agent.state[0], agent.state[1], agent.state[2], agent.state[3]
            #     # Remove previous arrow if it exists
            #     if self.velocity_arrows[i] is not None:
            #         self.velocity_arrows[i].remove()
            #         self.velocity_arrows[i] = None
            #     norm = np.hypot(vx, vy)
            #     if norm > 1e-3:
            #         scale = 0.25  # Make arrow longer for visibility
            #         dx, dy = (vx / norm) * scale, (vy / norm) * scale
            #         arrow = self.ax.arrow(
            #             x, y, dx, dy,
            #             head_width=0.125, head_length=0.125, fc='orange', ec='orange',
            #             length_includes_head=True, zorder=10, alpha=0.8
            #         )
            #         self.velocity_arrows[i] = arrow

    def update_comm_radius_circle(self, agent, agent_id, x, y):
        """Update communication/visibility radius circle for an agent."""
        # Only show radius for robot, base, and perching_pos agent types
        if agent.type in ['robot', 'base', 'UAV'] and self.swarm.vis_radius > 0:
            # Create or update communication radius circle using swarm vis_radius
            if agent_id not in self.agent_comm_radius_circles_dict or self.agent_comm_radius_circles_dict[agent_id] is None:
                # Create new circle
                circle = plt.Circle(
                    (x, y),
                    self.swarm.vis_radius,
                    edgecolor='yellow',
                    facecolor='yellow',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.1,
                    zorder=1  # Low z-order to appear behind agents
                )
                self.ax.add_patch(circle)
                self.agent_comm_radius_circles_dict[agent_id] = circle
            else:
                # Update existing circle position
                self.agent_comm_radius_circles_dict[agent_id].center = (x, y)
        else:
            # Remove circle if agent type doesn't match or no vis_radius
            if agent_id in self.agent_comm_radius_circles_dict and self.agent_comm_radius_circles_dict[agent_id] is not None:
                self.agent_comm_radius_circles_dict[agent_id].remove()
                self.agent_comm_radius_circles_dict[agent_id] = None

    def update_adjacency_lines(self):
        # Clear existing lines
        for line in self.adjacency_lines:
            line.remove()
        self.adjacency_lines.clear()
        
        # Draw perching edges FIRST (orange) with LOWER zorder
        perch_edges = {('UAV', 'perching_pos'), ('perching_pos', 'UAV')}
        perch_adjacency = self.swarm.compute_adjacency_matrix(allowed_edge_types=perch_edges)
        
        perch_allowed_types = set()
        for edge_type in perch_edges:
            perch_allowed_types.update(edge_type)
        # Only include discoverable agents
        perch_agents = [agent for agent in self.swarm.agents 
                       if agent.type in perch_allowed_types and getattr(agent, 'discoverability', True)]
        perch_positions = [agent.state[:2] for agent in perch_agents]
        
        # NO MORE TRANSFORM - DIRECT POSITIONS
        self._draw_edges(perch_positions, perch_adjacency, color='orange', alpha=0.7, linewidth=1.5, linestyle='-', label='Perching', zorder=1)
        
        # Draw communication edges SECOND (green) with HIGHER zorder
        comm_edges = {('UAV', 'UAV'), ('robot', 'UAV'), ('base', 'UAV'), ('UAV', 'robot'), ('UAV', 'base')}
        comm_adjacency = self.swarm.compute_adjacency_matrix(allowed_edge_types=comm_edges)
        
        comm_allowed_types = set()
        for edge_type in comm_edges:
            comm_allowed_types.update(edge_type)
        # Only include discoverable agents
        comm_agents = [agent for agent in self.swarm.agents 
                      if agent.type in comm_allowed_types and getattr(agent, 'discoverability', True)]
        comm_positions = [agent.state[:2] for agent in comm_agents]
        
        # NO MORE TRANSFORM - DIRECT POSITIONS
        self._draw_edges(comm_positions, comm_adjacency, color='lime', alpha=0.8, linewidth=2.0, linestyle='-', label='Communication', zorder=2)

    def _draw_edges(self, positions, adjacency_matrix, color, alpha, linewidth, linestyle, label, zorder=1):
        """Helper method to draw edges with specified style."""
        first_edge = True
        
        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if (i < len(adjacency_matrix) and j < len(adjacency_matrix[0]) and 
                    adjacency_matrix[i, j] and i != j):  # Don't draw self-loops
                    
                    edge_label = label if first_edge else ""
                    line, = self.ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]],
                                       color=color, zorder=zorder, alpha=alpha,  # USE THE ZORDER PARAMETER
                                       linewidth=linewidth, linestyle=linestyle, 
                                       label=edge_label)
                    self.adjacency_lines.append(line)
                    first_edge = False

    def update_paths(self):
        """Update paths using agent_id mapping instead of indices."""
        if not hasattr(self, 'agent_paths_dict'):
            self.agent_paths_dict = {}
        
        # Get current agent IDs
        current_agent_ids = {agent.agent_id for agent in self.swarm.agents}
        
        # Remove paths for deleted agents
        for agent_id in list(self.agent_paths_dict.keys()):
            if agent_id not in current_agent_ids:
                if self.agent_paths_dict[agent_id] is not None:
                    self.agent_paths_dict[agent_id].remove()
                del self.agent_paths_dict[agent_id]
        
        # Update paths for existing agents (only if discoverable)
        for agent in self.swarm.agents:
            # Skip agents that are not discoverable
            if not getattr(agent, 'discoverability', True):
                continue
                
            agent_id = agent.agent_id
            
            # Check if agent has a path and hasn't completed it yet
            if (agent.path is not None and 
                hasattr(agent, 'path_idx') and hasattr(agent, 'path_len') and
                agent.path_idx < agent.path_len):
                
                path_x = [p[0] for p in agent.path]
                path_y = [p[1] for p in agent.path]
                
                # Set path color based on agent type
                if agent.type == 'robot':
                    path_color = 'orange'
                else:  # UAV or other types
                    path_color = 'blue'

                if agent_id in self.agent_paths_dict and self.agent_paths_dict[agent_id] is not None:
                    self.agent_paths_dict[agent_id].set_xdata(path_x)
                    self.agent_paths_dict[agent_id].set_ydata(path_y)
                    self.agent_paths_dict[agent_id].set_color(path_color)
                else:
                    path_line, = self.ax.plot(path_x, path_y, color=path_color, linestyle='--', linewidth=2, alpha=0.6, zorder=3)
                    self.agent_paths_dict[agent_id] = path_line
            else:
                # Remove path if agent has no path, completed its path, or doesn't have path tracking
                if agent_id in self.agent_paths_dict and self.agent_paths_dict[agent_id] is not None:
                    self.agent_paths_dict[agent_id].remove()
                    self.agent_paths_dict[agent_id] = None

    def update_old_paths(self):
        """
        Updates or creates lines for the old paths of each agent, 
        with arrows showing the direction of movement at fixed intervals.
        """

        for i, agent in enumerate(self.swarm.agents):
            if agent.old_path is not None and len(agent.old_path) > 1 and agent.battery > 0.85:
                # Extract x and y coordinates
                old_path_x = [p[0] for p in agent.old_path]
                old_path_y = [p[1] for p in agent.old_path]

                # Update the existing dashed line for the path
                if self.old_paths[i]:
                    self.old_paths[i].set_data(old_path_x, old_path_y)
                else:
                    old_path_line, = self.ax.plot(
                        old_path_x, old_path_y, color='blue', linestyle='--', linewidth=3.0, alpha=0.4, zorder=1
                    )
                    self.old_paths[i] = old_path_line

                if len(agent.old_path) % 6 == 0:
                    dx = old_path_x[-1] - old_path_x[-2]
                    dy = old_path_y[-1] - old_path_y[-2]

                    self.ax.quiver(
                        old_path_x[-2], old_path_y[-2], dx, dy,
                        angles='xy', scale_units='xy', scale=0.6, color='blue', alpha=0.05, zorder=2
                    )

    def update_battery_circles(self):
        """Update battery circles using agent_id mapping instead of indices."""
        if not hasattr(self, 'agent_battery_circles_dict'):
            self.agent_battery_circles_dict = {}
        
        # Get current agent IDs
        current_agent_ids = {agent.agent_id for agent in self.swarm.agents}
        
        # Remove circles for deleted agents
        for agent_id in list(self.agent_battery_circles_dict.keys()):
            if agent_id not in current_agent_ids:
                if self.agent_battery_circles_dict[agent_id] is not None:
                    self.agent_battery_circles_dict[agent_id].remove()
                del self.agent_battery_circles_dict[agent_id]
        
        # Update circles for existing agents (only if discoverable)
        for agent in self.swarm.agents:
            if agent.type != 'UAV':  # Only show battery circles for UAVs
                continue
                
            # Skip agents that are not discoverable
            if not getattr(agent, 'discoverability', True):
                continue
                
            agent_id = agent.agent_id
            
            if agent.battery < self.swarm.add_agent_params['battery_of_concern']:
                color = 'red'
            elif agent.battery is not None and agent.battery > 0.85:
                color = 'green'
            else:
                color = None

            if color:
                if agent_id not in self.agent_battery_circles_dict or self.agent_battery_circles_dict[agent_id] is None:
                    # Create a new circle
                    circle = plt.Circle(
                        (agent.state[0], agent.state[1]),
                        0.2,
                        edgecolor=color,
                        facecolor='none',
                        linestyle='dashed',
                        linewidth=2,
                        zorder=2,
                        alpha=0.8
                    )
                    self.ax.add_patch(circle)
                    self.agent_battery_circles_dict[agent_id] = circle
                else:
                    # Update the existing circle
                    self.agent_battery_circles_dict[agent_id].center = (agent.state[0], agent.state[1])
                    self.agent_battery_circles_dict[agent_id].set_edgecolor(color)
            else:
                # Remove the circle if it exists and no longer needed
                if agent_id in self.agent_battery_circles_dict and self.agent_battery_circles_dict[agent_id] is not None:
                    self.agent_battery_circles_dict[agent_id].remove()
                    self.agent_battery_circles_dict[agent_id] = None

    def update_exploration_map(self):
        # Update the map display with the current inverted exploration map
        inverted_map = np.where(self.env.exploration_map == -1, 0.5, 1 - self.env.exploration_map)
        self.map_display.set_data(inverted_map)

    def update_battery_plot(self):
        """Update the battery plot with current battery levels."""
        if not self.show_battery_plot or self.battery_fig is None:
            return
            
        # Check if battery window was closed
        if not plt.fignum_exists(self.battery_fig.number):
            return
            
        self.current_time += 1
        self.time_data.append(self.current_time)
        
        # Update battery data for each UAV (only if discoverable)
        uavs = [agent for agent in self.swarm.agents 
               if agent.type == 'UAV' and getattr(agent, 'discoverability', True)]
        for uav in uavs:
            if uav.agent_id not in self.battery_data:
                # New UAV added during simulation - fill with NaN for missing time steps
                self.battery_data[uav.agent_id] = [float('nan')] * (len(self.time_data) - 1)
                line, = self.battery_ax.plot([], [], 'o-', linewidth=2, markersize=4, 
                                           label=f'UAV {uav.agent_id}')
                self.battery_lines[uav.agent_id] = line
                self.battery_ax.legend()
            
            self.battery_data[uav.agent_id].append(uav.battery)
            
            # Update the line data - ensure arrays are same length
            battery_data = self.battery_data[uav.agent_id]
            time_data_for_uav = self.time_data[-len(battery_data):]  # Match lengths
            self.battery_lines[uav.agent_id].set_data(time_data_for_uav, battery_data)
        
        # Remove data for UAVs that no longer exist
        current_uav_ids = {uav.agent_id for uav in uavs}
        for uav_id in list(self.battery_data.keys()):
            if uav_id not in current_uav_ids:
                # UAV was removed, keep its data but mark it as inactive
                self.battery_lines[uav_id].set_alpha(0.3)  # Make line semi-transparent
        
        # Auto-scale the plot
        if self.time_data:
            self.battery_ax.set_xlim(max(0, self.current_time - 200), self.current_time + 10)
        
        # Add battery removal threshold line
        if hasattr(self.swarm, 'battery_removal_threshold'):
            threshold = getattr(self.swarm, 'battery_removal_threshold', 0.1)
            self.battery_ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                                  label='Removal Threshold')
        
        # Refresh the battery plot
        self.battery_fig.canvas.draw()

    def update_highlight_circle(self):
        """Update all highlight circles around the hardcoded agent IDs (only if discoverable)."""
        for agent_id, highlight_info in self.highlight_circles.items():
            # Remove existing circle if it exists
            if highlight_info['circle'] is not None:
                highlight_info['circle'].remove()
                highlight_info['circle'] = None
            
            # Find the agent with the specified ID
            target_agent = None
            for agent in self.swarm.agents:
                if agent.agent_id == agent_id:
                    target_agent = agent
                    break
            
            # If agent not found or not discoverable, skip this circle
            if target_agent is None or not getattr(target_agent, 'discoverability', True):
                continue
            
            # Create the colored dashed circle
            circle = plt.Circle(
                (target_agent.state[0], target_agent.state[1]),
                highlight_info['radius'],
                edgecolor=highlight_info['color'],
                facecolor='none',
                linestyle='--',
                linewidth=3,
                zorder=10,  # High z-order to appear on top
                alpha=0.6
            )
            
            self.ax.add_patch(circle)
            highlight_info['circle'] = circle

    def update_danger_regions(self):
        """Update the visualization of danger regions."""
        # Remove existing danger region patches
        for patch in self.danger_region_patches:
            if patch in self.ax.patches:
                patch.remove()
        self.danger_region_patches.clear()
        
        # Add current danger regions
        if hasattr(self.swarm, 'danger_regions'):
            from matplotlib.patches import Rectangle
            
            for region in self.swarm.danger_regions:
                # Create a red rectangle with transparency
                rect = Rectangle(
                    (region['x_min'], region['y_min']),
                    region['width'], 
                    region['height'],
                    linewidth=2,
                    edgecolor='darkred',
                    facecolor='red',
                    alpha=0.3,  # Semi-transparent
                    zorder=1  # Above map but below agents
                )
                
                self.ax.add_patch(rect)
                self.danger_region_patches.append(rect)
                
                # Add simple text label for the region
                center_x, center_y = region['center']
                # Extract just the number/ID part if it contains "danger_region_"
                region_id = region['id']
                if region_id.startswith('danger_region_'):
                    display_text = region_id.split('_')[-1]  # Get just the number
                else:
                    display_text = region_id
                
                text = self.ax.text(
                    center_x, center_y, 
                    display_text, 
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white',
                    zorder=2
                )
                self.danger_region_patches.append(text)

    def render(self):
        if self.fig is None:
            self.initialize()
        self.update_markers()
        self.update_adjacency_lines()
        self.update_paths()
        # self.update_highlight_circle()  # Add this line
        self.update_danger_regions()  # Add danger region rendering
        # self.update_old_paths()
        # self.update_battery_circles()
        if self.render_type == 'explore':
            self.update_exploration_map()
        
        # Update battery plot
        self.update_battery_plot()
        
        plt.figure(self.fig.number)  # Switch back to main plot
        plt.draw()
        plt.pause(0.01)

    def initialize_sector_grid(self):
        """Initialize the visualization of the sector grid with colored shading and labels (once only)."""
        # Add sector grid if available
        if hasattr(self.env, 'sector_grid') and self.env.sector_grid:
            sector_grid = self.env.sector_grid
            print(f"Initializing {len(sector_grid['sectors'])} sectors")  # Debug print
            
            # Define colors for alternating pattern (brighter, more visible colors)
            colors = [
                (0.5, 0.5, 1.0),    # Brighter blue
                (1.0, 0.5, 0.5),    # Brighter red
                (0.5, 1.0, 0.5),    # Brighter green
                (1.0, 1.0, 0.5),    # Brighter yellow
                (1.0, 0.5, 1.0),    # Brighter magenta
                (0.5, 1.0, 1.0),    # Brighter cyan
            ]
            
            # Add colored sector patches (background)
            from matplotlib.patches import Rectangle
            for i, sector in enumerate(sector_grid['sectors']):
                # Choose color based on sector position for a nice pattern
                color_index = (sector['col'] + sector['row']) % len(colors)
                color = colors[color_index]
                
                # Create rectangle patch for each sector
                width = sector['x_max'] - sector['x_min']
                height = sector['y_max'] - sector['y_min']
                
                rect = Rectangle(
                    (sector['x_min'], sector['y_min']),
                    width,
                    height,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1.0,
                    alpha=0.1,  # More visible alpha
                    zorder=0.5,  # Above map background but below agents
                )
                
                self.ax.add_patch(rect)
                self.sector_patches.append(rect)
                
                if i < 5:  # Debug print for first few sectors
                    sector_name = sector.get('name', f"S{sector['id']}")
                    print(f"  Sector {i}: {sector_name} at ({sector['x_min']:.2f}, {sector['y_min']:.2f}) "
                          f"to ({sector['x_max']:.2f}, {sector['y_max']:.2f}) "
                          f"color_idx={color_index} color={color}")
            
            # Add chessboard-style labels outside the grid
            self._add_chessboard_labels(sector_grid)
            
            print(f"✅ Successfully created {len(self.sector_patches)} sector patches and chessboard labels")
        else:
            print("❌ No sector grid found in environment")  # Debug print

    def _add_chessboard_labels(self, sector_grid):
        """Add chessboard-style labels (column letters and row numbers) outside the grid."""
        sectors_x = sector_grid['sectors_x']
        sectors_y = sector_grid['sectors_y']
        sector_size = sector_grid['sector_size']
        
        # Get the map bounds
        sectors = sector_grid['sectors']
        if not sectors:
            return
            
        # Calculate grid boundaries
        x_min = min(s['x_min'] for s in sectors)
        x_max = max(s['x_max'] for s in sectors)
        y_min = min(s['y_min'] for s in sectors)
        y_max = max(s['y_max'] for s in sectors)
        
        # Add column labels (letters) at the bottom of the grid
        for col in range(sectors_x):
            col_letter = chr(ord('a') + col)
            x_center = x_min + (col + 0.5) * sector_size
            
            # Place label below the grid
            text = self.ax.text(
                x_center, y_min - sector_size * 0.2,  # Below the grid
                col_letter,
                ha='center', va='center',
                fontsize=26,
                color='black',
                # bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
                zorder=10  # High z-order to appear on top
            )
            self.sector_texts.append(text)
        
        # Add row labels (numbers) at the left of the grid
        for row in range(sectors_y):
            row_number = row + 1
            y_center = y_min + (row + 0.5) * sector_size
            
            # Place label to the left of the grid
            text = self.ax.text(
                x_min - sector_size * 0.2, y_center,  # Left of the grid
                str(row_number),
                ha='center', va='center',
                fontsize=26,
                color='black',
                # bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
                zorder=10  # High z-order to appear on top
            )
            self.sector_texts.append(text)
