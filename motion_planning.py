import argparse
import time
import msgpack
import utm
import random
from enum import Enum, auto
import networkx as nx
import numpy as np
import numpy.linalg as LA

from planning_utils import a_star, heuristic, create_grid_graph, G_weight, prune_path, closest_point
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        lat0, lon0= np.loadtxt('colliders.csv', delimiter=',', dtype='str', usecols = (0,1))[0]
        lat0 = float(lat0.split()[1])
        lon0 = float(lon0.split()[1])
        
        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)
        global_home = self.global_home

        # TODO: retrieve current global position
        global_position = self.global_position
 
        # TODO: convert to current local position using global_to_local()
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, 
                                                               self.global_position,self.local_position))
        local_position = global_to_local(global_position, global_home)
        
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, edges, north_offset, east_offset = create_grid_graph(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        
        # Define starting point on the grid (this is just grid center)
        # grid_start = (-north_offset, -east_offset)

        # TODO: convert start position to current position rather than map center
        grid_start = (local_position[0]-north_offset, local_position[1]-east_offset)
        
        # Set goal as some arbitrary position on the grid
        # grid_goal = (-north_offset + 10, -east_offset + 10)
        
        # TODO: adapt to set goal as latitude / longitude position and convert
        goal_global_position = (-122.39827005, 37.79639587, 0)
        goal_local_position = global_to_local(goal_global_position, global_home)
        grid_goal = (int(goal_local_position[0] - north_offset),int(goal_local_position[1] - east_offset))
        print('Local Start and Goal: ', grid_start, grid_goal)
        
        # Add weight to graph
        G = nx.Graph()
        for e in edges:
            p1 = tuple(e[0])
            p2 = tuple(e[1])
            dist = LA.norm(np.array(p2) - np.array(p1))
            G.add_edge(p1, p2, weight=dist)
        
        # Find the closest point in the graph to our current location, same thing for the goal location.
        start_g = closest_point(G, grid_start)
        goal_g = closest_point(G, grid_goal)
        print('Local Start and Goal in graph: ', start_g, goal_g)

        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        path, cost = a_star(G, heuristic, start_g, goal_g)
            
        # TODO: prune path to minimize number of waypoints
        # TODO (if you're feeling ambitious): Try a different approach altogether!
        pruned_path = prune_path(path)
        pruned_path.append(grid_goal)
        pruned_path.insert(0, grid_start)

        # Convert path to waypoints
        waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in pruned_path]
        
#         waypoints = [[0, 0, 5, 0], [0, 1, 5, 0], [1, 1, 5, 0], [1, 2, 5, 0], [2, 2, 5, 0], [2, 3, 5, 0], [3, 3, 5, 0], [3, 4, 5, 0], [4, 4, 5, 0], [4, 5, 5, 0], [5, 5, 5, 0], [5, 6, 5, 0], [6, 6, 5, 0], [6, 7, 5, 0], [7, 7, 5, 0], [7, 8, 5, 0], [8, 8, 5, 0], [8, 9, 5, 0], [9, 9, 5, 0], [9, 10, 5, 0], [10, 10, 5, 0]]
        
#         waypoints = [[10, 10, 5, 0], [7, 0, 5, 0], [26, 14, 5, 0], [32, 17, 5, 0], [34, 15, 5, 0], [44, 5, 5, 0], [54, -1, 5, 0], [64, -6, 5, 0], [84, -11, 5, 0], [93, -15, 5, 0], [94, -16, 5, 0], [98, -17, 5, 0], [104, -24, 5, 0], [129, -39, 5, 0], [149, -19, 5, 0], [154, -16, 5, 0], [174, -26, 5, 0], [184, -29, 5, 0], [204, -29, 5, 0], [204, -29, 5, 0], [212, -29, 5, 0], [214, -30, 5, 0], [224, -34, 5, 0], [244, -34, 5, 0], [254, -37, 5, 0], [258, -40, 5, 0], [274, -44, 5, 0], [279, -41, 5, 0], [294, -44, 5, 0], [314, -44, 5, 0], [324, -46, 5, 0], [327, -45, 5, 0], [344, -49, 5, 0], [362, -49, 5, 0], [364, -50, 5, 0], [374, -54, 5, 0], [397, -54, 5, 0], [404, -62, 5, 0], [407, -67, 5, 0], [409, -74, 5, 0], [412, -81, 5, 0], [428, -82, 5, 0], [432, -80, 5, 0], [434, -75, 5, 0]]
        
        # Set self.waypoints
        self.waypoints = waypoints
        print (waypoints)
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    parser.add_argument('--lat', type=float, default=1000, help="latitude")
    parser.add_argument('--lon', type=float, default=1000, help="latitude")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()