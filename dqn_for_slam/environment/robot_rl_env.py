from typing import Tuple
import logging
import time
import os

import gym
from gym import spaces
import numpy as np
import rospy
import rosparam
from nav_msgs.srv import GetMap
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, Point, Quaternion
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

# from gmapping.srv import SlamCmd
from .. import rl_worker


file_path = __file__
dir_path = file_path[:(len(file_path) - len('environment/robot_rl_env.py'))] + 'config/'
config_file_name = 'rlslam_map_reward.yaml'
config_file_path = os.path.join(dir_path, config_file_name)
parameters_list=rosparam.load_file(config_file_path)
for params, namespace in parameters_list:
	rosparam.upload_params(namespace,params)

INITIAL_POS_X = rospy.get_param('rlslam/initial_posx')
INITIAL_POS_Y = rospy.get_param('rlslam/initial_posy')

LIDAR_SCAN_MAX_DISTANCE = rospy.get_param('rlslam/scan_max_distance')
TRAINING_IMAGE_SIZE = rospy.get_param('rlslam/training_image_size')
MAP_SIZE_RATIO = rospy.get_param('rlslam/map_size_ratio')
MAP_COMPLETENESS_THRESHOLD = rospy.get_param('rlslam/map_completed_threshold')
COLLISION_THRESHOLD = rospy.get_param('rlslam/crash_distance')

REWARD_MAP_COMPLETED = rospy.get_param('rlslam/reward_map_completed')
REWARD_CRASHED = rospy.get_param('rlslam/reward_crashed')

MAX_PX = rospy.get_param('rlslam/obs_space_max/px')
MAX_PY = rospy.get_param('rlslam/obs_space_max/py')
MAX_QZ = rospy.get_param('rlslam/obs_space_max/qz')
MAX_ACTION_NUM = 3
MAX_MAP_COMPLETENESS = 100.
MAX_STEPS = rospy.get_param('rlslam/steps_in_episode')

MIN_PX = rospy.get_param('rlslam/obs_space_min/px')
MIN_PY = rospy.get_param('rlslam/obs_space_min/py')
MIN_QZ = rospy.get_param('rlslam/obs_space_min/qz')
MIN_ACTION_NUM = -1
MIN_STEPS = 0
MIN_MAP_COMPLETENESS = 0.

STEERING = rospy.get_param('rlslam/steering')
THROTTLE = rospy.get_param('rlslam/throttle') 

TIMEOUT = rospy.get_param('rlslam/timeout')


class RobotEnv(gym.Env):
    """
    Environment for reinforce learning
    -> generally it defines
        action
        status that responds to the action
    -> also it defines
        reward for a status
    Each valuables for the state is updated by ros
    it gets information of state and run somethings with them
    -> it don't define functions like "can_move_at" or "transit" or so
    -> it defines functions like "reward"
    """

    def __init__(self) -> None:

        self.position = Point(INITIAL_POS_X, INITIAL_POS_Y, 0)
        self.orientation = Quaternion(1, 0, 0, 0)
        self.ranges = None
        self.map_completeness_pct = MIN_MAP_COMPLETENESS
        self.occupancy_grid = None
        self.done = False
        self.steps_in_episode = 0
        self.min_distance = 100
        self.reward = None
        self.reward_in_episode = 0
        self.now_action = -1
        self.last_action = -1
        self.last_map_completeness_pct = 0
        self.next_state = None

        # define action space
        # steering(angle) is (-1, 1), throttle(speed) is (0, 1)
        # self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([+1, +1]), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        # define observation space
        scan_high = np.array([LIDAR_SCAN_MAX_DISTANCE] * TRAINING_IMAGE_SIZE)
        scan_low = np.array([0.0] * TRAINING_IMAGE_SIZE)
        num_high = np.array([
            MAX_PX,
            MAX_PY,
            MAX_QZ,
            MAX_ACTION_NUM,
            MAX_STEPS,
            MAX_MAP_COMPLETENESS
        ])
        num_low = np.array([
            MIN_PX,
            MIN_PY,
            MIN_QZ,
            MIN_ACTION_NUM, 
            MIN_STEPS,
            MIN_MAP_COMPLETENESS
        ])
        high = np.concatenate([scan_high, num_high])
        low = np.concatenate([scan_low, num_low])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # ROS initialization
        self.ack_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=100)
        self.map_reset_service = rospy.ServiceProxy('/clear_map', Empty)
        self.gazebo_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        rospy.init_node('rl_dqn', anonymous=True)

    def reset(self) -> np.ndarray:
        """
        initiate status and  return the first observed values
        """
        rospy.loginfo('start resetting')

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            rospy.loginfo("/gazebo/unpause_physics service call failed")
        
        self._send_action(0, 0)  # stop robot moving
  
        # clear map
        rospy.wait_for_service('/clear_map')
        if self.map_reset_service():
            rospy.loginfo('reset map')
        else:
            rospy.logerr('could not reset map')
         
        self.done = False
        self.position = Point(INITIAL_POS_X, INITIAL_POS_Y, 0)
        self.orientation = Quaternion(1, 0, 0, 0)
        self.steps_in_episode = 0
        self.map_completeness_pct = 0
        self.last_map_completeness_pct = 0
        self.reward_in_episode = 0
        self.occupancy_grid = None
        self.next_state = None
        self.ranges = None

        self._rosbot_reset()  # initialize gazebo robot status
        self._update_map_completeness()
        self._update_state()
        self._infer_reward()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            rospy.loginfo("/gazebo/pause_physics service call failed")

        rospy.loginfo('succeess: reset')
        # TODO (Kuwabara): add process when self.next_stage is None
        return self.next_state

    def _rosbot_reset(self) -> None:
        """
        """
        rospy.wait_for_service('gazebo/set_model_state')

        model_state = ModelState()
        model_state.model_name = 'rosbot'
        model_state.pose.position.x = INITIAL_POS_X
        model_state.pose.position.y = INITIAL_POS_Y
        model_state.pose.position.z = 0
        model_state.pose.orientation.x = 0
        model_state.pose.orientation.y = 0
        model_state.pose.orientation.z = 0
        model_state.pose.orientation.w = 1
        model_state.twist.linear.x = 0
        model_state.twist.linear.y = 0
        model_state.twist.linear.z = 0
        model_state.twist.angular.x = 0
        model_state.twist.angular.y = 0
        model_state.twist.angular.z = 0

        try:
            self.gazebo_model_state_service(model_state)
        except (rospy.ServiceException) as e:
            rospy.loginfo("/gazebo/set_model_state service call failed")       
  
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        run action and return results
        """        
        # rospy.loginfo('start step' + str(self.steps_in_episode + 1))

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            rospy.loginfo("/gazebo/unpause_physics service call failed")
        
        self.last_action = self.now_action
        self.now_action = action
        if action == 0:  # turn left
            steering = STEERING
            throttle = THROTTLE
        elif action == 1:  # turn right
            steering = -1 * STEERING
            throttle = THROTTLE
        elif action == 2:  # straight
            steering = 0
            throttle = THROTTLE
        elif action == 3:  # backward
            steering = 0
            throttle = -1 * THROTTLE
        else:
            raise ValueError("Invalid action")

        # initialize rewards, next_state, done
        self.reward = None
        self.done = False
        self.next_state = None

        self.steps_in_episode += 1
        self._send_action(steering, throttle)
        self._update_map_completeness()
        self._update_state()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            rospy.loginfo("/gazebo/pause_physics service call failed")

        self._infer_reward()

        if self.steps_in_episode >= MAX_STEPS:
            rl_worker.add_map_completeness(self.map_completeness_pct)

        # TODO (Kuwabara): add process when self.next_stage or self.reward is None
        # rospy.loginfo('end step' + str(self.steps_in_episode))
        info = {}
        return self.next_state, self.reward, self.done, info

   
    def _update_state(self) -> None:
        """
        """
        rospy.loginfo('waiting lidar scan')
        # adapt number of sensor information to TRAINING_IMAGE_SIZE
        self.ranges = None
        while not self.ranges:
          try:
                data = rospy.wait_for_message('/scan_filtered', LaserScan, timeout= TIMEOUT)
                self.ranges = data.ranges
          except:
                pass
        rospy.loginfo('end waiting scan')
        size = len(self.ranges)
        x = np.linspace(0, size - 1, TRAINING_IMAGE_SIZE)
        xp = np.arange(size)
        sensor_state = np.clip(np.interp(x, xp, self.ranges), 0, LIDAR_SCAN_MAX_DISTANCE)
        sensor_state[np.isnan(sensor_state)] = LIDAR_SCAN_MAX_DISTANCE

        # update distance to obstacles
        self.min_distance = np.amin(sensor_state)

        rospy.loginfo('waiting odom')
        # adapt number of sensor information to TRAINING_IMAGE_SIZE
        self.position = None
        self.orientation = None
        while not self.position:
          try:
                data = rospy.wait_for_message('/odom', Odometry, timeout=TIMEOUT)
                self.position = data.pose.pose.position
                self.orientation = data.pose.pose.orientation
          except:
                pass
        rospy.loginfo('end waiting odom')

        numeric_state = np.array([
            self.position.x,
            self.position.y,
            self.orientation.z,
            self.last_action,
            self.steps_in_episode,
            self.map_completeness_pct
        ])

        self.next_state = np.concatenate([sensor_state, numeric_state])

    def _infer_reward(self) -> None:
        """
        """
        
        if self.map_completeness_pct > MAP_COMPLETENESS_THRESHOLD:
            self.reward = REWARD_MAP_COMPLETED
            state = 'comp'
        elif self.min_distance < COLLISION_THRESHOLD:
            # Robot likely hit the wall
            self.reward = REWARD_CRASHED
            state = 'crashed'
        else:
            self.reward = self.map_completeness_pct - self.last_map_completeness_pct
            state = ''
        
        self.reward_in_episode += self.reward
        rospy.loginfo('reward:' + str(self.reward) + ' ' + state)

    def _update_map_completeness(self) -> None:
        
        rospy.loginfo('waiting map')
        self.occupancy_grid = None
        while not self.occupancy_grid:
          try:
                data = rospy.wait_for_message('/map', OccupancyGrid, timeout=TIMEOUT)
                self.occupancy_grid = data.data
          except:
                pass
        
        rospy.loginfo('ended waiting map')

        sum_grid = len(self.occupancy_grid)
        num_occupied = 0
        num_unoccupied = 0
        num_negative = 0
        for n in self.occupancy_grid:
            if n == 0:
                num_unoccupied += 1
            elif n == 100:
                num_occupied += 1 
        
        self.last_map_completeness_pct = self.map_completeness_pct
        self.map_completeness_pct = ((num_occupied + num_unoccupied) * 100 / sum_grid) / MAP_SIZE_RATIO
        rospy.loginfo('map completenes:' + str(self.map_completeness_pct)) 
    
    def _send_action(self, steering: float, throttle: float) -> None:
        speed = Twist()
        speed.angular.z = steering
        speed.linear.x = throttle
        self.ack_publisher.publish(speed)

    def render(self, mode='human') -> None:
        """
        unused function
        """
        return
