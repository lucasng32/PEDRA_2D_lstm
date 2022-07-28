from map_2D.aux_funcs import *
from map_2D.networks.network_models_2D import initialize_network_FCQN
import matplotlib.pyplot as plt
from copy import deepcopy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class agent_2D:
    def __init__(self, BHM, min_max, LIDAR_pixel_range, ground_truth_map, starting_pos,
                 plot_dir, weights_dir, custom_load=None):

        self.BHM = BHM
        self.previous_BHM = None

        self.min_max = min_max
        self.x_min, self.x_max, self.y_min, self.y_max = self.min_max

        qxx, qyy = np.meshgrid(np.linspace(self.x_min, self.x_max, 224), np.linspace(self.y_min, self.y_max, 224))

        self.qX = np.hstack((qxx.ravel().reshape(-1, 1), qyy.ravel().reshape(-1, 1)))   # used for querying model, resolution arbitrary

        self.position = starting_pos
        self.previous_positions = set()
        self.previous_positions_map = np.zeros((1, 224, 224, 1), dtype=np.float32)
        self.previous_actions = set()
        self.steps_taken = 0
        self.range = LIDAR_pixel_range
        self.gt = ground_truth_map
        #self.curr_lstmstate_drone = tf.truncated_normal(shape=(1, 2, 52**2), stddev = 0.05).eval(session=tf.Session())

        # -------------------- Network to be trained -------------------------
        self.network_model = initialize_network_FCQN(plot_directory=plot_dir, save_weights_dir=weights_dir, custom_load_path=custom_load, )

    def reset(self, fresh_BHM, starting_pos):
        """Call reset every time Agent finished exploring map"""
        self.BHM = fresh_BHM
        self.previous_BHM = None

        self.position = starting_pos
        self.previous_positions = set()
        self.previous_positions_map = np.zeros((1, 224, 224, 1), dtype=np.float32)
        self.previous_actions = set()
        self.steps_taken = 0
        #self.curr_lstmstate_drone = tf.truncated_normal(shape=(1, 2, 52**2), stddev = 0.05).eval(session=tf.Session())
        self.collect_data()     # must collect data again because BHM must be fitted at least once before querying

    def move_by_sequence(self, waypoints):
        """Moves agent along waypoints and build the BHM model at the same time.

        :param waypoints: List of (x, y) waypoints.
        :returns (True, path length) if safe travel, (False, path_length) if dangerous
        """

        danger_radius = 5
        occ_threshold = 0.7
        path_length = 0

        self.previous_BHM = deepcopy(self.BHM)

        for index, point in enumerate(waypoints):

            if point == self.position and index == 0:
                print("WARNING: Starting pos included in waypoints. This can cause errors. Remove the first point")
                assert False

            # assess danger of point. DO NOT INCLUDE FIRST POINT
            danger_circle = bloom(point, danger_radius, resolution_per_quadrant=16)
            pred_occupancies = self.BHM.predict_proba(danger_circle)[:, 1]
            waypoint_close_to_obstacle = any(occ_val > occ_threshold for occ_val in pred_occupancies)
            if waypoint_close_to_obstacle:
                # print("previously free pathway turns out to be blocked")
                return False, path_length

            # Update previous positions map
            self.update_prev_pos_map()

            path_length += np.linalg.norm((self.position[0] - point[0], self.position[1] - point[1]))
            self.position = point

            self.steps_taken += 1
            self.collect_data()

        return True, path_length

    def update_prev_pos_map(self):
        self.previous_positions_map *= 0.9      # DECAY POSITIONS OVER TIME
        self.previous_positions.add(self.position)
        neighbours_inc_c = neighbours_including_center(self.position)
        if neighbours_inc_c is not None:
            for p in neighbours_inc_c:
                px, py = p
                value_increment = 0.5 if (p == self.position) else 0.3  # central pixel 0.5, surrounding 0.3
                self.previous_positions_map[0, py, px, 0] += value_increment
        else:   # None neighbours means edge
            px, py = self.position
            self.previous_positions_map[0, py, px, 0] = 1

    def collect_data(self):  # note: the absolute first scan after initialising will take extra long due to SBHM starting
        """
        Sequence ->
        1) Bloom current position, to get points on circle
        2) Raytrace from center to edge, creating LiDAR beams
        3) Simulate LiDAR collisions with ground truth map
        4) Get training data to update BHM
        5) Fit BHM
        """
        circle = bloom(self.position, self.range)
        center_to_edge_lines = [split_points_evenly(self.position, point, self.range) for point in circle]
        end_points = get_end_points(center_to_edge_lines, self.gt)
        training_data = getTrainingData(end_points, self.position, self.range - 2, 0.05, 0.05)

        Xd, Yd = training_data[:, :2], training_data[:, 2]
        self.BHM.fit(Xd, Yd)

    def show_model(self):
        """Visualise BHM model"""
        y_pred = self.BHM.predict_proba(self.qX)[:, 1]
        #print(len(y_pred))
        plt.scatter(self.qX[:, 0], self.qX[:, 1], c=y_pred, s=4, cmap='jet')
        #plt.colorbar()
        #plt.scatter(self.BHM.grid[:, 0], self.BHM.grid[:, 1])
        plt.scatter(self.position[0], self.position[1], marker='X', c="#FFE4C4", s=100)
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_max, self.y_min)
        plt.draw()
        

    def get_free_mask(self):
        """Used to bitwise AND with ground truth map to determine finished ratio"""
        raw_pred = self.BHM.predict_proba(self.qX)[:, 1].reshape((224, 224))    # predict_proba returns a flat array
        free_threshold = 0.3    # arbitrary

        # NOTE: THIS IS PURELY TO DO BITWISE AND, 1 considered as free. OPPOSITE to the convention of 1 occupied, 0 free
        mask = np.where(raw_pred < free_threshold, 1, 0)  # any pixel below 0.3 occupied probability becomes a 1, else 0

        return mask

    def reward_gen(self, length_of_path, goal_in_unknown_space=False, safe_travel=True):
        alpha = 0.003  # make training process more stable
        p_coeff = 0.4  # dictate how much to penalise path length

        if length_of_path == 0:  # no exploration done this iter
            return -1   # penalise failing to find goal, or goal too close to obstacle (accounted for in main code)

        # if exploration done, evaluate efficiency

        # ********************************** USING RELATIVE ENTROPY *******************************
        # The higher RE, the better, because of my definition of more deviation from baseline -> more information
        previous_RE = self.relative_entropy(self.previous_BHM, baseline=0.5)
        current_RE = self.relative_entropy(self.BHM, baseline=0.5)
        gain_in_RE = current_RE - previous_RE
        reward = alpha * (gain_in_RE - p_coeff * length_of_path) - 0.7  # Extra -0.8 tagged on to hopefully reduce number of Goal points generated

        # ----- reward concerning safety of agent -----
        reward -= 1 if goal_in_unknown_space else 0
        reward -= 1 if not safe_travel else 0
        # Note that goal too close to obstacle isn't here because i accounted for it in my main code, where goal
        # too close causes me to not do any exploration, in turn making my path_length 0, which is early returned above

        return reward

    def relative_entropy(self, BHM_model, baseline=0.5):
        occ_pred = BHM_model.predict_proba(self.qX)[:, 1]
        return np.sum((1 - occ_pred) * np.log((1 - occ_pred)/(1 - baseline)) + occ_pred * np.log(occ_pred / baseline))

    def get_state(self):
        """3 Channels-> Occupancy map, current position, previous positions"""
        sampled_grids = self.BHM.predict_proba(self.qX)[:, 1].reshape((1, 224, 224, 1))
        curr_position = np.zeros((1, 224, 224, 1), dtype=int)

        px, py = self.position
        curr_position[0, py, px, 0] = 1
        return sampled_grids, curr_position, self.previous_positions_map

    def relative_entropy_infer(self):
        RE = self.relative_entropy(self.BHM, baseline=0.5)
        return RE
