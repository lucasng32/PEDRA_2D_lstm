#from asyncio.windows_events import NULL
import numpy as np


def array_shift(array, axis, n):
    """Bitshifting numpy arrays, but not used in this project"""
    empty = np.zeros_like(array, dtype=int)
    if axis == 'vertical':
        if n >= 0:  # down as positive
            empty[n:, :] = array[:-n, :]
        else:
            empty[:n, :] = array[-n:, :]
    if axis == 'horizontal':
        if n >= 0:
            empty[:, n:] = array[:, :-n]
        else:
            empty[:, :n] = array[:, -n:]
    return empty


def get_ground_truth_array(floorplan_path: str):
    """Used once at the start of the program to convert BnW image into numpy array."""
    from PIL import Image

    with Image.open(floorplan_path) as image_file:
        image = image_file.resize((224, 224))
        image_array = np.array(image.convert('1')).astype(np.uint8)

    return image_array


def split_points_evenly(pixel1, pixel2, count):
    # Note the rounding used here is to map perfectly to my pixel values
    x = np.round(np.linspace(pixel1, pixel2, count, dtype=int))

    return x


def get_end_points(lines: list, map: np.ndarray):
    """Returns contact point if hit wall, otherwise just the end of line"""
    end_points = np.zeros((len(lines), 2), dtype=int)

    for ii, line_of_pixels in enumerate(lines):
        for pixel in line_of_pixels:    # pixel is stored as (x, y)
            if map[pixel[1], pixel[0]]:  # to index into a 2D array, index by row, then column which is y then x
                continue
            else:
                end_points[ii] = pixel
                break
        else:
            end_points[ii] = line_of_pixels[-1]  # end point of line

    return end_points


def getTrainingData(endpoints, agent_pos, max_laser_distance, unoccupied_points_per_pixel, margin=0.05):
    """Used to create training data to update Bayesian Hilbert map (simple ML model)

    :param endpoints: np array of endpoints acquired from `get_end_points` function
    :param agent_pos: (x, y)

    :param max_laser_distance: in pixels. BTW, if you want real life distance, you need to implement a global mapping
                    of pixel values to real life coordinates, and remove the rounding to ints present in all functions

    :param unoccupied_points_per_pixel: eg. if there are 100 pixels in your raytrace, a value of 0.05 will mean 5 pixels
    :param margin: prevent from choosing a random point too close to endpoint

    :return: Training data that can be fed into Bayesian Hilbert Map model with features = [:, :2] and labels [:, 2]
    """
    distances = np.sqrt(np.sum((endpoints - agent_pos) ** 2, axis=1))

    # parametric filling
    for n in range(len(distances)):
        dist = distances[n]
        laser_endpoint = endpoints[n, :2]

        para = np.sort(np.random.random(np.int16(dist * unoccupied_points_per_pixel)) *
                       (1 - 2 * margin) + margin)[:, np.newaxis]  # TODO: Uniform[0.05, 0.95]
        points_scan_i = agent_pos + para * (laser_endpoint - agent_pos)

        if n == 0:  # first data point
            if dist >= max_laser_distance:  # there's no laser reflection
                points = points_scan_i
                labels = np.zeros((points_scan_i.shape[0], 1))
            else:  # append the arrays with laser end-point
                points = np.vstack((points_scan_i, laser_endpoint))
                labels = np.vstack((np.zeros((points_scan_i.shape[0], 1)), np.array([1])[:, np.newaxis]))
        else:
            if dist >= max_laser_distance:  # there's no laser reflection
                points = np.vstack((points, points_scan_i))
                labels = np.vstack((labels, np.zeros((points_scan_i.shape[0], 1))))
            else:  # append the arrays with laser end-point
                points = np.vstack((points, np.vstack((points_scan_i, laser_endpoint))))
                labels = np.vstack(
                    (labels, np.vstack((np.zeros((points_scan_i.shape[0], 1)), np.array([1])[:, np.newaxis]))))

    return np.hstack((points, labels))


def neighbours_including_center(point, radius=1):
    """1 radius means 3x3 square.
    If pixel at edge, return None because he shouldn't be in the first place, confirm crash (my map has borders)"""
    if radius == 1:
        if max(point) == 223 or min(point) == 0:
            return None
        L = point[0] - 1
        R = point[0] + 1
        T = point[1] - 1    # up negative
        B = point[1] + 1
        Cx = point[0]
        Cy = point[1]
        return [(L, T), (Cx, T), (R, T),
                (L, Cy), (Cx, Cy), (R, Cy),
                (L, B), (Cx, B), (R, B)]
    else:   # radius != 1
        assert False and "Not implemented"


def bloom(point, radius, resolution_per_quadrant=60):
    """Blooms a point about a certain radius, used to simulate LiDAR. This function is used in conjunction with
    `get_end_points` function to check for the LiDAR hitpoints.

    :param point: center, which is your current agent position
    :param radius: of circle
    :param resolution_per_quadrant: Originally, this function was written without int rounding in mind. Please set this
                        high enough (appropriate value depends on radius of circle) so the circle won't be broken.
                        Do some testing and visualisation (matplotlib) to determine what your resolution should be

    :return: np array of points on circle
    """
    new_points = []
    for angle in np.arange(0, 2 * np.pi, (np.pi / 2) / resolution_per_quadrant):
        dy = radius * np.sin(angle)
        dx = radius * np.cos(angle)
        point_on_circle = (point + np.array([dx, dy])).astype(int)
        new_points.append(point_on_circle)

    return np.clip(new_points, 0, 223)      # Clipped to min max pixel values of 0 and 223


def bloom_series(line_of_points, radius, resolution_per_quadrant=4):
    """Bloom but applied to all points at once, faster"""
    initial_points = line_of_points
    for angle in np.arange(0, 2 * np.pi, (np.pi / 2) / resolution_per_quadrant):
        dy = radius * np.sin(angle)
        dx = radius * np.cos(angle)
        new_points = (initial_points + np.array([dx, dy])).astype(int)
        line_of_points = np.vstack((line_of_points.astype(int), new_points))

    return np.clip(line_of_points, 0, 223)


# ----------- TRAINING FCQN MODEL RELATED ----------------
def get_err_FCQN(data_tuple, agent, gamma, Q_clip, first_step):
    curr_state_tuple, action, new_state_tuple, reward = data_tuple
    oldQval = agent.network_model.Q_val(curr_state_tuple, propagate_state=True, first_step=first_step)
    newQval = agent.network_model.Q_val(new_state_tuple, propagate_state=False, first_step=first_step)

    action_tuple = action[0][0], action[0][1]  # action[0] because action is a nested list of 1 item [[row, col]]

    # This applies to when model output is 52x52. now it is flattened to 2704 first
    # TD = reward + gamma * newQval[np.unravel_index(np.argmax(newQval), newQval.shape)] - oldQval[action_tuple]

    TD = reward + gamma * newQval[np.argmax(newQval)] - oldQval[action_tuple[0] * 52 + action_tuple[1]]
    if Q_clip:
        TD_clip = np.clip(TD, -1, 1)
    else:
        TD_clip = TD

    Q_step_rate = 1     # TODO: CAN MODIFY THIS STEP RATE
    # Q_target = oldQval[action_tuple] + Q_step_rate * TD_clip
    Q_target = oldQval[action_tuple[0] * 52 + action_tuple[1]] + Q_step_rate * TD_clip

    err = abs(TD_clip)

    return oldQval, np.array([Q_target]), err


def policy_FCQN(epsilon, curr_state_tuple, iter, b, epsilon_model, agent, first_step):
    base = -0.05
    epsilon_ceil = 0.95
    if epsilon_model == 'linear':
        epsilon = base + epsilon_ceil * iter / b
        if epsilon > epsilon_ceil:
            epsilon = epsilon_ceil

    elif epsilon_model == 'exponential':
        epsilon = base + 1 - np.exp(-2 / b * iter)
        if epsilon > epsilon_ceil:
            epsilon = epsilon_ceil

    if np.random.random() > epsilon:
        action = np.random.randint(0, 52, size=(1, 2), dtype=np.int32)
        action_type = 'Rand'
    else:
        # Use NN to predict action
        action = agent.network_model.action_selection(curr_state_tuple, first_step=first_step)
        action_type = 'Pred'

    return action, action_type, epsilon


def policy_FCQN_no_dupe(epsilon, curr_state_tuple, iter, b, epsilon_model, agent, first_step):
    """Same as `policy_FCQN but uses non_duplicated action selection. ONLY USED FOR INFERENCE/DEMONSTRATION"""
    base = -0.05
    epsilon_ceil = 0.95
    if epsilon_model == 'linear':
        epsilon = base + epsilon_ceil * iter / b
        if epsilon > epsilon_ceil:
            epsilon = epsilon_ceil

    elif epsilon_model == 'exponential':
        epsilon = base + 1 - np.exp(-2 / b * iter)
        if epsilon > epsilon_ceil:
            epsilon = epsilon_ceil

    if np.random.random() > epsilon:
        action = np.random.randint(0, 52, size=(1, 2), dtype=np.int32)
        action_type = 'Rand'
    else:
        # Use NN to predict action
        action = agent.network_model.action_selection_non_repeat(curr_state_tuple, agent.previous_actions, first_step=first_step)
        action_type = 'Pred'

    return action, action_type, epsilon


def action_idx_to_coords(action, min_max):
    """Maps action index to coordinates. In this current project, action is (0-52, 0-52) and coords is (0-223, 0-223)"""
    # NOTE: action is given in terms of row, column, which means [0] relates to Y while [1] relates to X
    x_min, x_max, y_min, y_max = min_max
    original_shape = (52, 52)
    x = action[1] / original_shape[1] * (x_max - x_min) + x_min
    y = action[0] / original_shape[0] * (y_max - y_min) + y_min
    return int(x), int(y)   # note the rounding again