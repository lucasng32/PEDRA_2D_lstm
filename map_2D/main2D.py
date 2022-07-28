# FROM THIS POINT OUT, LET COORD BE DESCRIBED AS X, Y.
# To interpret model output, since he gives in terms of 224x224 action space, the first index is actually row (which is Y)
# and the second index is width (which is X)
import Bayesian_Hilbert_Maps.BHM.original.sbhm as sbhm
from map_2D.aux_funcs import *
import map_2D.networks.agent2D as agent2D
import time
import map_2D.rrt_BHM as rrt_BHM
import matplotlib.pyplot as plt
valid_starting_points = [(113, 76), (52, 125), (182, 187), (81, 18), (75, 197), (193, 107), (151, 162)]  # X, Y

# Training map
gt = get_ground_truth_array(r'/Users/lucas/Desktop/PEDRA_2D_lstm/map_2D/environments/filled_simple_floorplan_v2.png')
#plt.imshow(gt, 'Greys_r')
#plt.show()
# Paths
plot_dir = '/Users/lucas/Desktop/PEDRA_2D_lstm/map_2D/results/stats'
weights_dir = '/Users/lucas/Desktop/PEDRA_2D_lstm/map_2D/results/weights'
log_dir = '/Users/lucas/Desktop/PEDRA_2D_lstm/map_2D/results/log'

custom_load = r'/Users/lucas/Desktop/PEDRA_2D_lstm/map_2D/results/weights/drone_2D_testnew_6000'

# Initialise variables
iter = 0
max_iters = 6000
save_interval = max_iters // 3
level = 0   # if implementing switching starting positions
current_starting_pos_index = 0
episode = 0  # how many times drone completed exploration
moves_taken = 0
epsilon_saturation = 10000 #perhaps increase this
epsilon_model = 'exponential'
epsilon = 0  # start with drone always taking random actions
cum_return = 0
discount_factor = 0.9
Q_clip = False   # clips TD error to -1, 1
learning_rate = 1e-6
first_step = True

consecutive_fails = 0
max_consecutive_fails = 6  # for debugging purposes

# RRT variables
danger_radius = 5
occ_threshold = 0.7

# SBHM variables
gamma = 0.02
cell_res = (12, 12)
min_max = (0, 223, 0, 223)
LIDAR_max_range = 50    # in pixels

BHM = sbhm.SBHM(gamma=gamma, cell_resolution=cell_res, cell_max_min=min_max)

# agent
drone = agent2D.agent_2D(BHM=BHM, min_max=min_max, LIDAR_pixel_range=LIDAR_max_range, ground_truth_map=gt, starting_pos=valid_starting_points[current_starting_pos_index],
                         plot_dir=plot_dir, weights_dir=weights_dir, custom_load=custom_load)
drone.collect_data()    # need to do 1 fitting of BHM first before can query
current_state = drone.get_state()

#plt.ion()
#plt.show()
print("******** SIMULATION BEGINS *********")
# TRAINING LOOP
log_file = open(log_dir + '/log.txt', mode='w')

while True:
    start_time = time.time()

    if cum_return < -200.0:  #prevent drone from being stuck for too long, reduce iterations per episode
        print("Restart Episode, bad Ret")
        #rrt_BHM.plot(G, drone.BHM, None)
        drone.reset(fresh_BHM=sbhm.SBHM(gamma=gamma, cell_resolution=cell_res,cell_max_min=min_max),
                                                                    starting_pos=valid_starting_points[current_starting_pos_index])
        current_state = drone.get_state()
        # don't +1 to episode, treat as same episode and reset move and return
        moves_taken = 0
        cum_return = 0
        consecutive_fails = 0
        first_step = True
        continue

    action, action_type, epsilon = policy_FCQN(epsilon, current_state,
                                                        iter, epsilon_saturation, 'exponential', drone, first_step=first_step)
    # action, action_type, epsilon = policy_FCQN_no_dupe(epsilon, current_state,
    #                                                    iter, epsilon_saturation, 'exponential', drone)

    drone.previous_actions.add(tuple(action[0]))    # TODO: Hide this working into drone class so won't forget to do

    # RRT* algo
    startpos = drone.position
    goalpos = action_idx_to_coords(action[0], min_max)

    valid_goal = True

    surroundings = bloom(goalpos, danger_radius, resolution_per_quadrant=16)
    pred_occupancies = drone.BHM.predict_proba(surroundings)[:, 1]
    
    goal_close_to_obstacle = any(occ_val > occ_threshold for occ_val in pred_occupancies)

    pred_goal = drone.BHM.predict_proba(np.array([goalpos]))[0][1]
    goal_in_unknown_space = 0.4 < pred_goal < 0.65  # roughly, if my probability of being occupied is around 0.5 +- 0.1, means im unsure, which is dangerous

    if pred_goal > occ_threshold or goal_close_to_obstacle:  # point selected is in obstacle / too close
        path = None
        path_length = 0
        safe_travel = None
    else:
        G = rrt_BHM.Graph(startpos, goalpos, min_max)
        # G = rrt_BHM.RRT_n_star(G, drone.BHM, n_iter=450, radius=5,      # RRT Params must be modified based on the environment, but this is not an issue of the agent
        #                        stepSize=14, crash_radius=5, n_retries_allowed=0)
        G = rrt_BHM.RRT_n_star_np_arr(G, np.reshape(drone.BHM.predict_proba(drone.qX)[:, 1], (224, 224)),
                                      n_iter=500, radius=5, stepSize=14, crash_radius=5, n_retries_allowed=0)
        if G.success:
            path = rrt_BHM.dijkstra(G)
            path = [(int(elem[0]), int(elem[1])) for elem in path]

            safe_travel, path_length = drone.move_by_sequence(path[1:])  # exclude first point

            if path_length == 0:
                consecutive_fails += 1
                if consecutive_fails == max_consecutive_fails:
                    print("DRONE STUCKKKK")
                    print('drone_pos:', drone.position)
                    print('goal_pos:', goalpos)
                    #rrt_BHM.plot(G, drone.BHM, None)
                    drone.reset(fresh_BHM=sbhm.SBHM(gamma=gamma, cell_resolution=cell_res,cell_max_min=min_max),
                                                                    starting_pos=valid_starting_points[current_starting_pos_index])
                    current_state = drone.get_state()
                    # don't +1 to episode, treat as same episode and reset move and return
                    moves_taken = 0
                    cum_return = 0
                    consecutive_fails = 0
                    first_step = True
                    continue
            else:
                consecutive_fails = 0
            moves_taken += 1
        else:

            path = None
            path_length = 0
            safe_travel = None

    reward = drone.reward_gen(path_length, goal_in_unknown_space=goal_in_unknown_space, safe_travel=safe_travel)

    # check for completeness and update state, only if moved
    done = False
    if path_length != 0:
        free_mask = drone.get_free_mask()
        correct = np.logical_and(gt, free_mask)
        #plt.imshow(correct, cmap='Greys_r')
        #plt.draw()
        #plt.pause(0.001)
        #drone.show_model()
        finished_ratio = np.sum(correct) / np.sum(gt)
        print("Finished ratio:", finished_ratio)

        if finished_ratio > 0.77:
            done = True
            reward += 1

        new_state = drone.get_state()

    else:
        new_state = current_state

    # TRAINING DONE HERE
    cum_return = cum_return + reward

    data_tuple = (current_state, action, new_state, reward)

    _, Q_target, err = get_err_FCQN(data_tuple, drone, discount_factor, Q_clip, first_step = first_step)

    q_map, loss = drone.network_model.train_n(current_state, action, Q_target, 1, learning_rate, epsilon, iter, first_step = first_step)
    # ------------------

    time_exec = time.time() - start_time

    s_log = 'drone_2D - Level {:>2d} - Iter: {:>5d}/{:<4d} Action: {}-{:>5s} Eps: {:<1.4f} lr: {:>1.6f} Ret = {:<+6.4f} t={:<1.3f} Moves: {:<2} Steps: {:<3} Reward: {:<+1.4f}  '.format(
        level,
        iter,
        episode,
        action,
        action_type,
        epsilon,
        learning_rate,
        cum_return,
        time_exec,
        moves_taken,
        drone.steps_taken,
        reward)

    print(s_log)
    log_file.write(s_log + '\n')
    print(q_map)
    log_file.write("Q: "+q_map+ '\n')
    print(loss)
    log_file.write("Loss: "+loss+ '\n')
    if path_length != 0: log_file.write("Finished ratio: "+str(finished_ratio)+'\n')
    #log_file.write(weightss_test + '\n')

    first_step = False
    if done:
        drone.network_model.log_to_tensorboard(tag='Return', group='drone_2D',
                                                           value=cum_return,
                                                           index=episode)
        drone.network_model.log_to_tensorboard(tag='Moves (valid goalpoints)', group='drone_2D',
                                                       value=moves_taken,
                                                       index=episode)
        drone.network_model.log_to_tensorboard(tag='Steps (waypoints)', group='drone_2D',
                                               value=len(drone.previous_positions),
                                               index=episode)

        drone.reset(fresh_BHM=sbhm.SBHM(gamma=gamma, cell_resolution=cell_res,
                                        cell_max_min=min_max),
                    starting_pos=valid_starting_points[current_starting_pos_index])

        current_state = drone.get_state()

        # drone.show_model()
        episode += 1
        moves_taken = 0
        cum_return = 0
        first_step = True

        if episode % 3 == 0 and episode > 0:    # Change starting points every 3 Episodes
            current_starting_pos_index += 1
            if current_starting_pos_index == len(valid_starting_points):
                current_starting_pos_index = 0
            level = current_starting_pos_index
            print("changing starting pos")

    else:
        current_state = new_state
    iter += 1

    if iter % save_interval == 0:
        drone.network_model.save_network(str(iter))
    if iter == max_iters:
        print("TRAINING DONE")
        break

log_file.close()
