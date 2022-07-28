import Bayesian_Hilbert_Maps.BHM.original.sbhm as sbhm
from map_2D.aux_funcs import *
import map_2D.networks.agent2D as agent2D
import map_2D.rrt_BHM as rrt_BHM
import matplotlib.pyplot as plt
import sys

valid_starting_points = [(113, 76), (52, 125), (182, 187), (81, 18), (75, 197), (193, 107), (151, 162)]  # X, Y

# Training map
gt = get_ground_truth_array(r'/Users/lucas/Desktop/PEDRA_2D_lstm/map_2D/environments/filled_simple_floorplan_v2.png')
#plt.imshow(gt, 'Greys_r')
#plt.show()

# Paths
custom_load_dir = '/Users/lucas/Desktop/PEDRA_2D_lstm/map_2D/results/weights/drone_2D1'
log_dir = '/Users/lucas/Desktop/PEDRA_2D_lstm/map_2D/results/inference/infer_log.txt'

# RRT variables
danger_radius = 4
occ_threshold = 0.7

# SBHM variables
gamma = 0.02
cell_res = (12, 12)
min_max = (0, 223, 0, 223)
LIDAR_max_range = 76
starting_pos_index = 0

BHM = sbhm.SBHM(gamma=gamma, cell_resolution=cell_res, cell_max_min=min_max)

# agent
drone = agent2D.agent_2D(BHM=BHM, min_max=min_max, LIDAR_pixel_range=LIDAR_max_range, ground_truth_map=gt, starting_pos=valid_starting_points[starting_pos_index],
                         plot_dir='', weights_dir='', custom_load=custom_load_dir)

# Inference Variables
minimum_finished_ratio = 0.77

#Overall Values
total_path_length = 0
total_entropy_gain = 0
total_explored_region_rate = 0
total_iters = 35

plt.ion()
#plt.scatter(drone.position[0], drone.position[1], cmap='jet')
#drone.show_model()
print("******** INFERENCE BEGINS *********")

for ep in range(0,total_iters):

    cum_path_length_ep = 0
    first_step = True
    drone.collect_data()    # need to do 1 fitting of BHM first before can query

    current_state = drone.get_state()

    initial_entropy = drone.relative_entropy_infer()
    plt.pause(0.001)
    drone.show_model()
    plt.show()
    plt.pause(3.0)
    while True:
        no_dupe = drone.network_model.action_selection_non_repeat(current_state, drone.previous_actions, first_step=first_step)

        print('no dupe actoin', no_dupe[0])
        # RRT* Algo
        startpos = drone.position
        goalpos = action_idx_to_coords(no_dupe[0], min_max)

        G = rrt_BHM.Graph(startpos, goalpos, min_max)
        G = rrt_BHM.RRT_n_star(G, drone.BHM, n_iter=450, radius=5, stepSize=14, crash_radius=5, n_retries_allowed=0)

        if G.success:
            path = rrt_BHM.dijkstra(G)

            path = [(int(elem[0]), int(elem[1])) for elem in path]

            _, path_length = drone.move_by_sequence(path[1:])  # exclude first point

            cum_path_length_ep += path_length


        else:
            path_length = 0

        done = False
        if path_length != 0:
            free_mask = drone.get_free_mask()
            correct = np.logical_and(gt, free_mask)
            #plt.imshow(correct, cmap='Greys_r')
            #plt.scatter(drone.position[0], drone.position[1], cmap='jet')
            #plt.draw()
            plt.pause(0.001)
            drone.show_model()
            finished_ratio = np.sum(correct) / np.sum(gt)
            print("Finished ratio:", finished_ratio)

            if finished_ratio > minimum_finished_ratio:
                done = True

            new_state = drone.get_state()

        else:
            new_state = current_state

        if done:
            plt.pause(3.0)
            final_entropy = drone.relative_entropy_infer()
            exploration_efficiency = (final_entropy - initial_entropy)/cum_path_length_ep
            print("******** EXPLORATION DONE ********* Ep: "+str(ep)+" "+"Starting Pos: "+str(starting_pos_index))
            print("Path Length:", cum_path_length_ep)
            print("Finished ratio:", finished_ratio)
            print("Exploration Efficiency:", exploration_efficiency)
            total_entropy_gain += final_entropy - initial_entropy
            total_path_length += cum_path_length_ep
            total_explored_region_rate += finished_ratio
            if starting_pos_index == 6: starting_pos_index = 0
            else: starting_pos_index += 1
            drone.reset(fresh_BHM=sbhm.SBHM(gamma=gamma, cell_resolution=cell_res, cell_max_min=min_max),starting_pos=valid_starting_points[starting_pos_index])
            first_step = True
            break

        else:
            current_state = new_state
            drone.previous_actions.add(tuple(no_dupe[0]))
            first_step = False


print("******** All EPS DONE *********")
print("Average Path Length:", total_path_length/total_iters)
print("Average Finished ratio:", total_explored_region_rate/total_iters)
print("Exploration Efficiency:", total_entropy_gain/total_path_length)