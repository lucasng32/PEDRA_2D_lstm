import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from collections import deque
from aux_funcs import bloom_series

"""THIS FILE IS USED IN RELATION TO BAYESIAN HILBERT MAPS FOR *2D*"""


class Line:
    """Line class made of Points"""

    def __init__(self, p0, p1):
        self.start = p0
        self.end = p1


def intersection_with_BHM_np_array(line: Line, BHM_arr, crash_radius=10, line_includes_starting_point=False):
    """Same as `intersection_with_BHM_map` just working with np arr instead, see `RRT_n_star_np_arr` for explanation"""
    # TODO: NOTE crash radius must be higher during path generation to give some leeway
    sample_rate = 10
    occ_threshold = 0.7

    if line_includes_starting_point:  # the explanation for this is basically, I found the algo does not work if the start is close to a wall. This prevents checking near the start pos.
        points_on_line = np.array([line.end])
    else:
        points_on_line = np.linspace(line.start, line.end, sample_rate)
    #print (points_on_line)
    points_on_line = bloom_series(points_on_line, crash_radius, 15)
    #print (points_on_line)
    #sprint (BHM_arr)
    pred_occupancies = [BHM_arr[p[1]][p[0]] for p in points_on_line]

    # as long as any predicted occupancy > threshold, considered occupied and cutting map obstacles
    return any(occ_val > occ_threshold for occ_val in pred_occupancies)


def intersection_with_BHM_map(line: Line, BHM_model_map, crash_radius=10, line_includes_starting_point=False):
    # TODO: NOTE crash radius must be higher during path generation to give some leeway
    sample_rate = 10
    occ_threshold = 0.7

    if line_includes_starting_point:  # the explanation for this is basically, I found the algo does not work if the start is close to a wall. This prevents checking near the start pos.
        points_on_line = np.array([line.end])
    else:
        points_on_line = np.linspace(line.start, line.end, sample_rate)

    points_on_line = bloom_series(points_on_line, crash_radius, 4)

    pred_occupancies = BHM_model_map.predict_proba(points_on_line)[:, 1]

    # as long as any predicted occupancy > threshold, considered occupied and cutting map obstacles
    return any(occ_val > occ_threshold for occ_val in pred_occupancies)


def distance(p0, p1):
    return np.linalg.norm(np.array(p0) - np.array(p1))


def nearest(G, new_vex):
    nearest_vex = None
    nearest_idx = None
    minDist = float('inf')

    for idx, v in enumerate(G.vertices):

        dist = distance(v, new_vex)
        if dist < minDist:
            minDist = dist
            nearest_idx = idx
            nearest_vex = v

    return nearest_vex, nearest_idx


def new_vertex(rand_vex, near_vex, step_size):
    dirn = np.array(rand_vex) - np.array(near_vex)
    length = np.linalg.norm(dirn)
    if length == 0:
        return None
    dirn = (dirn / length) * min(step_size, length)

    new_vex = (near_vex[0] + dirn[0], near_vex[1] + dirn[1])
    return new_vex


# noinspection PyBroadException
class Graph:
    """Define Graph"""

    def __init__(self, startpos, endpos, bounding_box: tuple):
        self.startpos = startpos
        self.endpos = endpos
        self.bounding_box = bounding_box
        # self.z_level = z_level

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {startpos: 0}
        self.neighbors = {0: []}
        self.distances = {0: 0.}

        self.sx = endpos[0] - startpos[0]
        self.sy = endpos[1] - startpos[1]

    def add_vex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
        return idx

    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))

    def random_position(self):
        min_x, max_x, min_y, max_y = (int(elem) for elem in self.bounding_box)
        rx = random.randrange(min_x, max_x)
        ry = random.randrange(min_y, max_y)

        posx = rx
        posy = ry
        return posx, posy


def RRT_n_star_np_arr(graph, BHM_np_arr, n_iter, radius, stepSize, crash_radius, n_retries_allowed=float('inf')):
    """RRT n star algorithm where n denotes number of retries to finish

    Same as `RRT_n_star` but this works with np array while that works with Bayesian Hilbert map.
    The difference is working with np array is much faster, but requires you to query BHM in main code and reshape to
    np array first
    """
    G = graph
    times_finished = 0
    # extra check for if start pos is already beside end pos, instant solve
    dist = distance(G.startpos, G.endpos)
    if dist < radius:
        end_idx = G.add_vex(G.endpos)
        G.add_edge(0, end_idx, dist)
        G.distances[end_idx] = dist

        G.success = True
        return G

    for step in range(n_iter):
        if times_finished > n_retries_allowed:
            break
        rand_vex = G.random_position()

        near_vex, near_idx = nearest(G, rand_vex)
        if near_vex is None:
            continue

        new_vex = new_vertex(rand_vex, near_vex, stepSize)
        if new_vex is None:
            continue

        new_line = Line(near_vex, new_vex)
        if near_vex == G.startpos:
            # print('line includes starting point')
            intersects_map = intersection_with_BHM_np_array(new_line, BHM_np_arr,
                                                            crash_radius=crash_radius, line_includes_starting_point=True)
        else:
            intersects_map = intersection_with_BHM_np_array(new_line, BHM_np_arr, crash_radius=crash_radius)
        if intersects_map:
            continue

        new_idx = G.add_vex(new_vex)
        dist = distance(new_vex, near_vex)
        G.add_edge(new_idx, near_idx, dist)
        G.distances[new_idx] = G.distances[near_idx] + dist

        # update nearby vertices distance (if shorter)
        for vex in G.vertices:
            if vex == new_vex:
                continue

            dist = distance(vex, new_vex)
            if dist > radius:
                continue

            line = Line(vex, new_vex)
            if intersection_with_BHM_np_array(line, BHM_np_arr):
                continue

            idx = G.vex2idx[vex]
            if G.distances[new_idx] + dist < G.distances[idx]:
                G.add_edge(idx, new_idx, dist)
                G.distances[idx] = G.distances[new_idx] + dist

        dist = distance(new_vex, G.endpos)
        if dist < 2 * radius:
            end_idx = G.add_vex(G.endpos)
            G.add_edge(new_idx, end_idx, dist)
            try:
                G.distances[end_idx] = min(G.distances[end_idx], G.distances[new_idx] + dist)
            except:
                G.distances[end_idx] = G.distances[new_idx] + dist

            G.success = True
            times_finished += 1
    # print('steps', step)
    return G


def RRT_n_star(graph, BHM_model_map, n_iter, radius, stepSize, crash_radius, n_retries_allowed=float('inf')):
    """RRT n star algorithm where n denotes number of retries to finish"""
    G = graph
    times_finished = 0
    # extra check for if start pos is already beside end pos, instant solve
    dist = distance(G.startpos, G.endpos)
    if dist < radius:
        end_idx = G.add_vex(G.endpos)
        G.add_edge(0, end_idx, dist)
        G.distances[end_idx] = dist

        G.success = True
        return G

    for step in range(n_iter):
        if times_finished > n_retries_allowed:
            break
        rand_vex = G.random_position()

        near_vex, near_idx = nearest(G, rand_vex)
        if near_vex is None:
            continue

        new_vex = new_vertex(rand_vex, near_vex, stepSize)
        if new_vex is None:
            continue

        new_line = Line(near_vex, new_vex)
        if near_vex == G.startpos:
            # print('line includes starting point')
            intersects_map = intersection_with_BHM_map(new_line, BHM_model_map,
                                                       crash_radius=crash_radius, line_includes_starting_point=True)
        else:
            intersects_map = intersection_with_BHM_map(new_line, BHM_model_map, crash_radius=crash_radius)
        if intersects_map:
            continue

        new_idx = G.add_vex(new_vex)
        dist = distance(new_vex, near_vex)
        G.add_edge(new_idx, near_idx, dist)
        G.distances[new_idx] = G.distances[near_idx] + dist

        # update nearby vertices distance (if shorter)
        for vex in G.vertices:
            if vex == new_vex:
                continue

            dist = distance(vex, new_vex)
            if dist > radius:
                continue

            line = Line(vex, new_vex)
            if intersection_with_BHM_map(line, BHM_model_map):
                continue

            idx = G.vex2idx[vex]
            if G.distances[new_idx] + dist < G.distances[idx]:
                G.add_edge(idx, new_idx, dist)
                G.distances[idx] = G.distances[new_idx] + dist

        dist = distance(new_vex, G.endpos)
        if dist < 2 * radius:
            end_idx = G.add_vex(G.endpos)
            G.add_edge(new_idx, end_idx, dist)
            try:
                G.distances[end_idx] = min(G.distances[end_idx], G.distances[new_idx] + dist)
            except:
                G.distances[end_idx] = G.distances[new_idx] + dist

            G.success = True
            times_finished += 1
    # print('steps', step)
    return G


def dijkstra(G):
    """Dijkstra algorithm for finding shortest path from start position to end."""
    srcIdx = G.vex2idx[G.startpos]
    dstIdx = G.vex2idx[G.endpos]

    # build dijkstra
    nodes = list(G.neighbors.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[srcIdx] = 0

    while nodes:
        curNode = min(nodes, key=lambda node: dist[node])
        nodes.remove(curNode)
        if dist[curNode] == float('inf'):
            break

        for neighbor, cost in G.neighbors[curNode]:
            newCost = dist[curNode] + cost
            if newCost < dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path = deque()
    curNode = dstIdx
    while prev[curNode] is not None:
        path.appendleft(G.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(G.vertices[curNode])

    return list(path)


def plot(G: Graph, BHM_model_map, path=None):
    """Plot RRT, obstacles and shortest path"""
    # print('1')
    fig, ax = plt.subplots()
    # print('2')
    x_min, x_max, y_min, y_max = G.bounding_box
    if G is not None:
        px = [point[0] for point in G.vertices]
        py = [point[1] for point in G.vertices]

        ax.scatter(px, py, c='cyan')
        ax.scatter(G.startpos[0], G.startpos[1], c='black')
        ax.scatter(G.endpos[0], G.endpos[1], c='black')

        qxx, qyy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 50))
        qX = np.hstack((qxx.ravel().reshape(-1, 1), qyy.ravel().reshape(-1, 1)))
        y_pred = BHM_model_map.predict_proba(qX)
        ax.scatter(qX[:, 0], qX[:, 1], c=y_pred[:, 1], s=4, cmap='jet', vmin=0, vmax=1, alpha=0.4)

        if path is not None:
            paths = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
            ax.add_collection(lc2)
    plt.xlim([x_min, x_max])
    plt.ylim([y_max, y_min])
    # ax.autoscale()
    ax.margins(0.1)

    plt.show()
