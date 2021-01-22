import cv2
import numpy as np
import math
from collections import defaultdict
from scipy.ndimage.filters import maximum_filter
import itertools

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
    (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
    (0, 15), (15, 17), (2, 16), (5, 17)]  # len = 19
CocoPairsRender = CocoPairs[:-2]
CocoPairsNetwork = [
    (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1),
    (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35),
    (32, 33), (36, 37), (18, 19), (26, 27)]  # len = 19
CocoColors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NMS_Threshold = 0.1
InterMinAbove_Threshold = 6
Inter_Threashold = 0.1
Min_Subset_Cnt = 4
Min_Subset_Score = 0.5
Max_Human = 96


def non_max_suppression(heatmap, window_size=3, th=0.3):
    heatmap[heatmap < th] = 0
    part_th = heatmap*(heatmap == maximum_filter(heatmap,
                                                 footprint=np.ones((window_size, window_size))))
    return part_th


def estimate_pose(heatMat, pafMat):
    if heatMat.shape[2] == 19:
        # transform from [height, width, n_parts] to [n_parts, height, width]
        heatMat = np.rollaxis(heatMat, 2, 0)
    if pafMat.shape[2] == 38:
        # transform from [height, width, 2*n_pairs] to [2*n_pairs, height, width]
        pafMat = np.rollaxis(pafMat, 2, 0)

    _NMS_Threshold = max(np.average(heatMat) * 4.0, NMS_Threshold)
    _NMS_Threshold = min(_NMS_Threshold, 0.3)
    # _NMS_Threshold = 0.3

    coords = []  # for each part index, it stores coordinates of candidates
    for heatmap in heatMat[:-1]:  # remove background
        part_candidates = non_max_suppression(heatmap, 3, _NMS_Threshold)
        coords.append(np.where(part_candidates >= _NMS_Threshold))

    # all connections detected. no information about what humans they belong to
    connection_all = []
    for (idx1, idx2), (paf_x_idx, paf_y_idx) in zip(CocoPairs, CocoPairsNetwork):
        connection = estimate_pose_pair(
            coords, idx1, idx2, pafMat[paf_x_idx], pafMat[paf_y_idx])
        connection_all.extend(connection)

    conns_by_human = dict()
    for idx, c in enumerate(connection_all):
        # at first, all connections belong to different humans
        conns_by_human['human_%d' % idx] = [c]

    no_merge_cache = defaultdict(list)
    empty_set = set()
    while True:
        is_merged = False
        for h1, h2 in itertools.combinations(conns_by_human.keys(), 2):
            if h1 == h2:
                continue
            if h2 in no_merge_cache[h1]:
                continue
            for c1, c2 in itertools.product(conns_by_human[h1], conns_by_human[h2]):
                # if two humans share a part (same part idx and coordinates), merge those humans
                if set(c1['uPartIdx']) & set(c2['uPartIdx']) != empty_set:
                    is_merged = True
                    # extend human1 connectios with human2 connections
                    conns_by_human[h1].extend(conns_by_human[h2])
                    conns_by_human.pop(h2)  # delete human2
                    break
            if is_merged:
                no_merge_cache.pop(h1, None)
                break
            else:
                no_merge_cache[h1].append(h2)

        if not is_merged:  # if no more mergings are possible, then break
            break

    # reject by subset count
    conns_by_human = {h: conns for (
        h, conns) in conns_by_human.items() if len(conns) >= Min_Subset_Cnt}
    # reject by subset max score
    conns_by_human = {h: conns for (h, conns) in conns_by_human.items() if max(
        [conn['score'] for conn in conns]) >= Min_Subset_Score}

    # list of humans
    humans = [human_conns_to_human_parts(
        human_conns, heatMat) for human_conns in conns_by_human.values()]
    return humans


def estimate_pose_pair(coords, partIdx1, partIdx2, pafMatX, pafMatY):
    connection_temp = []  # all possible connections
    peak_coord1, peak_coord2 = coords[partIdx1], coords[partIdx2]

    for idx1, (y1, x1) in enumerate(zip(peak_coord1[0], peak_coord1[1])):
        for idx2, (y2, x2) in enumerate(zip(peak_coord2[0], peak_coord2[1])):
            score, count = get_score(x1, y1, x2, y2, pafMatX, pafMatY)
            if (partIdx1, partIdx2) in [(2, 3), (3, 4), (5, 6), (6, 7)]:  # arms
                if count < InterMinAbove_Threshold // 2 or score <= 0.0:  # what's means?
                    continue
            elif count < InterMinAbove_Threshold or score <= 0.0:
                continue
            connection_temp.append({
                'score': score,
                'coord_p1': (x1, y1),
                'coord_p2': (x2, y2),
                'idx': (idx1, idx2),  # connection candidate identifier
                'partIdx': (partIdx1, partIdx2),
                'uPartIdx': ('{}-{}-{}'.format(x1, y1, partIdx1), '{}-{}-{}'.format(x2, y2, partIdx2))
            })

    connection = []
    used_idx1, used_idx2 = [], []
    # sort possible connections by score, from maximum to minimum
    for conn_candidate in sorted(connection_temp, key=lambda x: x['score'], reverse=True):
        # check not connected
        if conn_candidate['idx'][0] in used_idx1 or conn_candidate['idx'][1] in used_idx2:
            continue
        connection.append(conn_candidate)
        used_idx1.append(conn_candidate['idx'][0])
        used_idx2.append(conn_candidate['idx'][1])

    return connection


def get_score(x1, y1, x2, y2, pafMatX, pafMatY):
    num_inter = 10
    dx, dy = x2 - x1, y2 - y1
    normVec = math.sqrt(dx ** 2 + dy ** 2)

    if normVec < 1e-4:
        return 0.0, 0

    vx, vy = dx / normVec, dy / normVec

    xs = np.arange(
        x1, x2, dx / num_inter) if x1 != x2 else np.full((num_inter, ), x1)
    ys = np.arange(
        y1, y2, dy / num_inter) if y1 != y2 else np.full((num_inter, ), y1)
    xs = (xs + 0.5).astype(np.int8)
    ys = (ys + 0.5).astype(np.int8)

    # without vectorization
    pafXs = np.zeros(num_inter)
    pafYs = np.zeros(num_inter)
    for idx, (mx, my) in enumerate(zip(xs, ys)):
        pafXs[idx] = pafMatX[my][mx]
        pafYs[idx] = pafMatY[my][mx]

    local_scores = pafXs * vx + pafYs * vy
    thidxs = local_scores > Inter_Threashold
    score = sum(local_scores * thidxs)
    count = sum(thidxs)
    return score, count


def human_conns_to_human_parts(human_conns, heatMat):
    human_parts = defaultdict(lambda: None)
    for conn in human_conns:
        human_parts[conn['partIdx'][0]] = (
            conn['partIdx'][0],  # part index
            (conn['coord_p1'][0] / heatMat.shape[2], conn['coord_p1']
             [1] / heatMat.shape[1]),  # relative coordinates
            heatMat[conn['partIdx'][0], conn['coord_p1']
                    [1], conn['coord_p1'][0]]  # score
        )
        human_parts[conn['partIdx'][1]] = (
            conn['partIdx'][1],
            (conn['coord_p2'][0] / heatMat.shape[2],
             conn['coord_p2'][1] / heatMat.shape[1]),
            heatMat[conn['partIdx'][1], conn['coord_p2']
                    [1], conn['coord_p2'][0]]
        )
    return human_parts


def draw_to_image(img, human_list):
    img_copied = np.copy(img)
    image_h, image_w = img_copied.shape[:2]
    centers = {}
    for human in human_list:
        part_idxs = human.keys()

        # draw point
        for i in range(18):
            if i not in part_idxs:
                continue
            part_coord = human[i][1]
            center = (int(part_coord[0] * image_w + 0.5),
                      int(part_coord[1] * image_h + 0.5))
            centers[i] = center
            cv2.circle(img_copied, center, 2,
                       CocoColors[i], thickness=2, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in part_idxs or pair[1] not in part_idxs:
                continue

            img_copied = cv2.line(
                img_copied, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 2)
    return img_copied


# if __name__ == "__main__":
    #  for local test:
    # import paf
    # import heat
    # pafs = paf.paf
    # heatmaps = heat.heat
    # img = cv2.imread('./resource/test.JPG')
    # c = estimate_pose(heatmaps,pafs)
    # cv2.imshow('test',draw_to_image(img,c))
