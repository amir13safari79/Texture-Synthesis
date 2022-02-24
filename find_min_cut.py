# import libraries:
import numpy as np
import matplotlib.pyplot as plt
import cv2

def find_min_cut(input_ssd):
    ssd_h, ssd_w = input_ssd.shape
    direction = 'vertical'
    if ssd_h < ssd_w:
        direction = 'horizontal'

    if direction == 'vertical':
        dp_cost = np.zeros((ssd_h, ssd_w))
        dp_next_col = np.zeros((ssd_h, ssd_w))

        # initial last row of dp_cost = cost with dp_next_col = 0
        for i in range(ssd_w):
            dp_cost[ssd_h-1, i] = input_ssd[ssd_h-1, i]
            dp_next_col[ssd_h-1, i] = 0

        # fill other rows of dp_cost and dp_next_col:
        for i in np.arange(ssd_h-2, -1, -1):
            for j in range(ssd_w):
                min_cost = np.inf
                min_cost_ind = 0
                for k in range(max(0,j-1), min(j+2,ssd_w)):
                    if dp_cost[i+1, k] < min_cost:
                        min_cost = dp_cost[i+1, k]
                        min_cost_ind = k

                dp_cost[i, j] = input_ssd[i, j] + min_cost
                dp_next_col[i, j] = min_cost_ind

        # find minimum cut(index of each row that makes minimum cost):
        # produce min_cut_ind:
        min_cut_ind = []

        # fill min_cut_ind with find first columns:
        minimum_cost = np.amin(dp_cost[0, :])
        min_cut_ind.append(np.where(dp_cost[0, :] == minimum_cost)[0][0])

        for i in range(0, ssd_h - 1):
            min_cut_ind.append(int(dp_next_col[i, min_cut_ind[i]]))

    if direction == 'horizontal':
        dp_cost = np.zeros((ssd_h, ssd_w))
        dp_next_row = np.zeros((ssd_h, ssd_w))

        # initial last column of dp_cost = cost with dp_next_row = 0
        for i in range(ssd_h):
            dp_cost[i, ssd_w-1] = input_ssd[i, ssd_w-1]
            dp_next_row[i, ssd_w-1] = 0

        # fill other columns of dp_cost and dp_next_row:
        for j in np.arange(ssd_w-2, -1, -1):
            for i in range(ssd_h):
                min_cost = np.inf
                min_cost_ind = 0
                for k in range(max(0,i-1), min(i+2,ssd_h)):
                    if dp_cost[k, j+1] < min_cost:
                        min_cost = dp_cost[k, j+1]
                        min_cost_ind = k

                dp_cost[i, j] = input_ssd[i, j] + min_cost
                dp_next_row[i, j] = min_cost_ind

        # find minimum cut(index of each column that makes minimum cost):
        # produce min_cut_ind:
        min_cut_ind = []

        # fill min_cut_ind with find first row:
        minimum_cost = np.amin(dp_cost[:, 0])
        min_cut_ind.append(np.where(dp_cost[:, 0] == minimum_cost)[0][0])

        for j in range(0, ssd_w - 1):
            min_cut_ind.append(int(dp_next_row[min_cut_ind[j], j]))

    return min_cut_ind