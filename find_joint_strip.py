# import libraries:
import numpy as np
import matplotlib.pyplot as plt
import cv2

def find_joint_strip(synthesized_strip, org_strip, min_cut_ind):
    strip_h, strip_w, _ = synthesized_strip.shape
    direction = 'vertical'
    if strip_h < strip_w:
        direction = 'horizontal'

    if direction == 'vertical':
        joint_strip = np.zeros((strip_h, strip_w, 3))
        for i in range(strip_h):
            for j in range(strip_w):
                if (j <= min_cut_ind[i]):
                    joint_strip[i, j, :] = synthesized_strip[i, j, :]
                else:
                    joint_strip[i, j, :] = org_strip[i, j, :]

    if direction == 'horizontal':
        joint_strip = np.zeros((strip_h, strip_w, 3))
        for j in range(strip_w):
            for i in range(strip_h):
                if (i <= min_cut_ind[j]):
                    joint_strip[i, j, :] = synthesized_strip[i, j, :]
                else:
                    joint_strip[i, j, :] = org_strip[i, j, :]

    return joint_strip