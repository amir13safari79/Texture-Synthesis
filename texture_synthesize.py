# import libraries and functions:
import numpy as np
import cv2
from find_min_cut import find_min_cut
from find_joint_strip import find_joint_strip


def texture_synthesize(org_texture, output_size=2500, patch_size=120, strip_width=30):
    # determine length and width of patch and strip according to inputs:
    patch_h, patch_w = patch_size, patch_size
    strip_h, strip_w = patch_size, strip_width

    # find patch_count in each row of synthesized_texture according to output_size:
    patch_count = int(np.ceil((output_size - strip_w) / (patch_h - strip_w)))
    synthesized_size = (patch_size - strip_width) * patch_count + strip_w
    synthesized_texture = np.zeros((synthesized_size, synthesized_size, 3))

    # find first patch of synthesized_texture randomly:
    org_texture_h, org_texture_w, _ = org_texture.shape
    start_h = np.random.randint(0, org_texture_h - patch_h)
    start_w = np.random.randint(0, org_texture_w - patch_w)
    synthesized_texture[0:patch_h, 0:patch_w] = org_texture[start_h:start_h + patch_h, start_w:start_w + patch_w]

    # Synthesis other patchs of synthesized_texture:
    for i in range(1, int(patch_count ** 2)):
        # Synthesis first row of synthesized_texture:
        if (i // patch_count == 0):
            synthesized_strip_start_h = 0
            synthesized_strip_start_w = (patch_w - strip_w) * (i % int(patch_count))
            synthesized_strip_right = synthesized_texture[0:strip_h,
                                      synthesized_strip_start_w:synthesized_strip_start_w + strip_w]

            # use template matching with SSD method to find 10 most similar strips to synthesized_strip_right
            ssd_results = cv2.matchTemplate(np.uint8(org_texture[:, 0:org_texture_w - (strip_h - strip_w)]),
                                            np.uint8(synthesized_strip_right), cv2.TM_SQDIFF)
            rand_ind = np.random.randint(1, 10)
            ssd_select = np.sort(ssd_results.flatten())[rand_ind]
            ssd_ind = (np.where(ssd_results == ssd_select)[0][0], np.where(ssd_results == ssd_select)[1][0])
            org_strip_left = org_texture[ssd_ind[0]:ssd_ind[0] + strip_h, ssd_ind[1]:ssd_ind[1] + strip_w]

            # find ssd matrix between synthesized_strip_right and org_strip_left
            org_strip_left_gray = cv2.cvtColor(org_strip_left, cv2.COLOR_BGR2GRAY)
            synthesized_strip_right_gray = cv2.cvtColor(np.uint8(synthesized_strip_right), cv2.COLOR_BGR2GRAY)
            strip_ssd = (org_strip_left_gray - synthesized_strip_right_gray) ** 2

            # find vertically minimum cut for strip_ssd:
            min_cut_ind = find_min_cut(strip_ssd)
            joint_strip = find_joint_strip(synthesized_strip_right, org_strip_left, min_cut_ind)

            # replace joint_strip with synthesized_strip_right:
            synthesized_texture[0:strip_h, synthesized_strip_start_w:synthesized_strip_start_w + strip_w] = joint_strip
            synthesized_texture[0:patch_h, synthesized_strip_start_w + strip_w:synthesized_strip_start_w + patch_w] = (
                org_texture[ssd_ind[0]:ssd_ind[0] + patch_h, ssd_ind[1] + strip_w:ssd_ind[1] + patch_w]
            )

        # Synthesis first column of synthesized_texture:
        elif (i % patch_count == 0):
            synthesized_strip_start_h = int((patch_h - strip_w) * (i // patch_count))
            synthesized_strip_start_w = 0
            synthesized_strip_down = synthesized_texture[synthesized_strip_start_h:synthesized_strip_start_h + strip_w,
                                     0:synthesized_strip_start_w + strip_h]

            # use template matching with SSD method to find 10 most similar strips to synthesized_strip_down
            ssd_results = cv2.matchTemplate(np.uint8(org_texture[0:org_texture_h - (strip_h - strip_w), :]),
                                            np.uint8(synthesized_strip_down), cv2.TM_SQDIFF)
            rand_ind = np.random.randint(1, 10)
            ssd_select = np.sort(ssd_results.flatten())[rand_ind]
            ssd_ind = (np.where(ssd_results == ssd_select)[0][0], np.where(ssd_results == ssd_select)[1][0])
            org_strip_up = org_texture[ssd_ind[0]:ssd_ind[0] + strip_w, ssd_ind[1]:ssd_ind[1] + strip_h]

            # find ssd matrix between synthesized_strip_down and org_strip_up
            org_strip_up_gray = cv2.cvtColor(org_strip_up, cv2.COLOR_BGR2GRAY)
            synthesized_strip_down_gray = cv2.cvtColor(np.uint8(synthesized_strip_down), cv2.COLOR_BGR2GRAY)
            strip_ssd = (org_strip_up_gray - synthesized_strip_down_gray) ** 2

            # find horizontal minimum cut for strip_ssd:
            min_cut_ind = find_min_cut(strip_ssd)
            joint_strip = find_joint_strip(synthesized_strip_down, org_strip_up, min_cut_ind)

            # replace joint_strip with synthesized_strip_down:
            synthesized_texture[synthesized_strip_start_h:synthesized_strip_start_h + strip_w,
            0:synthesized_strip_start_w + strip_h] = joint_strip
            synthesized_texture[synthesized_strip_start_h + strip_w:synthesized_strip_start_h + patch_h, 0:patch_w] = (
                org_texture[ssd_ind[0] + strip_w:ssd_ind[0] + patch_h, ssd_ind[1]:ssd_ind[1] + patch_w]
            )

        # Synthesis other patchs of synthesized_texture:
        else:
            i_quotient = i // int(patch_count)
            i_res = i % int(patch_count)

            # find L shape strip:
            synthesized_strip_start_h = int((patch_h - strip_w) * i_quotient)
            synthesized_strip_start_w = (patch_w - strip_w) * i_res
            synthesized_strip_full = synthesized_texture[synthesized_strip_start_h:synthesized_strip_start_h + patch_h,
                                     synthesized_strip_start_w:synthesized_strip_start_w + patch_w]

            # use template matching with SSD method to find 10 most similar L_strips to synthesized_strip_full with mask
            # produce mask for synthesized_strip_full:
            mask = np.zeros((patch_h, patch_w))
            mask[strip_w:strip_h, strip_w:strip_h] = 0
            mask[0:strip_w, 0:strip_h] = 1
            mask[0:strip_h, 0:strip_w] = 1
            mask = np.uint8(mask)

            ssd_results = cv2.matchTemplate(np.uint8(org_texture), np.uint8(synthesized_strip_full), cv2.TM_SQDIFF,
                                            None, mask)
            rand_ind = np.random.randint(1, 10)
            ssd_select = np.sort(ssd_results.flatten())[rand_ind]
            ssd_ind = (np.where(ssd_results == ssd_select)[0][0], np.where(ssd_results == ssd_select)[1][0])
            org_strip_full = org_texture[ssd_ind[0]:ssd_ind[0] + patch_h, ssd_ind[1]:ssd_ind[1] + patch_w]

            # split L shape of synthesized_strip_full and org_strip_full to vertical and horizontal strips:
            synthesized_strip_L_vertical = synthesized_strip_full[0:strip_h, 0:strip_w]
            synthesized_strip_L_horizontal = synthesized_strip_full[0:strip_w, 0:strip_h]

            org_strip_L_vertical = org_strip_full[0:strip_h, 0:strip_w]
            org_strip_L_horizontal = org_strip_full[0:strip_w, 0:strip_h]

            # find ssd matrix between synthesized_strip_Ls and org_strip_Ls:
            synthesized_strip_L_v_gray = cv2.cvtColor(np.uint8(synthesized_strip_L_vertical), cv2.COLOR_BGR2GRAY)
            org_strip_L_v_gray = cv2.cvtColor(np.uint8(org_strip_L_vertical), cv2.COLOR_BGR2GRAY)
            strip_ssd_L_v = (org_strip_L_v_gray - synthesized_strip_L_v_gray) ** 2

            synthesized_strip_L_h_gray = cv2.cvtColor(np.uint8(synthesized_strip_L_horizontal), cv2.COLOR_BGR2GRAY)
            org_strip_L_h_gray = cv2.cvtColor(np.uint8(org_strip_L_horizontal), cv2.COLOR_BGR2GRAY)
            strip_ssd_L_h = (org_strip_L_h_gray - synthesized_strip_L_h_gray) ** 2

            # find vertical and horizontal minimum cut for strip_ssd_L_v and strip_ssd_L_h respectively:
            min_cut_v_ind = find_min_cut(strip_ssd_L_v)
            min_cut_h_ind = find_min_cut(strip_ssd_L_h)

            # find joint_strip_v and joint_strip_h
            joint_strip_v = find_joint_strip(synthesized_strip_L_vertical, org_strip_L_vertical, min_cut_v_ind)
            joint_strip_h = find_joint_strip(synthesized_strip_L_horizontal, org_strip_L_horizontal, min_cut_v_ind)

            # replace joint_strip_v and joint_strip_h with synthesized_strip_L_vertical and synthesized_strip_L_horizontal respectively:
            synthesized_texture[synthesized_strip_start_h:synthesized_strip_start_h + strip_h,
            synthesized_strip_start_w:synthesized_strip_start_w + strip_w] = joint_strip_v
            synthesized_texture[synthesized_strip_start_h:synthesized_strip_start_h + strip_w,
            synthesized_strip_start_w:synthesized_strip_start_w + strip_h] = joint_strip_h

            synthesized_texture[synthesized_strip_start_h + strip_w:synthesized_strip_start_h + strip_h,
            synthesized_strip_start_w + strip_w:synthesized_strip_start_w + patch_w] = (
                org_texture[ssd_ind[0] + strip_w:ssd_ind[0] + patch_h, ssd_ind[1] + strip_w:ssd_ind[1] + patch_w]
            )

    # Crop the output image to size output_size:
    synthesized_output = synthesized_texture[0:output_size, 0:output_size]

    return np.uint8(synthesized_output)