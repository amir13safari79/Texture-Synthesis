# import libraries:
import cv2

def subplot(org_texture, synthesized_texture):
    # add border to org_texture for use cv2.hconcat:
    white = [255, 255, 255]     # border color
    org_texture_border = cv2.copyMakeBorder(org_texture, synthesized_texture.shape[0] - org_texture.shape[0],0,0,synthesized_texture.shape[1] - org_texture.shape[1],cv2.BORDER_CONSTANT,value=white)
    result = cv2.hconcat((org_texture_border, synthesized_texture))
    return result