# import libraries and functions:
import cv2
from texture_synthesize import texture_synthesize
from subplot import subplot

# load original textures
# with BGR space:
org_texture1 = cv2.imread('org_images/texture02.png')
org_texture2 = cv2.imread('org_images/texture06.jpg')
org_texture3 = cv2.imread('org_images/texture-dani.png')
org_texture4 = cv2.imread('org_images/knitting_pattern.jpg')

# find synthesized_texture of org_textures:
synthesized_texture1 = texture_synthesize(org_texture1)
synthesized_texture2 = texture_synthesize(org_texture2)
synthesized_texture3 = texture_synthesize(org_texture3)
synthesized_texture4 = texture_synthesize(org_texture4)

# find subplot of original and synthesized textures and save them:
res11 = subplot(org_texture1, synthesized_texture1)
res12 = subplot(org_texture2, synthesized_texture2)
res13 = subplot(org_texture3, synthesized_texture3)
res14 = subplot(org_texture4, synthesized_texture4)

cv2.imwrite('res_images/texture02_result.jpg', res11)
cv2.imwrite('res_images/texture06_result.jpg', res12)
cv2.imwrite('res_images/texture-dani_result.jpg', res13)
cv2.imwrite('res_images/knitting_pattern_result.jpg', res14)