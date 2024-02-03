import cv2
import numpy as np

img1 = cv2.imread('book_Cover2.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('book_Cover.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1500)
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key = lambda x:x.distance)

matching_result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:18], None)

#for d in descriptors1:
 #   print(d)
#cv2.imshow("Image1", img1)
#cv2.imshow("Image2", img2)

cv2.imshow("Matching result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()