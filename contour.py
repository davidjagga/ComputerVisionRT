import cv2
image = cv2.imread("images/face4.png")
#gray scale image
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#binary threshold
ret, thresh = cv2.threshold(grayImage, 150, 255, cv2.THRESH_BINARY)
#show Image
# cv2.imshow("Binary Image", thresh)
# cv2.waitKey(0)
#save image
# cv2.imwrite('img1thresh.jpg', thresh)
# cv2.destroyAllWindows()
# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1,
                 lineType=cv2.LINE_AA)
# see the results
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
#cv2.imwrite('contours_none_image1.jpg', image_copy)
cv2.destroyAllWindows()
