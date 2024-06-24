import cv2 as cv
import imutils
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def mix_match(leftImage, warpedImage):
        
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]


        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  \
                        np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
                        # print "BLACK"
                        # instead of just putting it with black, 
                        # take average of all nearby values and avg it.
                        warpedImage[j,i] = [0, 0, 0]
                    else:
                        if(np.array_equal(warpedImage[j,i],[0,0,0])):
                            # print "PIXEL"
                            warpedImage[j,i] = leftImage[j,i]
                        else:
                            if not np.array_equal(leftImage[j,i], [0,0,0]):
                                bl,gl,rl = leftImage[j,i]                               
                                warpedImage[j, i] = [bl,gl,rl]
                except:
                    pass
        cv.imshow("waRPED mix", warpedImage)
        cv.waitKey()
        return warpedImage

def detect_kp_features(image):

    descriptor = cv.SIFT_create()
    key_pts, features = descriptor.detectAndCompute(image, None)
    key_pts_array = []

    for key_pt in key_pts:
        key_pts_array.append((round(key_pt.pt[0],2), round(key_pt.pt[1],2)))

    return (key_pts_array, features)

def match_key_points(key_pts_A, key_pts_B, features_A, features_B, ratio):

    points_A = []
    points_B = []

    match = cv.DescriptorMatcher_create("FlannBased")
    rough_matches = match.knnMatch(features_A, features_B, 2)

    for match in rough_matches:
        if match[0].distance < match[1].distance * ratio:
            points_A.append(key_pts_A[match[0].queryIdx])
            points_B.append(key_pts_B[match[0].trainIdx])

    return (points_A, points_B)

def get_correspondence(img1, img2):
    """
    Args
    img1: left image
    img2: right image

    Output
    points1: coordinates of matched keypoints in img1 | Shape (N, 2), where N is the number of keypoints detected in img1, having corresponding matched keypoint in img2
    points2: coordinates of matched keypoints in img2 | Shape (N, 2)
    """
    ## TODO: Complete this function
    key_pts_A, features_A = detect_kp_features(img1)
    key_pts_B, features_B = detect_kp_features(img2)

    # match features between the two images
    points1, points2 = match_key_points(key_pts_A, key_pts_B, features_A, features_B, ratio = 0.1)

    return points1, points2

def visualize_keypoints_and_correspondences(img1, img2, points1, points2):
    """
    Args
    img1: left image
    img2: right image
    points1: coordinates of matched keypoints in img1 | Shape (N, 2), where N is the number of keypoints detected in img1, having corresponding matched keypoint in img2
    points2: coordinates of matched keypoints in img2 | Shape (N, 2)
    """
    # Convert the coordinates to integers
    points1_int = np.int32(points1)
    points2_int = np.int32(points2)

    # Find the size of the output canvas
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_height = max(h1, h2)
    canvas_width = w1 + w2

    # Create a blank canvas
    img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place the images on the canvas
    img[:h1, :w1, :] = img1
    img[:h2, w1:, :] = img2

    # Draw correspondences
    for i in range(len(points1_int)):
        point1 = tuple(points1_int[i])
        point2 = tuple(points2_int[i] + np.array([w1, 0]))  # Add width of img1 to x-coordinate of points2
        img = cv.line(img, point1, point2, (0, 255, 0), 1)

    plt.figure(figsize=(20, 10))
    plt.imshow(img)
    plt.title("Correspondences")
    plt.show()

def compute_matrix(points1, points2):

    rows = 2*len(points1)
    array = np.zeros(rows * 9)
    A = array.reshape(rows, 9)

    for i in range(len(points1)):

        x_s = points1[i][0]
        y_s = points1[i][1]
        x_d = points2[i][0]
        y_d = points2[i][1]

        A[2*i]   = np.array([x_s, y_s, 1, 0, 0, 0, -x_d * x_s, -x_d * y_s, -x_d])
        A[2*i+1] = np.array([0, 0, 0, x_s, y_s, 1, -y_d * x_s, -y_d * y_s, -y_d])

    A = np.array(A)

    return A

def get_homography(points1, points2):
    """
    Args
    points1: coordinates of keypoints in img1 | Shape (N, 2), where N is the number of keypoints detected in img1, which has corresponding matched keypoint in img2
    points2: coordinates of keypoints in img2 | Shape (N, 2)

    Output
    H: homography matrix | Shape (3, 3)

    NOTE - Hint:
    You can use a RANSAC-based method to robustly estimate the homography matrix H.
    """
    ## TODO: Complete this function
    
    A = compute_matrix(points1[0:50], points2[0:50])
    combined_matrix = np.transpose(A) @ A
    U, S, V = np.linalg.svd(combined_matrix)
    h_vector = V[-1, :]
    h_matrix = h_vector.reshape(3, 3) / h_vector[-1]

    print(h_matrix)

    return h_matrix
    
def stitch(img1, img2, H):
    """
    Args
    img1: left image
    img2: right image
    H: homography matrix

    Output
    img: stitched image

    NOTE - Hint: 
    The homography matrix H computed from get_homography() does not account for translation needed to map the entire output into a single canvas. 
    Hence, take the min and max coordinate ranges (or dimensions) of left and right images to estimate the bounds (min - (x_min, y_min) and max coordinates - (x_max, y_max)) for the final canvas image.
    You might need to incorporate this translation of (x_min, y_min) for warping the final image to the canvas.
    """
    ## TODO: Complete this function
    
    h1, w1 = img1.shape[0], img1.shape[1]
    h2, w2 = img2.shape[0], img2.shape[1]

    c1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype = float)
    c2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype = float)

    c1 = c1.reshape(-1, 1, 2)
    c2 = c2.reshape(-1, 1, 2)

    transformed_c1 = cv.perspectiveTransform(c1, H)

    corners = np.concatenate((transformed_c1, c2))

    x_min = np.min(corners[:, :, 0])
    y_min = np.min(corners[:, :, 1])

    x_max = np.max(corners[:, :, 0])
    y_max = np.max(corners[:, :, 1])

    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    H_translation = translation_matrix @ H

    result1 = cv.warpPerspective(img1, H_translation, (int(x_max - x_min), int(y_max - y_min)))
    result2 = cv.warpPerspective(img2, translation_matrix, (int(x_max - x_min), int(y_max - y_min)))

    #blended_image = cv.addWeighted(result1, 0.5, result2, 0.5, 0)

    return result1, result2

if __name__ == "__main__":

    image1 = cv.imread("left.JPG")
    image2 = cv.imread("right.JPG")

    points1, points2 = get_correspondence(image1, image2)
    visualize_keypoints_and_correspondences(image1, image2, points1, points2)
    
    H = get_homography(points1, points2)
    result1, result2 = stitch(image1, image2, H)

    output = mix_match(result1, result2)
    cv.imwrite("output.jpg", output)
    plt.imshow(output)
    plt.show()

