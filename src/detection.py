from scipy.spatial import distance

# calculate the centroids of detected bounding boxes.
def get_centroids(detections, classes, valid_classes=[2, 3]):
    centroids = []

    # iterate over each detection and its corresponding class
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = box
        class_id = int(classes[i])

        # check if the class is head or enemy
        if class_id in valid_classes:
            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)

            # append the centroid coordinates
            centroids.append((centroid_x, centroid_y, class_id))
    return centroids

# this function selects the target based on proximity to the center of the screen
def select_target(centroids, screen_center):
    
    # filter out centroids corresponding to headshots
    headshots = [c for c in centroids if c[2] == 2]
    bodies = [c for c in centroids if c[2] == 3]

    # if there are heads, go for the headshot, else go for bodies
    if headshots:
        return min(headshots, key=lambda c: distance.euclidean(screen_center, (c[0], c[1])))
    elif bodies:
        return min(bodies, key=lambda c: distance.euclidean(screen_center, (c[0], c[1])))
    return None