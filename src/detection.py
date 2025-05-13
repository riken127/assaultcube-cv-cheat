from scipy.spatial import distance

def get_centroids(detections, classes, valid_classes=[2, 3]):
    centroids = []
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = box
        class_id = int(classes[i])
        if class_id in valid_classes:
            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)
            centroids.append((centroid_x, centroid_y, class_id))
    return centroids

def select_target(centroids, screen_center):
    headshots = [c for c in centroids if c[2] == 2]
    bodies = [c for c in centroids if c[2] == 3]
    if headshots:
        return min(headshots, key=lambda c: distance.euclidean(screen_center, (c[0], c[1])))
    elif bodies:
        return min(bodies, key=lambda c: distance.euclidean(screen_center, (c[0], c[1])))
    return None