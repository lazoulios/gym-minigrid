def point_in_ellipse(cx, cy, rx, ry):
    def wrapper(x, y):
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
    return wrapper