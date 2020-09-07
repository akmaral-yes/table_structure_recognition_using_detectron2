def intersection_lines(line1, line2):
    """
    Compute the length of the intersection of two lines
    """
    line1_start, line1_end = line1
    line2_start, line2_end = line2
    if line1_start >= line2_end or line2_start >= line1_end:
        return 0
    return min(line1_end, line2_end) - max(line1_start, line2_start)
