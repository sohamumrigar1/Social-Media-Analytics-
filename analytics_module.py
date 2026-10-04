def calculate_engagement(likes, comments, shares):
    return likes + comments + shares


def engagement_category(rate):
    if rate >= 0.10:
        return "High"
    elif rate >= 0.05:
        return "Medium"
    else:
        return "Low"