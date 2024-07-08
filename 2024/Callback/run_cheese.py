import cheese

def report_cheese(name):
    """ callback func"""
    print("Found cheese: " + name)

cheese.find(report_cheese)