import time

class TimeMetrics:
    def __init__(self):
        pass

    def millis(self):
        return int(round(time.time() * 1000))