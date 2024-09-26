from numpy.random import binomial


class Arm:
    def __init__(self, p):
        self._p = p
        self.success = 0
        self.fail = 0
        self.arm_call_num = 0

    def reset(self):
        self.success = 0
        self.fail = 0
        self.success = 0

    def play(self):
        self.arm_call_num += 1
        result = binomial(n=1, p=self._p)
        if result == 1:
            self.success += 1
        else:
            self.fail += 1
        return result
