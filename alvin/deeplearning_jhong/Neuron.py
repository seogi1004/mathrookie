class Neuron:
    def __init__(self, w, b):
        self.w_ = w
        self.b_ = b

    def getAct(self, x):
        return x

    def feedForward(self, input):
        sigma = self.w_ * input + self.b_
        return self.getAct(sigma)


my_neuron = Neuron()
my_neuron.w_ = 2.0
my_neuron.b_ = 1.0

print(my_neuron.feedForward(1.0))
print(my_neuron.feedForward(2.0))
print(my_neuron.feedForward(3.0))