import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from keras import backend

#%%Prepare the dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
rows, cols = 28,28

if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, rows, cols)
    x_test = x_test.reshape(x_test.shape[0], 1, rows, cols)
    inpx = (1, rows, cols)

else:
    x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
    x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)
    inpx = (rows, cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#%% More detailed CNN model
def convolve2d(image, kernel, stride=1, p=0): #CNN modelinin kilit katmanı
    image = np.array(image)
    kernel = np.array(kernel)
    kernel = np.flipud(np.fliplr(kernel))

    if p > 0:
        image = np.pad(image, ((p, p), (p,p)), mode='constant')
    
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape

    out_height = int((i_height - k_height) / stride) + 1
    out_width = int((i_width - k_width) / stride) + 1

    output = np.zeros((out_height, out_width))

    for y in range(0, out_height):
        for x in range(0, out_width):
            region = image[y*stride:y*stride+k_height, x*stride:x*stride+k_width] 
            output[y, x] = np.sum(region*kernel)
    
    return output

class convlayer: 
    def __init__(self, fil_num, fil_size):
        self.fil_num = fil_num
        self.fil_size = fil_size
        self.filters = np.random.randn(fil_num, fil_size, fil_size) / 9
    
    def iterate_regions(self, image):
        h, w = image.shape
        for i in range (h - self.fil_size + 1):
            for j in range(w - self.fil_size + 1):
                region = image [i: i+self.fil_size, j:j+self.fil_size]
                yield i,j,region # yield allows a function to pause its execution and yield a value to the claler without losing its state
    
    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros(h - self.fil_size + 1, w - self.fil_size + 1, self.fil_num)

        for i, j, region in self.iterate_regions(input):
            for f in range(self.fil_num):
                output[i,j, f] = np.sum(region*self.filters[f])
        
        return output
    
def relu(x):
    return np.maximum(0,x)

# there are many types of pooling layers in different CNN architectures, kernel içindeki max değeri alıp yeni kernel değeri yapıyoruz
def max_pooling(inout, size=2, stride=2):
    h,w, fil_num = input.shape
    outh= h // size
    outw = w // size
    output = np.zeros((outh, outw, fil_num))

    for f in range(fil_num):
        for i in range ( 0, h, stride):
            for j in range(0, w, stride):
                region = input[i:i+size, j:j+size, f]
                if region.shape == (size, size):
                    output[i//stride, j//stride, f] = np.max(region)

    return output

def flatten(x):
    return x.reshape(-1)

class DenseLayer:
    def __init__(self, input_len, output_len):
        self.weights = np.random.randn(input_len, output_len) /input_len
        self.biases = np.zeros(output_len) # zeros() returns a new array of given shape and type, filled with zeros 
    
    def forward(self, x):
        self.last_input = x
        return x @ self.weight + self.biases
    
    def backward(self, dLd_out, learning_rate):
        dLd_w = self.last_input[:, None] @ dLd_out[None, :]
        dLd_input = dLd_out @ self.weights.T

        # Gradient descent 
        self.weights -= learning_rate * dLd_w
        self.biases -=  learning_rate * dLd_out

        return dLd_input
    

def softmax(x):
    exps = np.exp(x-np.max(x))
    return exps / np.sum(exps)

def cross_entropy_loss(pred,label):
    return -np.log(pred[label] + 1e-7)

def cross_entropy_grad(pred, label):
    grad = pred.copy()
    grad[label] -= 1
    return grad

class SimpleCNN:
    def __init__(self):
        self.conv = convlayer(8, 3)
        self.fc = DenseLayer(13 * 13 * 8, 10)  # after 2x2 pooling

    def forward(self, img):
        out = self.conv.forward(img)
        out = relu(out)
        out = max_pooling(out)
        out = flatten(out)
        out = self.fc.forward(out)
        return out

    def train(self, img, label, lr=0.005):
        # Forward
        out = self.conv.forward(img)
        out = relu(out)
        out = max_pooling(out)
        flat = flatten(out)
        logits = self.fc.forward(flat)
        probs = softmax(logits)

        # Loss
        loss = cross_entropy_loss(probs, label)

        # Backward
        grad = cross_entropy_grad(probs, label)
        grad = self.fc.backward(grad, lr)
        # Conv backward eksik – biz şimdilik sadece FC katmanını optimize ediyoruz

        return loss, np.argmax(probs) == label

model = SimpleCNN()
losses = []
acc = []

for i in range(100):  # sade test için küçük örnek
    img = x_train[i]
    label = y_train[i]

    loss, correct = model.train(img, label)
    losses.append(loss)
    acc.append(int(correct))

    if i % 10 == 0:
        print(f"{i}: loss={loss:.3f}, acc={np.mean(acc[-10:])*100:.2f}%")
