import numpy as np
import matplotlib.pyplot as plt

# loss function
def cross_entropy(y, pred_y):
    """
        loss function
        
        Args:
            y: np.array((n,1))
            pred_y: np.array((n,1))
        
        Returns:
            loss: float
    
    """
    size = y.shape[0]
    pred_y = np.clip(pred_y, 1e-15, 1 - 1e-15)
    loss = -(1/size)*(y.T@np.log(pred_y) + ((1-y).T@np.log(1-pred_y)))
    return loss[0][0]



# activation functions and their derivatives
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def without_act(x):
    return x
def derivative_without_act(x):
    return np.ones_like(x)

def relu(x):
    return np.maximum(0,x)

def derivative_relu(x):
    return np.where(x < 0, 0, 1) 

def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1-np.tanh(x)**2

# Input function
def generate_linear(n=100):
    import numpy as np
    pts = np.random.uniform(0, 1, (n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1)

def show_result(x, y, pred_y):
    """
        plot the results of Ground truth and predicted result

        Args: 
            x: np.array [[x1,y1], [x2,y2]...]
            y: np.array [[0],[1], ...]
            pred_y: np.array [[0], [1], ...]
        Returns:
            None
    
    """
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predicted result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

class SimpleNN:
    """
        Simple neural network with two hidden layers        
    """

    def __init__(self, h1_size, h2_size, lr=0.01, iters=1000, activationf='sigmoid', optimizer='gd'):
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.lr = lr
        self.iters = iters
        self.activationf = activationf
        self.optimizer = optimizer
        self.lossf = cross_entropy
        

        # define activiation function
        if self.activationf == 'sigmoid':
            self.act = sigmoid
            self.diff_act = derivative_sigmoid        
        elif self.activationf == 'relu':
            self.act = sigmoid
            self.diff_act = derivative_sigmoid    
        elif self.activationf == 'tanh':
            self.act = tanh
            self.diff_act = derivative_tanh
        else:
            self.act = without_act
            self.diff_act = derivative_without_act

    def init_params(self):
        """
            Define initial weights of our model
        """
        self.W = [np.random.randn(2, self.h1_size), np.random.randn(self.h1_size, self.h2_size), np.random.randn(self.h2_size, 1)]
        self.b = [np.zeros((1, self.h1_size)), np.zeros((1, self.h2_size)), np.zeros((1, 1))]
        self.z = [None, None, None]
        self.a = [None, None, None]
        self.momentum_w = [np.zeros_like(w) for w in self.W]
        self.adasquare_w = [np.zeros_like(w) for w in self.W]
        self.adam_w1 = [np.zeros_like(w) for w in self.W]
        self.adam_w2 = [np.zeros_like(w) for w in self.W]

    def update(self, dw_2, dw_1, dw_0, db_2, db_1, db_0):
        dw = [dw_0, dw_1, dw_2]
        db = [db_0, db_1, db_2]
        if self.optimizer == 'gd':
            for i in range(len(self.W)):
                self.W[i] -= self.lr*dw[i]
                self.b[i] -= self.lr*db[i]

        elif self.optimizer == 'mm':
            # default setting: momentum = 0.9
            for i in range(len(self.W)):
                self.momentum_w[i] = 0.9 * self.momentum_w[i] + self.lr * dw[i]
                self.W[i] -= self.momentum_w[i]

            for i in range(len(self.b)):
                self.b[i] -= self.lr*db[i]
        
        elif self.optimizer == 'adagrad':
            # default setting: epsilon = 0.00001
            for i in range(len(self.W)):
                self.adasquare_w[i] += dw[i]**2
            self.W[i] -= (self.lr / (np.sqrt(self.adasquare_w[i]) + 0.00001)) * dw[i]

            for i in range(len(self.b)):
                self.b[i] -= self.lr*db[i]

        elif self.optimizer == 'adam':
            # using default seeting of adam
            beta1 = 0.9  
            beta2 = 0.999  
            epsilon = 1e-8  
            t = 0  
    
            for i in range(len(self.W)):
                t += 1
                self.adam_w1[i] = beta1 * self.adam_w1[i] + (1 - beta1) * dw[i]
                self.adam_w2[i] = beta2 * self.adam_w2[i] + (1 - beta2) * dw[i] ** 2
                moment1 = self.adam_w1[i] / (1 - beta1 ** t)
                moment2= self.adam_w2[i] / (1 - beta2 ** t)
                self.W[i] -= (self.lr / (np.sqrt(moment2) + epsilon)) * moment1

            for i in range(len(self.b)):
                self.b[i] -= self.lr*db[i]
    
    def foward(self, X):
        """
            input:x np.array(n, 2)
            W_0@x+b0 = z0
            nonlinear(z0) = a0

            W_1@a0+b1 = z1
            nonlinear(z1) = a1

            W_2@a_1+b2 = z2
            nonlinear(z2) = a2 
            loss = cross_entropy(a2, y)
        """

        # input layer
        x = X
        # first hidden layer
        self.z[0] = x@self.W[0]+self.b[0] # (n, h1)
        self.a[0] = self.act(self.z[0])
        # second layer
        self.z[1] = self.a[0]@self.W[1]+self.b[1] # (n,h2)
        self.a[1] = self.act(self.z[1])
        # output layer
        self.z[2] = self.a[1]@self.W[2]+self.b[2] # (nx1)
        self.a[2] = sigmoid(self.z[2])
        return self.a[2]
        
    def backward(self, pred_y, X, Y):
        """
            da_2 = (first derivative of loss function)dy
            dz_2 = da_2*(first derivative of sigmoid func)
            dw_2 = dz_2@a[2]
            db_2 = dz_2@1
        """
                
        # prevent denominator = 0
        pred_y = np.clip(pred_y, 1e-15, 1 - 1e-15)

        da_2 = -(Y/pred_y) + (1-Y)/(1-pred_y)
        dz_2 = da_2*derivative_sigmoid(self.a[2])
        dw_2 = self.a[1].T@dz_2
        db_2 = np.sum(dz_2, axis=0, keepdims=True)

        da_1 = dz_2@self.W[2].T
        dz_1 = da_1*self.diff_act(self.a[1])
        dw_1 = self.a[0].T@dz_1
        db_1 = np.sum(dz_1, axis=0, keepdims=True)

        da_0 = dz_1@self.W[1].T
        dz_0 = da_0*self.diff_act(self.a[0])
        dw_0 = X.T@dz_0
        db_0 = np.sum(dz_0, axis = 0, keepdims=True)

        # update params
        self.update(dw_2, dw_1, dw_0, db_2, db_1, db_0)


    def train(self, X, Y):
        """
            Test the model based on the same sample space

            Args: 
                X: np.array [[x1,y1], [x2,y2]...]
                Y: np.array [[0],[1], ...]
        
            Returns:
                training result: np.array [[0],[1], ...]
        """
        self.init_params()
        loss_list = []
        for i in range(self.iters):
            pred_y = self.foward(X)
            loss = self.lossf(Y, pred_y)
            self.backward(pred_y, X, Y)
            
            # record training result
            print(f'epoch {i} loss: {loss:.5f}')
            loss_list.append(loss)
            
            # breakpoint
            if loss < 0.00001:
                break
        
        print(f"stop at iter:{i} with loss:{loss}")
        plt.plot(np.arange(i+1), loss_list)
        plt.title("Learning curve")
        plt.show()
        return np.round(pred_y)
    
    def test(self, X, Y):
        """
            Test the model based on the same sample space

            Args: 
                X: np.array [[x1,y1], [x2,y2]...]
                Y: np.array [[0],[1], ...]
        
            Returns:
                predicted result: np.array [[0],[1], ...]
        """

        pred_y = self.foward(X)
        loss = self.lossf(Y, pred_y)
        print(pred_y)
        
        # print the information of each estimation and calculate the accuracy
        correct = 0       
        for i in range(pred_y.shape[0]):            
            print(f'Iter{i}|   Ground truth: {Y[i][0]:.1f}|   prediction: {pred_y[i][0]:.5f}')
            correct += 1 if np.round(pred_y[i]) == Y[i] else 0
        correct_rate = correct/pred_y.shape[0]*100

        print(f'loss={loss:.5f} accuracy={correct_rate:.2f}%')
        return np.round(pred_y)
    
if __name__ == "__main__":
    
    # param 
    inputs = 'XOR' # linear or XOR
    trans = 'linear' # CNN or Linear
    activationf = 'relu' # sigmoid, tanh, relu, w/o
    iters = 100000 # training iterations
    lr = 0.01 # learning rate: 0.1, 0.01, 0.001, 1
    h1_size = 5 # num of units in first hidden layer
    h2_size = 5 # num of units in second hidden layer
    optimizer = 'gd' # gd: gradient descent / mm: monmentum / adagrad: Adagrad / adam: Adam

    # read the data
    if inputs == 'linear':
        train_x, train_y = generate_linear(n=100)
        test_x, test_y = train_x.copy(), train_y.copy()
    else:
        train_x, train_y = generate_XOR_easy()
        test_x, test_y = train_x.copy(), train_y.copy()

    # training
    model = SimpleNN(h1_size=h1_size, h2_size=h2_size, lr=lr, iters=iters, activationf=activationf, optimizer = 'gd')
    model.train(train_x, train_y)
    pred_y = model.test(test_x, test_y)
    show_result(train_x, train_y, pred_y) 
