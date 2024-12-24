# ai - artificial intelligence - ИИ

# 1950-1960 Alan Turing test

# 1970-1980 logic expert systems + neuron's web(perceptron)

# 1990-2000 statistics + static methods

# 2010- н.в. deep learning neuron's web(Multi-layer perceptron MLP), RL

# supervised learning

# unsupervised learning

# RL

# Deep learning

# import numpy as np
#
#
# class Perceptron:
#     def __init__(self, learning_rate=0.01, n_iterations=1000):
#         self.lr = learning_rate
#         self.n_ter = n_iterations
#         self.weights = None
#         self.bias = None
#
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#
#         self.weights = np.zeros(n_features)
#         self.bias = 0.0
#
#         for _ in range(self.n_ter):
#             for idx, x_i in enumerate(X):
#                 linear_output = np.dot(x_i, self.weights) + self.bias
#                 y_pred = np.where(linear_output > 0, 1, -1)
#
#                 if y[idx] != y_pred:
#                     update = self.lr * y[idx]
#                     self.weights = x_i*update
#                     self.bias += update
#
#     def predict(self, X):
#         linear_output = np.dot(X, self.weights) + self.bias
#         y_pred = np.where(linear_output > 0, 1, -1)
#         return y_pred
#
#
# if __name__ == '__main__':
#     np.random.seed(42)
#
#     X1 = np.random.randn(50, 2) + np.array([2, 2])
#     X2 = np.random.randn(50, 2) + np.array([-2, -2])
#
#     X = np.vstack([X1, X2])
#     y = np.array([1]*50 + [-1]*50)
#
#     perceptron = Perceptron(learning_rate=0.02, n_iterations=5000)
#     perceptron.fit(X, y)
#
#     predictions = perceptron.predict(X)
#     accuracy = np.mean(predictions == y)
#     print(f"Точность: {accuracy}")
#


# AI - artificial intelligence


# 1950
# 1960-1970
# 1980
# 1990
# 2000 - н.в.

# Perceptron/MLP
# RL
# supervised learning/ unsupervised learning
# CNN
# Recurrent NN


# import numpy as np


# class Perceptron:
#     def __init__(self, learning_rate=0.01, n_iter=1000):
#         self.lr = learning_rate
#         self.n_iter = n_iter
#         self.weights = None
#         self.bias = None
#
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#
#         self.weights = np.zeros(n_features)
#         self.bias = 0.0
#
#         for _ in range(self.n_iter):
#             for idx, x_i in enumerate(X):
#                 linear_output = np.dot(x_i, self.weights) + self.bias
#
#                 y_predict = np.where(linear_output > 0, 1, -1)
#
#                 if y[idx] != y_predict:
#                     update = self.lr * y[idx]
#                     self.weights += update * x_i
#                     self.bias += update
#
#     def predict(self, X):
#         linear_output = np.dot(X, self.weights) + self.bias
#         y_predict = np.where(linear_output > 0, 1, -1)
#         return y_predict
#
#
# if __name__ == '__main__':
#     np.random.seed(42)
#     X1 = np.random.randn(50, 2) + np.array([2, 2])
#     X2 = np.random.randn(50, 2) + np.array([-2, -2])
#     X = np.vstack([X1, X2])
#     y = np.array([1]*50 + [-1]*50)
#
#     perceptron = Perceptron(learning_rate=0.05, n_iter=5000)
#     perceptron.fit(X, y)
#     predictions = perceptron.predict(X)
#     accuracy = np.mean(predictions == y)
#     print(f"Точность: {accuracy}")



# # AI
# # 1950 -
# # 1960-1970 -
# # 1980
# # 1990
# # 2000
#
#
# # Q-learning
# # Perceptron
# # MLP
#
# import numpy as np
#
#
# class Perceptron:
#     def __init__(self, learning_rate=0.01, n_iter=1000):
#         self.lr = learning_rate
#         self.n_iter = n_iter
#         self.weights = None
#         self.b = None
#
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#
#         self.weights = np.zeros(n_features)
#         self.b = 0.0
#
#         for _ in range(self.n_iter):
#             for idx, x_i in enumerate(X):
#                 linear_output = np.dot(x_i, self.weights) + self.b
#
#                 y_predict = np.where(linear_output > 0, 1, -1)
#
#                 if y[idx] != y_predict:
#                     update = self.lr*y[idx]
#                     self.weights += update * x_i
#                     self.b += update
#
#     def predict(self, X):
#         linear_output = np.dot(X, self.weights) + self.b
#         y_predict = np.where(linear_output > 0, 1, -1)
#         return y_predict
#
#
# if __name__ == '__main__':
#     np.random.seed(42)
#     X1 = np.random.randn(50, 2) + np.array([2, 2])
#     X2 = np.random.randn(50, 2) + np.array([-2, -2])
#
#     X = np.vstack([X1, X2])
#     y = np.array([1]*50 + [-1]*50)
#
#     perceptron = Perceptron(learning_rate=0.05, n_iter=5000)
#     perceptron.fit(X, y)
#
#     predictions = perceptron.predict(X)
#     acurracy = np.mean(predictions == y)
#     print(f" Точность на обучающей выборке равна: {acurracy}")







# 1950
# 1960-1970
# 1980
# 1990
# 2000-н.в


# # perceptron/MLP
# # Deep Learning
# # RL
#

# import nltk
#
# nltk.download('punkt')
#
# text = "Hello, World! I' m glad to see you. I like programming and Deep learning and artificial ... Glad to see me. World champion was World"
# tokens = nltk.word_tokenize(text.lower())
#
# freq = nltk.FreqDist(tokens)
#
# print(freq.most_common(3))


# import numpy as np

#
# class Perceptron:
#     def __init__(self, learning_rate=0.01, n=1000):
#         self.lr = learning_rate
#         self.n = n
#         self.weights = None
#         self.b = None
#
#     def fit(self, X, y):
#
#         n_sample, n_features = X.shape
#
#         self.weights = np.zeros(n_features)
#         self.b = 0.0
#
#         for _ in range(self.n):
#             for idx, x_i in enumerate(X):
#                 linear_result = np.dot(x_i, self.weights) + self.b
#
#                 prediction = np.where(linear_result > 0, 1, -1)
#
#                 if y[idx] != prediction:
#                     update = self.lr * y[idx]
#                     self.weights += update*y[idx]
#                     self.b += update
#
#     def predict(self, X):
#         linear_result = np.dot(X, self.weights) + self.b
#         prediction = np.where(linear_result > 0, 1, -1)
#         return prediction
#
#
# if __name__ == "__main__":
#     np.random.seed(42)
#
#     X1 = np.random.randn(50, 2) + np.array([2, 2])
#     X2 = np.random.randn(50, 2) + np.array([-2, -2])
#
#     X = np.vstack([X1, X2])
#     y = np.array([1]*50 + [-1]*50)
#
#     perceptron = Perceptron(learning_rate=0.01, n=100)
#     perceptron.fit(X, y)
#
#     predictions = perceptron.predict(X)
#     accuracy = np.mean(predictions == y)
#     print(f"Точность на обучающей модели с выборкой: {accuracy} ")

#

# import numpy as np
# from sklearn.cluster import KMeans
#
# # обучение с учителем
#
# X = np.random.rand(50, 2)
# kmeans = KMeans(n_clusters=3)
# labels = kmeans.fit_predict(X)
# print(f"Метки кластеров: {labels}")


# from sklearn.datasets import load_digits
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
#
#
# digits = load_digits()
#
# X, y = digits.data, digits.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#
# model = LogisticRegression(max_iter=3000000)
# model.fit(X_train, y_train)
#
# print(f"Точность: {model.score(X_test, y_test)}")


import numpy as np

Q = np.zeros((12, 2))
learning_rate = 0.1
discount = 0.9

iterations = 500


for _ in range(iterations):
    state = np.random.randint(0, 8)
    done = False
    while not done:
        if np.random.rand() < 0.1:
            action = np.random.randint(2)
        else:
            action = np.argmax(Q[state, :])

        if action == 0:
            new_state = max(0, state-1)
        else:
            new_state = max(9, state+1)

        reward = 1.0 if new_state == 5 else 0.0

        Q[state, action] += learning_rate * (reward + discount * np.max(Q[new_state, :]) - Q[state, action])

        state = new_state

        if state == 5:# условие выхода
            done = True


print(f"Обученная Q-таблица: {Q}")

