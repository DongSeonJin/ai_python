# 1. 데이터
import numpy as np

x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(30))
model.add(Dense(1))

# 3. 컴파일훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs= 1000, batch_size=3)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)

result = model.predict([6])
print('result :', result)