from environment import Environment
from brain import Brain
from DQN import Dqn
import numpy as np
import matplotlib.pyplot as plt
import os

printOutIntervals = [50, 100, 150, 200, 250, 300]
printOutCount = 0
memSize = 60000
batchSize = 32
# learningRate = 0.0001
learningRate = 0.0001

gamma = 0.9
nLastStates = 4

epsilon = 1
epsilonDecayRate = 0.0004
minEpsilon = 0.05

filePath = 'snake_model.keras'
env = Environment(0)
brain = Brain()

# check if keras file exists first
if os.path.exists(filePath):
    brain.loadModel(filePath)
else:
    brain = Brain((env.nRows, env.nColumns, nLastStates), learningRate)

model = brain.model
dqn = Dqn(memSize, gamma)
iterationCounter = 0


def resetStates():
    currentState = np.zeros((1, env.nRows, env.nColumns, nLastStates))

    for i in range(nLastStates):
        currentState[:, :, :, i] = env.screenMap

    return currentState, currentState


def makeGraph(totalCollected):
    global printOutCount
    scores.append(totalCollected / 100)

    plt.plot(scores)
    plt.xlabel(f'Epoch / {epoch}')
    plt.ylabel('Average Score')
    plt.savefig(f'stats{printOutCount}.png')
    plt.close()
    printOutCount += 1
    return 0


epoch = 0
scores = list()
maxNCollected = 0
nCollected = 0.
totNCollected = 0
saveCounter = 0
while True:
    env.reset()
    currentState, nextState = resetStates()

    epoch += 1
    gameOver = False

    while not gameOver:
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 4)
        else:
            qvalues = model.predict(currentState)[0]
            action = np.argmax(qvalues)

        state, reward, gameOver = env.step(action)


        state = np.reshape(state, (1, env.nRows, env.nColumns, 1))
        nextState = np.append(nextState, state, axis=3)
        nextState = np.delete(nextState, 0, axis=3)

        dqn.remember([currentState, action, reward, nextState], gameOver)
        inputs, targets = dqn.get_batch(model, batchSize)
        model.train_on_batch(inputs, targets)

        if env.collected:
            nCollected += 1

        currentState = nextState

        if nCollected > maxNCollected and nCollected > 2:
            maxNCollected = nCollected
        #     model.save(filePath)

        totNCollected += nCollected
        nCollected = 0

        # if epoch % 100 == 0 and epoch != 0:
        #     scores.append(totNCollected / 100)
        #     totNCollected = 0
        #     plt.plot(scores)
        #     plt.xlabel('Epoch / 100')
        #     plt.ylabel('Average Score')
        #     plt.savefig('stats.png')
        #     plt.close()

        if epoch in printOutIntervals:
            totNCollected = makeGraph(totNCollected)

        if saveCounter == 200:
            model.save(f'snake_model{iterationCounter}.keras')
            iterationCounter += 1
            saveCounter = 0
            print('Epoch: ' + str(epoch) + ' Current Best: ' +
                  str(maxNCollected) + ' Epsilon: {:.5f}'.format(epsilon) + ' also saved..')
        saveCounter += 1

        if epsilon > minEpsilon:
            epsilon -= epsilonDecayRate
