import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

x = np.linspace(0,1, 5000)


plt.figure(figsize=(7,7))


plt.plot(x, beta.pdf(x, 10, 10, 0, 1), 'r-')


plt.title('Beta Distribution', fontsize='15')
plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
plt.ylabel('Probability', fontsize='15')
plt.show()

x = np.linspace(0,1, 5000)


plt.figure(figsize=(7,7))


plt.plot(x, beta.pdf(x, 3, 8, 0, 1), 'r-')
plt.plot(x, beta.pdf(x, 8, 3, 0, 1), 'g-')


plt.title('Beta Distribution', fontsize='15')
plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
plt.ylabel('Probability', fontsize='15')
plt.show()


x = np.linspace(0,1, 5000)


plt.figure(figsize=(7,7))


plt.plot(x, beta.pdf(x, 2, 8, 0, 1), 'r-')
plt.plot(x, beta.pdf(x, 8, 2, 0, 1), 'b-')
plt.plot(x, beta.pdf(x, 5, 5, 0, 1), 'g-')
plt.plot(x, beta.pdf(x, 7, 3, 0, 1), 'y-')

plt.title('Beta Distribution', fontsize='15')
plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
plt.ylabel('Probability', fontsize='15')
plt.show()