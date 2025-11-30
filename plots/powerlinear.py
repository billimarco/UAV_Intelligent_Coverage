import matplotlib.pyplot as plt
import numpy as np

# Parametri
k = 1
x = np.linspace(0, 1, 200)
y = x**k

# Creazione del grafico con colore personalizzato
plt.plot(x, y, color="orange", label=r"$\alpha(k) = x^1$", linewidth=2)
plt.title(r"Curva di $\alpha(k) = x^k$ con $k = 1$")
plt.xlabel("x")
plt.ylabel(r"$\alpha(k)$")
plt.grid(True)
plt.legend()
plt.show()