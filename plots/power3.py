import matplotlib.pyplot as plt
import numpy as np

# Parametri
k = 3
x = np.linspace(0, 1, 200)
y = x**k

# Creazione del grafico con colore personalizzato
plt.plot(x, y, color="purple", label=r"$\alpha(k) = x^3$", linewidth=2)
plt.title(r"Curva di $\alpha(k) = x^k$ con $k = 3$")
plt.xlabel("x")
plt.ylabel(r"$\alpha(k)$")
plt.grid(True)
plt.legend()
plt.show()