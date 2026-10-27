
import matplotlib.pyplot as plt

# Posicoes dos neuronios
entradas = [(0, 1), (0, 0)]
ocultos = [(1, 1.5), (1, 1), (1, 0.5), (1, 0)]
saida = [(2, 0.75)]

fig, ax = plt.subplots(figsize=(10, 6))

# Desenha as conexoes entre as camadas
for x1, y1 in entradas:
    for x2, y2 in ocultos:
        ax.plot([x1, x2], [y1, y2],
                color="gray", alpha=0.5)

for x1, y1 in ocultos:
    for x2, y2 in saida:
        ax.plot([x1, x2], [y1, y2],
                color="gray", alpha=0.5)

# Desenha os neuronios de entrada
for i, (x, y) in enumerate(entradas):
    ax.scatter(x, y, s=1000, color="skyblue",
               edgecolors="navy", zorder=3)
    ax.text(x, y, f"x{i+1}", ha="center",
            va="center", fontsize=12)

# Desenha os neuronios ocultos
for i, (x, y) in enumerate(ocultos):
    ax.scatter(x, y, s=800, color="plum",
               edgecolors="purple", zorder=3)
    ax.text(x, y, f"h{i+1}", ha="center",
            va="center", fontsize=11)

# Desenha o neuronio de saida
x, y = saida[0]
ax.scatter(x, y, s=1000, color="lightgreen",
           edgecolors="darkgreen", zorder=3)
ax.text(x, y, "y", ha="center",
        va="center", fontsize=12)

# Titulos das camadas
ax.text(0, 1.8, "Entradas", ha="center", fontsize=13)
ax.text(1, 1.9, "Camada oculta", ha="center", fontsize=13)
ax.text(2, 1.2, "Saida", ha="center", fontsize=13)

ax.set_title("Arquitetura da rede neural XOR")
ax.set_xlim(-0.4, 2.4)
ax.set_ylim(-0.3, 2.2)
ax.axis("off")

plt.tight_layout()
plt.show()
