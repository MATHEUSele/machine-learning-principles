
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. ENTRADAS E RESPOSTAS CORRETAS
# ==========================================

x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

print("Entradas:")
print(x)

print("\nSaidas esperadas:")
print(y)

print("\nFormato das entradas:", x.shape)
print("Formato das saidas:", y.shape)


# ==========================================
# 2. PRIMEIRO NEURONIO ARTIFICIAL
# ==========================================

x1 = 1
x2 = 0

w1 = 0.5
w2 = 0.2

bias = 0.1

z = (x1 * w1) + (x2 * w2) + bias

print("\nResultado do neuronio:", z)


# ==========================================
# 3. FUNCAO DE ATIVACAO SIGMOID
# ==========================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


ativacao = sigmoid(z)

print("Resultado antes da ativacao:", z)
print("Resultado depois da ativacao:", ativacao)


# ==========================================
# 4. INICIALIZACAO DOS PESOS DA REDE
# ==========================================

np.random.seed(42)

# Pesos das entradas para a camada oculta
W1 = np.random.randn(2, 4) * 0.5

# Bias dos quatro neuronios ocultos
b1 = np.zeros((1, 4))

# Pesos da camada oculta para a saida
W2 = np.random.randn(4, 1) * 0.5

# Bias do neuronio de saida
b2 = np.zeros((1, 1))

print("\nFormatos dos parametros:")
print("W1:", W1.shape)
print("b1:", b1.shape)
print("W2:", W2.shape)
print("b2:", b2.shape)


# ==========================================
# 5. FORWARD PASS - PREVISOES DA REDE
# ==========================================

# Camada oculta
z1 = x @ W1 + b1
a1 = sigmoid(z1)

# Camada de saida
z2 = a1 @ W2 + b2
y_pred = sigmoid(z2)

print("\nPrevisoes iniciais da rede:")
print(y_pred)

print("\nRespostas corretas:")
print(y)


# ==========================================
# 6. GRAFICO DAS PREVISOES INICIAIS
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    y.flatten(),
    "o-",
    label="Resposta correta"
)

plt.plot(
    y_pred.flatten(),
    "x--",
    label="Previsao da rede"
)

plt.xticks(
    range(4),
    ["0 XOR 0", "0 XOR 1", "1 XOR 0", "1 XOR 1"]
)

plt.ylim(-0.1, 1.1)
plt.ylabel("Saida")
plt.xlabel("Exemplos XOR")
plt.title("Previsoes antes do treinamento")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
