# 🧠 Aula prática — Transfer Learning (ENG4502)

Material para os alunos da disciplina **Introdução à Ciência de Dados** (PUC-Rio).

Neste exercício você vai usar uma rede neural **já treinada** (ResNet-18) para
classificar imagens do CIFAR-10 — sem precisar treinar nada do zero e sem instalar
nada no seu computador. Tudo roda no **Google Colab**, de graça.

## ▶️ Como executar

Clique no botão abaixo para abrir o notebook direto no Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/transfer_learning_colab.ipynb)

Depois, dentro do Colab:

1. Ative a GPU: **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)**.
2. Rode as células em ordem (Shift+Enter), de cima para baixo.
3. O notebook inteiro leva **menos de 1 minuto** na GPU.

## 📋 O que você vai aprender

- O que é **Transfer Learning** e por que ele funciona.
- A estratégia de **Feature Extraction** (congelar a rede pré-treinada).
- Como reaproveitar características de uma rede treinada no ImageNet para uma nova tarefa.

> Não é necessário instalar Python nem bibliotecas — o Colab já tem tudo pronto.
