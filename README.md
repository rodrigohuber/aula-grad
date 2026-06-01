# Transfer Learning na Prática — ENG4502

Material dos alunos da disciplina **Introdução à Ciência de Dados** (PUC-Rio).

Nesta sequência de notebooks você vai implementar as principais técnicas de **Transfer Learning** usando uma rede convolucional pré-treinada (ResNet-18) no dataset CIFAR-10 — tudo rodando no **Google Colab**, sem instalar nada no seu computador.

---

## 📚 Sequência de Notebooks

| # | Notebook | Tópico principal | Colab |
|---|---|---|---|
| Intro | Demonstração guiada | Feature Extraction com truque de cache | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/transfer_learning_colab.ipynb) |
| Lab 1 | Prática 1 | Feature Extraction — você implementa | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/lab01_feature_extraction.ipynb) |
| Lab 2 | Prática 2 | Fine-Tuning Parcial + Discriminative LRs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/lab02_fine_tuning.ipynb) |
| Lab 3 | Dever de Casa | Data Augmentation + Fine-Tuning autônomo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/lab03_homework.ipynb) |

---

## ▶️ Como executar qualquer notebook

1. Clique no botão **Open in Colab** do notebook desejado.
2. Ative a GPU: **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)**.
3. Execute as células em ordem, de cima para baixo (**Shift+Enter**).

> Não é necessário instalar Python, PyTorch ou qualquer biblioteca — o Colab já tem tudo pronto.

---

## 🗺️ O que cada notebook cobre

### Notebook de Introdução — `transfer_learning_colab.ipynb`
**Demonstração guiada, sem exercícios para preencher.**

Apresenta o conceito de Feature Extraction de forma visual e rápida. O diferencial é o **truque de cache de features**: o backbone da ResNet-18 é executado uma única vez sobre todo o dataset, e suas saídas (vetores de 512 dimensões) são salvas em memória. A partir daí, treinamos apenas um classificador linear sobre esses vetores — isso transforma um treino de vários minutos em segundos, mesmo sem GPU.

- Estratégia: Feature Extraction com cache
- Parâmetros treinados: 5.130 (apenas a camada linear final)
- Tempo estimado: **< 1 minuto** na GPU T4
- Acurácia esperada: ~75% (vs. ~38% treinando do zero)

---

### Lab 1 — `lab01_feature_extraction.ipynb`
**Você implementa a Feature Extraction passo a passo.**

Cobre o mesmo conceito do notebook de introdução, mas desta vez você constrói cada peça:
- **Exercício 1:** criar os `DataLoader`s com batch_size correto.
- **Exercício 2:** carregar a ResNet-18 pré-treinada, congelar o backbone e substituir a camada final.
- **Exercício 3:** definir a loss (`CrossEntropyLoss`) e o otimizador (SGD passando apenas `model.fc.parameters()`).

Cada bloco de código tem uma explicação detalhada do *por que* de cada decisão — por que 224×224, por que normalização ImageNet, o que `requires_grad=False` faz concretamente.

- Tempo estimado: ~5 min na GPU T4; ~30–50 min na CPU
- Acurácia esperada: ~70–75% após 5 épocas

---

### Lab 2 — `lab02_fine_tuning.ipynb`
**Fine-Tuning Parcial: descongelar seletivamente + taxas discriminativas.**

Avança além do Lab 1. A ResNet-18 é primeiro usada como linha de base (Feature Extraction, já implementada), e depois você desbloqueia partes do backbone:
- **Exercício 1:** congelar tudo e descongelar apenas `layer4` (último bloco residual).
- **Exercício 2:** criar um otimizador SGD com dois grupos de parâmetros, cada um com taxa de aprendizado diferente — `lr=1e-4` para `layer4` (refinamento suave dos pesos pré-treinados) e `lr=1e-3` para `fc` (aprendizado mais rápido, pesos aleatórios).

Ao final, o notebook gera automaticamente gráficos comparativos de acurácia e loss, e matrizes de confusão lado a lado para identificar quais classes cada estratégia confunde mais.

- Tempo estimado: ~10 min na GPU T4 (dois modelos); ~1h30 na CPU
- Acurácia esperada: Fine-Tuning > Feature Extraction nas épocas finais

---

### Lab 3 — `lab03_homework.ipynb`
**Dever de casa: Data Augmentation e Fine-Tuning autônomo.**

Dois experimentos independentes para consolidar o aprendizado:

**Parte A — Scratch com Data Augmentation:**  
Você define o pipeline de transformações com `RandomHorizontalFlip` e `RandomRotation(15)` aplicados apenas ao conjunto de treino. O modelo treinado do zero (sem pesos pré-treinados) usa essas transformações para tentar superar o benchmark de 37.8% visto em aula.

**Parte B — Fine-Tuning Parcial autônomo:**  
Reimplementação completa do Fine-Tuning da Prática 2, sem scaffolding. Você carrega o modelo, congela, descongela `layer4`, substitui `fc` e configura o otimizador discriminativo — desta vez sem dicas de código.

**Questões de discussão:** comparação dos resultados das duas partes, análise do tradeoff eficiência vs. acurácia, e uma questão sobre **Transferência Negativa** (o que acontece quando o domínio de origem e o de destino são muito diferentes).

- Tempo estimado: ~15 min na GPU T4; ~1h40 na CPU
- Conceitos novos: Data Augmentation, treinamento do zero, análise comparativa

---

## 🎯 O que você vai aprender

| Conceito | Onde aparece |
|---|---|
| O que é Transfer Learning e por que funciona | Todos |
| Feature Extraction (backbone congelado) | Intro + Lab 1 |
| Normalização ImageNet e redimensionamento | Lab 1 |
| Loop de treino PyTorch (forward, backward, step) | Lab 1 |
| Fine-Tuning Parcial (descongelamento seletivo) | Lab 2 |
| Discriminative Learning Rates | Lab 2 |
| Matriz de confusão e análise de erros | Lab 2 |
| Data Augmentation como regularização | Lab 3 |
| Treinamento do zero vs. Transfer Learning | Lab 3 |
| Transferência Negativa | Lab 3 |
