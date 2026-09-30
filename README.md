# Transfer Learning na Prática — ENG4502

<p align="center">
  <a href="https://rodrigohuber.github.io/aula-grad/SLIDES_AULA-transfer_learning.html">
    <img src="assets/abrir-slides.svg" alt="Abrir os slides da aula no navegador" width="760">
  </a>
</p>

<p align="center">
  <sub>Os slides abrem direto no navegador. O <b>simulador interativo</b> abre de dentro deles (slide de resultados → 🎮 Abrir Simulador Interativo), ou <a href="https://rodrigohuber.github.io/aula-grad/SLIDES_AULA-simulator.html">direto por aqui</a>.<br>
  Prefere offline? Baixe <code>SLIDES_AULA-transfer_learning.html</code> e <code>SLIDES_AULA-simulator.html</code> para a <b>mesma pasta</b> — o link entre eles continua funcionando.</sub>
</p>

Material dos alunos da disciplina **Introdução à Ciência de Dados** (PUC-Rio).

Nesta sequência de notebooks você vai implementar as principais técnicas de **Transfer Learning** usando uma rede convolucional pré-treinada (ResNet-18) no dataset CIFAR-10 — tudo rodando no **Google Colab**, sem instalar nada no seu computador.

---

## 📚 Sequência de Notebooks

| # | Notebook | Tópico principal | Colab |
|---|---|---|---|
| Intro | Demonstração guiada | Feature Extraction com truque de cache | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/transfer_learning_colab.ipynb) |
| Lab 1 | Prática 1 | Feature Extraction — você implementa | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/lab01_feature_extraction.ipynb) |
| Lab 2 | Prática 2 | Fine-Tuning Parcial + Discriminative LRs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/lab02_fine_tuning.ipynb) |
| Lab 3 | Dever de Casa | Treino do zero vs. Fine-Tuning autônomo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/lab03_homework.ipynb) |

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
- Acurácia esperada: ~85% (vs. ~35–50% treinando a mesma rede do zero)

---

### Lab 1 — `lab01_feature_extraction.ipynb`
**Você implementa a Feature Extraction passo a passo.**

Cobre o mesmo conceito do notebook de introdução, mas desta vez você constrói cada peça:
- **Exercício 1:** criar os `DataLoader`s com batch_size correto.
- **Exercício 2:** carregar a ResNet-18 pré-treinada, congelar o backbone e substituir a camada final.
- **Exercício 3:** definir a loss (`CrossEntropyLoss`) e o otimizador (SGD passando apenas `model.fc.parameters()`).

Cada bloco de código tem uma explicação detalhada do *por que* de cada decisão — por que 224×224, por que normalização ImageNet, o que `requires_grad=False` faz concretamente. A função `freeze_bn` mantém o backbone **100% congelado**, inclusive as estatísticas da BatchNorm.

- Tempo estimado: ~1–2 min na GPU T4; ~30–50 min na CPU
- Acurácia esperada: ~83–85% após 5 épocas

---

### Lab 2 — `lab02_fine_tuning.ipynb`
**Fine-Tuning Parcial: descongelar seletivamente + taxas discriminativas.**

Avança além do Lab 1. A ResNet-18 é primeiro usada como linha de base (Feature Extraction, já implementada), e depois você desbloqueia partes do backbone:
- **Exercício 1:** congelar tudo e descongelar apenas `layer4` (último bloco residual).
- **Exercício 2:** criar um otimizador SGD com dois grupos de parâmetros, cada um com taxa de aprendizado diferente — `lr=1e-4` para `layer4` (refinamento suave dos pesos pré-treinados) e `lr=1e-2` para `fc` (aprendizado mais rápido, pesos aleatórios).

Ao final, o notebook gera automaticamente gráficos comparativos de acurácia e loss, e matrizes de confusão lado a lado para identificar quais classes cada estratégia confunde mais.

- Tempo estimado: ~2–3 min na GPU T4 (dois modelos); ~1h30 na CPU
- Acurácia esperada: Fine-Tuning ~85–87% contra ~83–85% da Feature Extraction

---

### Lab 3 — `lab03_homework.ipynb`
**Dever de casa: treino do zero vs. Fine-Tuning autônomo.**

Dois experimentos com a mesma ResNet-18, os mesmos 5.000 exemplos e as mesmas 5 épocas — a única diferença é o ponto de partida da rede:

**Parte A — Treino do zero (Scratch):**
A ResNet-18 começa com pesos aleatórios (`weights=None`) e precisa aprender tudo a partir das 5.000 imagens. Resultado esperado: ~35–50% (a acurácia oscila bastante de uma época para outra).

**Parte B — Fine-Tuning Parcial autônomo:**
Reimplementação completa do Fine-Tuning da Prática 2, sem scaffolding. Você carrega o modelo pré-treinado, congela, descongela `layer4`, substitui `fc` e configura o otimizador discriminativo. Resultado esperado: ~85–87%.

**Questões de discussão:** de onde vem a diferença de ~45 pontos entre as duas partes, o tradeoff eficiência vs. acurácia, e uma questão sobre **Transferência Negativa** (o que acontece quando o domínio de origem e o de destino são muito diferentes).

- Tempo estimado: ~3 min na GPU T4; ~1h30 na CPU
- Conceitos novos: treinamento do zero, análise comparativa, transferência negativa

---

## 📊 Referência: Benchmarks Completos

Após completar todos os Labs, você pode abrir estes notebooks para ver modelos **totalmente treinados** com todos os resultados de fine-tuning já calculados e visualizados. Use-os para comparar seus resultados e entender como o desempenho escala com o tamanho do dataset.

### ✅ Benchmark 5k (atual) — `Versão completa (atual) - benchmark_professor_5k_10ep.ipynb`

Mesma configuração dos labs (backbone 100% congelado, inclusive a BatchNorm, na Feature Extraction e no Fine-Tuning Parcial; `lr=1e-2` na `fc`; sem Data Augmentation), agora com **10 épocas**. Quatro experimentos no subset de 5.000 imagens. Resultados salvos:

| Estratégia | Params treináveis | Acurácia |
|---|---|---|
| Scratch (do zero) | 11,2M | 45,0% |
| Feature Extraction | 5,1K | 84,8% |
| Fine-Tuning Parcial (`layer4`) | 8,4M | **88,0%** |
| Fine-Tuning Completo | 11,2M | 87,7% |

Repare: com poucos dados, abrir a rede inteira **não** supera o Fine-Tuning Parcial.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/Vers%C3%A3o%20completa%20%28atual%29%20-%20benchmark_professor_5k_10ep.ipynb)

### ✅ Benchmark 50k (atual) — `Versão completa (atual) - benchmark_professor_50k_10ep.ipynb`

Os mesmos quatro experimentos com o **dataset completo** (50.000 imagens de treino, 10.000 de validação): mostra quanto cada estratégia ganha com 10× mais dados. Resultados salvos:

| Estratégia | Params treináveis | Acurácia |
|---|---|---|
| Scratch (do zero) | 11,2M | 76,6% |
| Feature Extraction | 5,1K | 86,9% |
| Fine-Tuning Parcial (`layer4`) | 8,4M | 90,7% |
| Fine-Tuning Completo | 11,2M | **94,4%** |

Com dados suficientes, o Fine-Tuning Completo passa a valer a pena. Numa T4 do Colab o notebook inteiro leva cerca de 1h30 (estimativa).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/Vers%C3%A3o%20completa%20%28atual%29%20-%20benchmark_professor_50k_10ep.ipynb)

> Os resultados e tempos salvos nos dois notebooks foram gerados numa GPU local (NVIDIA RTX 5050), não numa T4: a acurácia é comparável, os tempos não.

> ⚠️ **Os dois benchmarks abaixo estão DEPRECATED.** Foram feitos com a configuração original da aula (BatchNorm do backbone não congelada, `lr=1e-3` na `fc` do Fine-Tuning Parcial e um experimento de Data Augmentation que saiu do curso), por isso seus números **não batem com os labs**. Ficam aqui apenas como registro.

### (DEPRECATED) Benchmark 5k — `Versão completa (DEPRECATED) - benchmark_professor_5k_10ep.ipynb`

Este notebook demonstra fine-tuning de ResNet-18 em um subconjunto reduzido de CIFAR-10 com 5.000 imagens ao longo de 10 épocas. A estratégia usa descongelamento parcial de camadas (`layer4` + `fc`) com taxas de aprendizado discriminativas (1e-4 para `layer4`, 1e-3 para `fc`) para balancear a preservação do conhecimento pré-treinado com adaptação específica da tarefa. Este dataset menor permite validar todo o pipeline rapidamente e entender como o desempenho do modelo escala com dados limitados. Nos resultados salvos, o Fine-Tuning Parcial chega a **79,3%** e o Fine-Tuning Completo a **88,7%**; o notebook inteiro (5 experimentos) leva cerca de **13 minutos** em GPU T4 do Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/Vers%C3%A3o%20completa%20%28DEPRECATED%29%20-%20benchmark_professor_5k_10ep.ipynb)

### (DEPRECATED) Benchmark 50k — `Versão completa (DEPRECATED) - benchmark_professor_50k_10ep.ipynb`

Este notebook aplica a mesma estratégia de fine-tuning ao dataset completo de CIFAR-10 (todas as 50.000 imagens de treino) ao longo de 10 épocas. Ao treinar com o dataset completo, você observa como dados adicionais melhoram a generalização e vê o modelo convergir para seu melhor desempenho. A mesma configuração de descongelamento de camadas e taxas de aprendizado discriminativas é usada, mas o volume aumentado de dados resulta em aprendizado de features mais robusto e maior acurácia final (nos resultados salvos, **86,5%** no Fine-Tuning Parcial e **94,6%** no Completo). Este treinamento maior leva cerca de **2 horas** em GPU T4. Ambos os benchmarks usam hiperparâmetros idênticos, permitindo comparação direta do impacto do tamanho do dataset no desempenho do modelo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rodrigohuber/aula-grad/blob/main/Vers%C3%A3o%20completa%20%28DEPRECATED%29%20-%20benchmark_professor_50k_10ep.ipynb)

---

## 🎯 O que você vai aprender

| Conceito | Onde aparece |
|---|---|
| O que é Transfer Learning e por que funciona | Todos |
| Feature Extraction (backbone congelado) | Intro + Lab 1 |
| Normalização ImageNet e redimensionamento | Lab 1 |
| Loop de treino PyTorch (forward, backward, step) | Lab 1 |
| Fine-Tuning Parcial (descongelamento seletivo) | Lab 2 |
| Congelamento completo do backbone (BatchNorm) | Lab 1 |
| Discriminative Learning Rates | Lab 2 |
| Matriz de confusão e análise de erros | Lab 2 |
| Treinamento do zero vs. Transfer Learning | Lab 3 |
| Transferência Negativa | Lab 3 |

---

## 📖 Leituras Recomendadas

### Leitura principal — imagens

**Yosinski, J., Clune, J., Bengio, Y. & Lipson, H. (2014).** *How transferable are features in deep neural networks?* Advances in Neural Information Processing Systems 27 (NeurIPS 2014).
🔗 [arxiv.org/abs/1411.1792](https://arxiv.org/abs/1411.1792) (acesso livre)

É a base conceitual dos Labs. O artigo mostra, com experimentos na ImageNet, que as primeiras camadas de uma rede convolucional aprendem features **gerais** (bordas, texturas), enquanto as últimas aprendem features **específicas** da tarefa original — e que a transferibilidade cai à medida que a tarefa de destino se afasta da tarefa de origem. Mostra também que inicializar com pesos transferidos e depois fazer fine-tuning melhora a generalização. É exatamente o porquê de congelarmos o backbone no **Lab 1** e descongelarmos apenas a `layer4` no **Lab 2**.

### Leitura complementar — séries temporais

**Mendes, R. H. M. M., Baião, F. A. & Souza, R. C. (2025).** *Exploring Transfer Learning Techniques for Solar Irradiation Forecast across Geographically Diverse Locations in Brazil Using Reanalysis Data.* Anais do LVII Simpósio Brasileiro de Pesquisa Operacional (SBPO 2025).
🔗 DOI: [10.59254/sbpo-2025-212275](https://doi.org/10.59254/sbpo-2025-212275)

Transfer Learning fora do mundo das imagens: um modelo de previsão de irradiação solar do dia seguinte (Random Forest), treinado com dados do Aeroporto do Galeão, é aplicado **sem fine-tuning** a outros 19 locais no Rio de Janeiro e no Brasil. O desempenho é muito bom, mas cai à medida que origem e destino divergem. O artigo compara cada local com um modelo treinado localmente (a *transfer loss*) e investiga métricas de distância para antecipar quando a transferência vai funcionar — a mesma pergunta por trás da **transferência negativa** discutida no **Lab 3**.
