# Classificação de doenças do arroz com redes neurais profundas

Trabalho da disciplina SIN 393 – Introdução à Visão Computacional (UFV, *campus* Rio Paranaíba, 2026).

**Vídeo de apresentação:** https://youtu.be/NAzgm0vu1AE

## Descrição do projeto

O objetivo do trabalho é classificar doenças do arroz a partir de fotos tiradas no campo e verificar o que pesa mais
no desempenho: a família da arquitetura ou o tamanho do modelo. Para isso, comparamos seis redes neurais
convolucionais (CNNs) pré-treinadas no ImageNet, de três famílias e em dois tamanhos cada:

| Família | Menor | Maior |
|---|---|---|
| MobileNetV3 | MobileNetV3-Small | MobileNetV3-Large |
| ResNet | ResNet-18 | ResNet-50 |
| ConvNeXt | ConvNeXt-Tiny | ConvNeXt-Large |

Como comparação, também treinamos dois modelos baseados em atenção de tamanho parecido com a ConvNeXt-Tiny:
Swin-Tiny e MaxViT-Tiny.

### Dataset

Usamos as 10.407 imagens rotuladas de treino da competição
[Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification) do Kaggle, que
vêm do conjunto *Paddy Doctor*. As imagens têm 480×640 pixels, foram tiradas com celular em lavouras e têm fundo
complexo e iluminação variável. São 10 classes: nove doenças ou pragas e a classe `normal` (plantas saudáveis).
As imagens estão na pasta `datasets/`, com uma subpasta por classe.

| Classe | Imagens | Classe | Imagens |
|---|---:|---|---:|
| `normal` | 1.764 | `brown_spot` | 965 |
| `blast` | 1.738 | `downy_mildew` | 620 |
| `hispa` | 1.594 | `bacterial_leaf_blight` | 479 |
| `dead_heart` | 1.442 | `bacterial_leaf_streak` | 380 |
| `tungro` | 1.088 | `bacterial_panicle_blight` | 337 |

### Metodologia

Todos os modelos foram treinados com o mesmo protocolo:

- Divisão estratificada em treino (70%), validação (15%) e teste (15%), com semente 42. A divisão fica salva em
  `results/split.csv`, e o teste só é usado na avaliação final.
- Entrada de 224×224 pixels, normalizada com a média e o desvio padrão do ImageNet.
- Aumento de dados no treino: recorte aleatório, espelhamentos, rotação de até 30° e variação de brilho,
  contraste, saturação e matiz.
- Entropia cruzada com pesos por classe, inversamente proporcionais ao número de imagens de treino, por causa do
  desbalanceamento.
- Ajuste fino em duas etapas: 3 épocas treinando só a camada de saída (taxa de aprendizado 10⁻³) e depois a rede
  inteira (10⁻⁴).
- AdamW, lotes de 32, taxa reduzida pela metade após 2 épocas sem melhora, até 50 épocas, parada antecipada após
  10 épocas sem melhora e precisão mista. O teste usa os pesos da época com menor perda de validação.
- Avaliação com matriz de confusão, acurácia, precisão, revocação e F1-score (médias macro e weighted), além de
  exemplos de acertos e erros.

### Resultados

Desempenho no conjunto de teste (1.562 imagens), em %:

| Modelo | Acurácia | Precisão (macro) | Revocação (macro) | F1 (macro) | F1 (weighted) |
|---|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 94,30 | 93,81 | 93,71 | 93,72 | 94,30 |
| MobileNetV3-Large | 96,35 | 96,12 | 96,49 | 96,29 | 96,35 |
| ResNet-18 | 96,09 | 95,76 | 96,48 | 96,09 | 96,10 |
| ResNet-50 | 95,26 | 94,25 | 95,98 | 95,02 | 95,29 |
| ConvNeXt-Tiny | 97,18 | 96,87 | 97,17 | 97,01 | 97,19 |
| ConvNeXt-Large | 97,31 | 97,15 | 97,51 | 97,32 | 97,31 |
| Swin-Tiny | 97,44 | 97,30 | 97,61 | 97,45 | 97,44 |
| MaxViT-Tiny | 97,12 | 97,02 | 97,10 | 97,04 | 97,11 |

As ConvNeXt e os modelos de atenção ficaram praticamente empatados. A ConvNeXt-Large não foi significativamente
melhor que a ConvNeXt-Tiny, que é sete vezes menor, e o aumento de tamanho só trouxe ganho claro na MobileNetV3.
A MobileNetV3-Large, com 4,2 milhões de parâmetros, é a opção mais leve com bom desempenho. Os erros se concentram
nas classes `hispa` e `downy_mildew`.

As métricas de todos os modelos estão em `results/metrics_summary.csv`. Em `results/<modelo>/` ficam o histórico de
treino, o relatório por classe, a matriz de confusão, as predições no teste e os exemplos de acertos e erros.

<p align="center">
  <img src="results/convnext_large/confusion_matrix.png" width="600" alt="Matriz de confusão da ConvNeXt-Large">
</p>

## Estrutura do repositório

```
├── main.ipynb          # código: dados, modelos, treino, avaliação e gráficos
├── requirements.txt    # dependências
├── datasets/           # imagens, uma subpasta por classe
└── results/
    ├── split.csv            # divisão treino/validação/teste
    ├── metrics_summary.csv  # métricas de teste de cada modelo
    └── <modelo>/            # resultados de cada modelo
```

## Como executar

Todo o código está no notebook `main.ipynb`. Ele foi feito para o Google Colab, mas também roda localmente.

### Requisitos

- Python 3.10 ou superior.
- GPU NVIDIA recomendada. Na CPU funciona, mas o treino é muito lento. O MaxViT-Tiny com lote de 32 precisa de
  mais de 6 GB de memória de GPU.
- Dependências (em `requirements.txt`): PyTorch, torchvision, timm, scikit-learn, pandas, NumPy, Matplotlib,
  seaborn e Pillow.

Os resultados acima foram obtidos com PyTorch 2.14.1 e torchvision 0.29, as CNNs em uma NVIDIA A100 (Google Colab)
e os modelos de atenção em uma RTX 4050 Laptop.

### Execução local

1. Clone o repositório (o dataset vem junto):

   ```bash
   git clone https://github.com/SrEmanuel/trabalho-sin393-visao.git
   cd trabalho-sin393-visao
   ```

2. Crie um ambiente e instale as dependências:

   ```bash
   conda create -n sin393 python=3.12 -y
   conda activate sin393
   pip install -r requirements.txt
   pip install notebook
   ```

   O `requirements.txt` instala o PyTorch para CUDA 13.0. Com um driver NVIDIA mais antigo, troque `cu130` por
   `cu128` ou `cu126` na linha `--extra-index-url`. Para usar só a CPU, troque por `cpu`.

3. Abra o notebook com `jupyter notebook main.ipynb`.

4. Pule as células das seções 1 (instalação) e 2 (Google Drive) e, na seção 14, apague a linha
   `shutil.copytree(...)`, que copia os resultados para o Drive.

5. Execute as demais células. A seção 14 treina as seis CNNs em sequência; para treinar também os modelos de
   atenção, acrescente `"swin_t"` e `"maxvit_t"` à lista. Cada modelo salva seus resultados em `results/<modelo>/`.

Os hiperparâmetros ficam na seção 4 do notebook. No Windows, se o carregamento dos lotes travar, use
`NUM_WORKERS = 0`.

### Google Colab

1. Compacte a pasta `datasets/` em `datasets.zip` (o zip deve conter a própria pasta `datasets/`).
2. Envie o `datasets.zip` para a raiz do seu Google Drive.
3. No [Google Colab](https://colab.research.google.com), abra o `main.ipynb` em **Arquivo → Fazer upload de notebook**.
4. Em **Ambiente de execução → Alterar o tipo de ambiente de execução**, escolha uma GPU.
5. Execute tudo e autorize o acesso ao Google Drive quando for pedido.

Ao fim de cada modelo, a pasta `results/` é copiada para o Drive. A última célula gera um `results.zip` com todos os
resultados.

## Autores

- Emanuel Teixeira Martins - 8743
- Pedro Augusto Simões da Cruz - 8116
- Raul Suzuki Borges - 9352
