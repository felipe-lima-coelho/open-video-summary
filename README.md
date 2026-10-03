# Open Video Summary

Projeto de pesquisa em sumarização de vídeos e multivídeos. A biblioteca reúne
segmentação por transcrição/tópicos e critérios de seleção como introdução,
subjetividade, redundância, qualidade visual e cronologia.

## Teste local no Windows

Use **Python 3.11 x64**. O Python 3.13 não é compatível com o TensorFlow 2.17
usado neste projeto. O ambiente abaixo usa CPU e funciona com GPU integrada AMD.
Não é necessário ativar o ambiente virtual nem instalar Poetry para esse teste.

Na raiz do clone, execute no PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\.venv\Scripts\python.exe -m open_video_summary summarize
```

O primeiro comando cria `.venv`, instala as versões de `requirements-windows.lock`,
instala o projeto em modo editável e prepara os recursos originais do notebook.
Ele pode ser repetido: os vídeos e o modelo completos são reutilizados.
`-ExecutionPolicy Bypass` vale somente para esse processo PowerShell.

A primeira preparação baixa aproximadamente 662 MB do classificador original de
subjetividade, além das dependências Python. PyTorch CPU tem cerca de 207 MB e
TensorFlow para Windows cerca de 382 MB de download. Reserve alguns GB em disco.
O ZIP do modelo tem seu SHA256 verificado antes da extração. Os três vídeos de
exemplo já estão no clone em `data/raw/bebe_real.zip` (35 MB).

O comando `summarize` executa o **HSMVideoSumm original** nos três vídeos e nos
13 segmentos de `data/processed/bebe_real.json`. Usa o classificador treinado e
os critérios científicos existentes. As transcrições e os tópicos desse exemplo
já foram calculados; esse teste não precisa de servidor Ollama, chave de API ou
download do Whisper. O processamento e a codificação usam CPU, com dois threads
por padrão. Durante testes, prefira fechar aplicações que consomem muita RAM.

Os resultados são:

- `outputs/bebe_real_summary.mp4`: resumo com vídeo e áudio;
- `outputs/bebe_real_summary.json`: segmentos selecionados e seus tempos;
- `outputs/bebe_real_summary_handler.json`: decisões dos critérios de seleção;
- `app.log`: registro da execução.

A duração e os segmentos escolhidos dependem da seleção do algoritmo. A execução
do exemplo verifica o funcionamento do pipeline; a qualidade científica exige
avaliação com as métricas e os dados da pesquisa.

Para conferir o ambiente ou repetir apenas a preparação:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary doctor
.\.venv\Scripts\python.exe -m open_video_summary prepare-demo
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Caminhos e reprodução em outro clone

Todos os caminhos relativos da API, da CLI e dos JSONs são interpretados a partir
da raiz do repositório, encontrada pelo código do pacote. Por exemplo,
`data/raw/bebe_real/jornal_nacional.mp4` não depende do usuário, da letra do disco
ou do diretório de trabalho. O carregador resolve tanto `path` do vídeo quanto
`video_path` dos segmentos. Os exportadores gravam caminhos internos ao projeto
relativos à raiz e usam UTF-8. Caminhos absolutos externos também são aceitos.

Não versione `.venv`, modelos baixados, caches e resultados. Copie os vídeos
necessários para `data/raw/<conjunto>/` ao compartilhar um experimento. O clone
inclui metadados de cinco conjuntos, mas somente `bebe_real` inclui os MP4s.
Outros vídeos estão no [diretório original de conjuntos de dados](https://drive.google.com/drive/folders/1y19ih3j36UqXlWFcgyxNXsNWluE3lky6?usp=drive_link).

Para testar outro conjunto já segmentado:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary summarize --dataset data/processed/meu_conjunto.json --output outputs/meu_resumo.mp4
```

Para executar a partir de outro diretório, invoque o Python do clone pelo caminho
até ele. Os argumentos continuam relativos à raiz do projeto. A instalação
editável precisa ser refeita se você mover a pasta ou criar outro clone.

## Novos vídeos: Whisper e Ollama

A criação de segmentos a partir de MP4s brutos é uma etapa adicional. Ela exige
o executável `ffmpeg` no PATH, um servidor [Ollama](https://ollama.com/download)
em execução e um modelo disponível nesse servidor. O pacote Python `ollama`
instalado pelo projeto é o cliente da API; ele não instala o servidor.
O pacote Python `ffmpeg` também não substitui o executável.

Confira o FFmpeg com `ffmpeg -version`. Caso precise instalá-lo no Windows,
use `winget install -e --id Gyan.FFmpeg` e abra um novo PowerShell.

Depois de instalar o servidor Ollama, por exemplo:

```powershell
ollama pull gemma2
.\.venv\Scripts\python.exe -m open_video_summary segment --input data/raw/meu_conjunto --output outputs/meu_conjunto_segments.json --whisper-model base --llm-model gemma2
.\.venv\Scripts\python.exe -m open_video_summary summarize --dataset outputs/meu_conjunto_segments.json --output outputs/meu_resumo.mp4
```

`gemma2` é o modelo usado pelo adaptador original e requer um download adicional
de vários GB; ele não é baixado pelo setup. A CLI verifica o servidor/modelo
antes de carregar o Whisper. O Whisper `base` é uma opção inicial mais leve
para CPU e baixa seus pesos na primeira transcrição. `tiny` também está disponível.
O notebook original usava `medium`; escolha `--whisper-model medium` para esse
modelo maior. A escolha do modelo pode mudar as transcrições e os resultados.
O cache do Whisper fica em `.cache/whisper` na raiz do projeto.

## Notebooks e versões

Abra os notebooks com o kernel `.venv/Scripts/python.exe` em seu editor Jupyter.
`notebooks/hsmvideosumm.ipynb` prepara os mesmos recursos e executa o resumo.
`notebooks/video-segmenter.ipynb` usa `base` e somente o conjunto de vídeos
incluído no clone; requer Whisper/Ollama e grava novos resultados em `outputs`.
`notebooks/llm-evaluations.ipynb` requer seus próprios conjuntos de avaliação,
modelos Ollama e, quando aplicável, acesso ao Kaggle. Esses recursos adicionais
não são necessários ao teste local de HSMVideoSumm.

`requirements-windows.lock` foi derivado de `poetry.lock` para Windows x64 e
Python 3.11. Preserva as versões do projeto, com PyTorch `2.6.0+cpu`, acrescenta
`tensorflow-intel==2.17.1` requerido pelo wheel Windows de TensorFlow e usa
`tensorflow-io-gcs-filesystem==0.31.0`, que possui wheel para essa plataforma.
O lock original contém `0.37.1`, sem wheel Windows. Para outros ambientes,
o fluxo original continua sendo `poetry install` com Python compatível.
