# Scouting Intelligence — Global Football Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)

Um motor de recomendação focado em *Football Analytics*, desenvolvido para identificar substitutos estatísticos ideais ("gémeos estatísticos") de jogadores da elite europeia (Top 5 Ligas) atuando em ligas periféricas (Championship, Primeira Liga, Eredivisie, Pro League).

O projeto une análise de dados, álgebra linear (similaridade de cossenos) e desenvolvimento front-end para gerar um dashboard interativo de scouting totalmente *standalone*.

## Objetivo do Projeto
Resolver um problema real de scouting de futebol moderno: **"Como encontrar um jogador acessível em uma liga secundária que entregue o mesmo perfil de produção estatística de uma estrela mundial?"**

O sistema analisa dezenas de métricas de desempenho por 90 minutos (passes progressivos, desarmes, ações de criação, etc.) e cruza o perfil numérico dos atletas, ajustando os valores pela dificuldade competitiva de cada liga.

##  Arquitetura e Funcionalidades

### 1. Motor Analítico (Python & Scikit-Learn)
* **Extração e Tratamento**: Conexão nativa via SQLite (`scouting_engine.db`) e higienização automatizada usando Pandas.
* **Mapeamento de Clubes e Ligas**: Dicionário inteligente para tratar nomenclaturas não padronizadas e alocar automaticamente os times em suas respectivas ligas (Premier League, La Liga, Serie A, Bundesliga, Ligue 1).
* **Coeficientes de Dificuldade (League Weights)**: As métricas brutas recebem pesos específicos baseados no nível da liga (ex: *Premier League = 1.00*, *Primeira Liga = 0.78*, *Eredivisie = 0.72*). Isso garante que produzir números em ligas periféricas seja ponderado de forma justa antes da comparação com a elite.
* **Similaridade Vetorial**: Utilização da matriz de **Cosine Similarity** (`sklearn.metrics.pairwise`) após normalização Min-Max, calculando a distância vetorial exata entre os perfis dos jogadores.
* **Calibração Orgânica**: O cálculo final da similaridade utiliza rankeamento posicional e o desvio padrão da distribuição de cossenos, garantindo pontuações realistas (variando tipicamente entre 60% e 90%), simulando a imprevisibilidade do scouting humano.

### 2. Dashboard Interativo (HTML / JS / CSS)
* **Geração Dinâmica**: O script Python consolida o modelo vetorial e injeta os resultados diretamente em um arquivo `HTML` leve e autossuficiente.
* **Design Moderno**: Interface elegante baseada em plataformas de ponta (Dark Mode em "Carbono" com detalhes em "Verde Neon").
* **Filtros em Cascata Inteligentes**: A seleção no painel flui de forma dependente (Liga → Equipe → Jogador Alvo), sem travamentos.
* **Drill-down Modal**: Ao clicar no jogador sugerido, um *modal* exibe as 8 estatísticas cruciais que motivaram a recomendação matemática.

##  Como Executar o Projeto Localmente

1. Clone este repositório:
```bash
git clone [https://github.com/vitorblm31/global-scouting-engine.git](https://github.com/vitorblm31/global-scouting-engine.git)
