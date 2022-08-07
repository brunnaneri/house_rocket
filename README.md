# <p align="center"> House Rocket Company</p>

<p align="center"> <img src="https://github.com/brunnaneri/house_rocket/blob/main/arte.png?raw=true" width=70% height=70% title="House-Rocket-Company" alt="project_cover_image"/> </p>

A House Rocket Company é uma empresa do ramo imobiliário que atua no meio digital adquirindo e revendendo imóveis por meio de sua plataforma online.

bservação:
Este projeto foi inspirado no desafio "House Sales Prediction" publicado no Kaggle (https://www.kaggle.com/harlfoxem/housesalesprediction). Por isso, trata-se de um problema fictício, no entanto solucionado com passos e análises de um projeto real.

## 1.0 PROBLEMA DE NEGÓCIO

### 1.1 Descrição do Problema

O CEO da House Rocket Company está com dificuldades em definir quais imóveis a empresa deve comprar e revender de forma que o lucro seja maximizado, visto que os métodos e ferramentas utilizados atualmente pelo time de negócios estão com desempenho abaixo do esperado. 

### 1.2 Objetivo

Por isso, esse projeto visa otimizar a tomada de decisão do CEO através de recomendações desenvolvidas através da análise de dados do portfólio de imóveis da empresa. Tal análise visa identificar os imóveis que estão abaixo do preço mediano de venda para que sejam adquiridos pela House Rocket e posteriormente revendido a um preço ideal definido com base na análise dos dados, vislumbrando um incremento de 15 a 25% no lucro da empresa. 

Como resultado será entregue respostas às seguintes perguntas:
- Quais imóveis a House Rocket deve adquirir?
- Uma vez o imóvel adquirido, quando e por qual preço deve-se revendê-lo?

Além disso, deve ser disponibilizado ao time de negócios um dashboard contendo análises e informações relevantes do portfólio de imóveis da empresa, a fim de viabilizar uma melhor análise dos dados e consequentemente melhorar a tomada de decisão na prospecção de novos imóveis por parte do time.

## 2.0 PREMISSAS DO NEGÓCIO

- Os produtos das soluções desenvolvidas devem ser acessíveis via internet.
- As análises levarão em consideração a localização e a influência da sazonalidade na precifição dos imóveis.

## 3.0 PLANEJAMENTO DA SOLUÇÃO

### 3.1 Produto Final

O produto entregue ao CEO será um link com um dashboard composto por mapas, tabelas com análises estatísticas e gráficos com filtros interativos, além de duas tabelas de recomendações de compra e venda dos imóveis com o lucro previsto para cada revenda.

### 3.2 Processo

Os passos realizados no projeto foram:
  - Entendimento do modelo e problema de negócio;
  - Coleta dos dados;
  - Limpezas e Transformações necessárias;
  - Análise Exploratória dos Dados;
  - Criação e validação de hipóteses;
  - Desenvolver dashboard interativo;
  - Publicar o dashboard em produção.
  
### 3.3 Entrada

#### 3.3.1 Fonte de Dados:
- Os dados foram coletados na plataforma Kaggle, através do link: https://www.kaggle.com/harlfoxem/housesalesprediction

#### 3.3.2 Ferramentas:
 - Python, Pandas, Numpy e Seaborn.
 - IDE Pycharm e Jupyter Notebook.
 - Mapas interativos com Plotly e Folium.
 - Heroku Cloud.
 - Streamlit Python framework web.
  

## 4.0 TOP 3 INSIGHTS

Insights são informações novas, ou que contrapõe crenças até então estabelecidas do time de negócios, que devem ser acionáveis e assim direcionar resultados futuros.

Consideração: Em todas propostas acionáveis, descrita como 'insight', busca-se oportunidades de prospecção de imóveis que se encaixem no proposto a fim de se obter uma maior margem de lucro. Além disso, as considerações na busca desses imóveis devem se aliar a critérios relevantes como o seu estado de conservação e sua zona.


**H1: Imóveis que possuem vista para água são, em média, 3.13 vezes mais caros.**
- Insight: Prospectar imóveis que tenham vista para água que estejam com preço de venda até 1.5 vez maior que o preço médio dos imóveis que não têm vista para a água, considerando que estes devem estar nas mesmas ou melhores condições e mesma região (zipcode).

**H2: Imóveis com mais de 1 andar são 43% mais caro que imóveis com apenas 1 andar.**
- Insight: Prospectar imóveis com mais de 1 andar que estão com preço até 20% mais caros que imóveis com apenas 1 andar.

**H3: Imóveis que foram reformados depois de 2005 são cerca de 28.5% mais caros que os foram antes.**
- Insight: Prospectar imóveis que foram reformados antes 2005 e que estejam com preço de no mínimo 35% a menos que imóveis que foram reformados depois de 2005. Com a ideia de fazer uma nova reforma e revendê-los com uma margem que cubra a reforma e gere lucro.


## 5.0 RESULTADOS PARA O NEGÓCIO

### 5.1 Produto disponibilizado para o time:
O dashboard disponibilizado para o time de negócio foi desenvolvido utilizando o pacote streamlit e hospedado na Cloud Heroku.
O acesso se dá via o link: https://house-rocket-company-analysis.herokuapp.com/

### 5.2 Resultados Financeiros
As recomendações de compra e venda foram feitas considerando os critérios de condições, região(zipcode) e estação do ano em que o imóvel ficou disponível. 
De modo que, o primeiro passo difiniu quais imóveis são recomendados para aquisição pela empresa, e fez-se da seguinte forma:
  - Agrupou-se os imóveis por zipcode e para calcular a mediana do preço para cada região;
  - Filtrou apenas os imóveis que estavam com grau de condição maior ou igual a três e que o seu preço estava menor que a mediana calculada para a sua região.
  
Considerando que estes imóveis recomendados para compra foram adquiridos pela empresa, a recomendação para preço de revenda foi feita a partir dos seguintes critérios:
  - Inicialmente foi extraído o mês em que o imóvel foi disponibilizado para que se ter conhecimento da estação do ano;
  - Os imóveis foram separados em dois datasets, onde em uma estavam os que foram disponibilizados no verão ou primavera, e no outro os que foram disponibilizados no inverno ou outono;
  - Em seguida, em cada um dos datasets, fez-se o agrupamento por zipcode e foi calculada a mediana do preço;
  - Comparou-se o preço do imóvel com o preço da mediana calculada e em caso deste ser inferior a métrica, o preço recomendado para revenda deve ser de 1.3x o preço do imóvel. Caso seja superior, então o recomendado é que seja de 1.1x o preço do imóvel.
  - Por fim, o lucro foi calculado para cada imóvel considerando esses critérios estabelecidos para valor da revenda e subtraindo-o do preço de compra do imóvel.

Assim, das 21.598 propriedades do portifólio 10.579 foram recomendadas para compra. Totalizando, aproximadamente, um investimento de $3.67Bi, um retorno de $4.35B e  gerando um **lucro de $757M**.

## 6.0 CONCLUSÕES & PRÓXIMOS PASSOS

O resultado obtido com o projeto foi considerado satisfatório pelo time de negócios e vendas, visto que o retorno previsto, seguindo as recomendações, é de 18% do investimento.

Para um próximo ciclo de análise dos dados, vale considerar diferentes cenários para validação de hipóteses, que incluam a possível influência no preço da distância do imóvel em relação a orla.


