# House Rocket Company

# Questão de negócio

O CEO da House Rocket Company está com dificuldades em definir quais imóveis a empresa deve comprar e revender de forma que o lucro seja maximizado. 
Por isso, esse projeto objetiva otimizar a tomada de decisão do CEO através de recomendações desenvolvidas analisando os dados do portfólio de imóveis da empresa. Tal análise visa identificar os imóveis que estão abaixo do preço mediano de venda para que sejam adquiridos pela House Rocket e posteriormente revendido a um preço ideal definido com base na análise dos dados. 

# Planejamento da solução

A base de dados foi obtida através em : https://www.kaggle.com/harlfoxem/housesalesprediction

As recomendações de compra e venda foram feitas a partir da análise dos dados considerando as características, estação do ano em que foi comprada e localização dos imóveis. Sendo assim, as medianas de preço que foram usadas como base para a decisão foi feita para cada região (zipcode) em questão, a fim de se ter uma análise mais justa.

O produto entregue ao CEO foi um link com um dashboard composto por mapas, tabelas com análises estatísticas e gráficos com filtros interativos, e duas tabelas de recomendações de compra e venda dos imóveis com o lucro previsto para cada revenda.

# Os principais insights de negócio

H1: Imóveis com 2 quartos são 20% mais caros que imóveis com menos quartos.
R: Como observado, os imóveis com 2 quartos são, em média, 25% mais caros que os imóveis com menos quartos.

H2: Imóveis com mais de 1 andar são 20% mais caros que imóveis com apenas 1 andar.
R: Como observado, os imóveis com mais de 1 andar são, em média, 43% mais caros que os imóveis de apenas 1 andar.

H3: Imóveis que possuem vista para água são 20% mais caros em média.
R: Como observado, os imóveis que possuem vista para água na realidade são, em média, 3x mais caros que os imóveis sem vista para água.

H4: O lucro na venda os imóveis no verão é 50% maior que no inverno.
R: Como observado, os imóveis vendidos durante o verão gerariam 58% de lucro a mais que os vendidos no inverno.

# Resultados financeiros para o negócio

Assumindo as recomendações feitas no projeto, o lucro final da empresa seria de: $35,3385,864.1

