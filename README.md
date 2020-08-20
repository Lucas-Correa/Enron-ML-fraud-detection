# Enron-ML-fraud-detection

##  1. Motivação

O trabalho foi motivado pelo curso de Fundamentos de Data Science 2 pela Udacity. Esse  projeto era a última etapa para a conclusão do mesmo.

## 2. Objetivo do modelo 

O objetivo do projeto é tentar prever as pessoas envolvidas na fraude da Enron por meio de um conjunto de dados sobre seus funcionários. Com o aprendizado de máquina é possível treinar um algoritmo para classificar se determinado funcionário está ou não envolvido no esquema de fraude através de atributos comuns a cada tipo de pessoa. Especificamente nos dados da Enron, são disponibilizadas informações financeiras, principalmente sobre o valor das ações de cada funcionário, e de e-mails. Como as pessoas envolvidas na fraude acabaram usufruindo da supervalorização das ações da Enron, o conjunto de dados financeiros se torna uma boa métrica de quem participou do esquema. Da mesma forma, os dados de e-mail, especialmente os textos desses e-mails, pode ser muito útil para identificar padrões de palavras nos e-mails que indicam as pessoas que estão envolvidas na fraude.

## 3. Como usar

O  modelo está contruido em cima do arquivo poi.py que pode ser econtrado na pasta Projeto_Final. O arquivo irá baixar automáticamente os dados públicos do esquema de fraude da Enron

## 4. Sobre a Base e o processo realizado

O conjunto de dados possui 144 observações, sendo 18 pessoas de interesse e 126 colaboradores que não estavam envolvidos na fraude. Dos atributos escolhidos inicialmente, salary possui 36 observações sem registro e total_stock_value, 5.

### 4.1 Outliers 
Foram encontrados 2 outliers no conjunto de dados financeiro: um era a chave “TOTAL” no dicionário e a outra a chave "THE TRAVEL AGENCY IN THE PARK". O primeiro foi encontrado por meio de analise visual dos valores das características selecionadas e o outro pela leitura do documento em pdf disponibilizado sobre os últimos pagamento recebidos de cada colaborador. Como a chave “TOTAL” foi incluída por estar no documento pdf mencionado, era possível que outra chave indesejada existisse também, o que, por uma rápida conferencia nas chaves do dicionário, foi comprovado.

### 4.2 Atributos utilizados 
Eu utilizei os atributos: salary e total_stock_value. Além desses, criei atributos de palavras por meio dos textos dos e-mails disponíveis no dicionário de dados. Criei também outros 3 atributos:

1. Alavancagem: total_stock_value/ salary . A partir da hipótese de que as pessoas possuem uma proporção padrão de valor em ação em relação ao salário, acredito que sabendo da fraude, os colaboradores de interesse tenderiam a ter uma proporção maior do que a padrão.

2. de_poi: from_poi_to_this_person/ to_messages. Pessoas de interesse tenderiam a manter a frequência de e-mails entre si maior do que a para outras pessoas. Utilizei esse atributo das aulas por achar eficiente.

3. para_poi: from_this_person_to_poi/ from_messages. Pessoas de interesse tenderiam a manter a frequência de e-mails entre si maior do que a para outras pessoas. Utilizei esse atributo das aulas por achar eficiente.

Realizei o escalonamento de características na Alavancagem, salary e total_stock_value, pois eles ganhariam um peso muito maior comparado com os atributs de palavra e de_poi e para_poi. Como o de_poi e para_poi já são valores entre 0 e 1, não precisei modifica-los. Esse reescalonamento trás efeitos em apenas alguns algoritmos testados (GaussianBaes e Support Vector Machine), mas como ele não altera os resultados em nos demais classificadores, essa mudança acabou servindo para todos.

Como foram utilizadas as palavras dos textos dos e-mails da Enron, a quantidade de atributos no classificador ficou muito grande. Dessa forma, primeiro foi utilizada a função SelectPorcentaile() para selecionar, somente entre os atributos gerados a partir dos textos, as palavras que mais tinham significância para o classificador. A porcentagem escolhida foi de 5% de atributos totais, pois o tempo de processamento com essa quantidade de atributos já foi muito alto, em torno de 10 minutos, e uma quantidade maior tornaria inviável computacionalmente qualquer teste futuro. Foram selecionadas 1569 palavras. O uso desse método seguiu as orientações vistas nas aulas para redução da alta dimensionalidade que um conjunto de dados a partir de textos tem.

Adicionalmente, após juntar os 1569 atributos de palavras com os demais atributos escolhidos, eu utilizei o método de SelectKBest para testar com quantas características o modelo escolhido melhor se ajustava.

O meu teste realizado sem nenhum atributo criado, ou seja, apenas com salary e total_stock_values, resultou em um classificador com 0 de precisão e 0 de recall. Esses resultados foram suficientes para mim na decisão de manter os atributos criados.

### 4.3 Classificadores Testados 

Eu testei os algoritmos GaussianBaes, Support Vector Machine, Decision Trees e Adaboost. A escolha do algoritmo foi o Decision Trees. O teste inicial dos algoritmos trouxe uma acurácia alta entre 85 e 92% para os classificadores Support Vector Machine, Decision Trees e Adaboost, mas a precisão e a abrangência vieram iguais a 0. Já o GaussianBaes acabou tendo uma acurácia de 11% mais precisão de 0,25 e abrangência de 0,5.

Verificando os resultados sobre os mesmos parâmetros no tester.py, as métricas de precisão e acurácia se elevaram para todos os classificadores com exceção do SVM, que não teve nenhum verdadeiro positivo, e o GaussianBaes, que reduziu levemente o valor das suas métricas. O Decision Trees acabou trazendo melhores resultados e foi o que eu escolhi ao final.

No meu classificador final, eu acabei refinando o 'min_samples_split', que é referente a quantas vezes o meu classificador irá dividir o conjunto de pontos nos nodos da árvore. Pelo método do GridSearch eu disponibilizei 3 opções: 2, 20 e 50. A opção escolhida foi 2.

## 5. Resultados

As minhas métricas de avaliação para o meu classificador final foram:

- Accuracy: 0.90920
- Precision: 0.70370
- Recall: 0.5510
- F1: 0.61806
- F2: 0.57600
- Total predictions: 15000
- True positives: 1102
- False positives: 464
- False negatives: 898
- True negatives: 12536

De acordo com essas informações, o meu classificador acerta de forma geral 90,92% das vezes a sua classificação. Além disso, quando ele classifica uma pessoa como participante da fraude, ele costuma acertar a classificação sobre essa pessoa em 70,37% e, se a pessoa participou de fato na fraude, o meu classificador tem 55,10% de chances de marcar ela como uma pessoa de interesse.

## 6. Referências

Utilizei para esse projeto as fórmulas featureFormat() e parseOutText() que foram utilizadas nas aulas do módulo 4 do curso. 

Utilizei como base a documentação do scikit Learn para definir e usar os meus classificadores. Essa documentação pode ser encontrada no link:
https://scikit-learn.org/stable/index.html
