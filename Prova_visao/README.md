# Descrição do problema:
O problema seria referente o treinamento de uma rede neural convolucional (no meu caso), capaz de classificar imagens como cachorro ou gato.

# Justificativa das técnicas utilizadas:
Foi utilizado várias bibliotecas para conseguir implementar o dataset e conseguir treiná-lo com o CNN, utilizando os métodos para teste e treinamento do modelo.

# Etapas realizadas:
Ao todo, foram realizadas 5 etapas:
Coleta dos dados do dataset;
Pré-processamento das imagens;
Utilização do modelo CNN;
Treinamento e teste do modelo;
Avaliação do modelo com imagens fora do dataset.

# Resultados obtidos:
O modelo não foi capaz de utilizar o dataset de maneira correta, pelo dataset possuir imagens corrompidas, o que impediu o fechamento correto do código. Porém, se não houvesse imagens corrompidas, o modelo ia passar as imagens pelas 3 camadas convolucionais, a primeira de inputa, a segunda para aprofundar o conhecimento do modelo e a terceira para dar uma resposta. Foi utilizado esse "Método" de pooling, que reduz a dimensão da imagem, mantendo informações mais importantes e descartando as redundantes. Depois temos o Flatten para fazer a matriz virar um vetor para passar pelas camadas densas, sendo a última delas para representar 2 classes (no caso o resultado seria ou cachorro ou gaot). O modelo foi treinado em 10 épocas, a quantidade padrão, mas como o código não conseguiu executar corretamente por conta das imagens corrompidas, não pude testar nenhuma quantidade de épocas. Depois disso o modelo ia ser avaliado com as 6 imagens que estão na pasta ./imagens, de forma a trazer a precision, recall e f1-score.

# Tempo total gasto:
Mais do que o necessário, tive muitos probleminhas pra conseguir instalar as libs utilizadas no pc da faculdade, consegui resolver chegando em casa, mas mesmo assim tive outros problemas com o código em si. De maneira geral eu acho que foi, tempo codando mesmo, umas 4 horas e pouco, talvez 5 horas.

# Dificuldades encontradas:
Dificuldade principal foi o mal entendimento do enunciado, fiquei martelando muito que era pra jogar as imagens com os filtros para o modelo avaliar, estava tentando rodar tudo em um único código e o pc da faculdade não tava deixando eu instalar as libs pra poder rodar tudo. Depois que entendi que eram duas coisas separadas, a etapa dos filtros foi bem tranquila, a parte da rede convolucional que me pegou um pouco, o cansaso acabou tomando conta e não consegui resolver esse b.o. das imagens corrompidas.