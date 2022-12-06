# Detecção e classificação de patologias cardíacas em eletrocardiograma utilizando redes neurais profundas


Doenças cardiovasculares vêm figurando, no decorrer das últimas décadas, dentre as principais causas de morte registradas no mundo. Tais doenças podem ser detectadas/diagnosticadas por um profissional de saúde através de um exame não invasivo denominado eletrocardiograma (ECG), o qual fornece uma representação da atividade elétrica do coração. Entretanto, a interpretação do ECG não é uma tarefa trivial, demandando considerável experiência do profissional de saúde na identificação apropriada de alterações morfológicas. Nesse contexto, técnicas de aprendizado de máquina (em especial, aprendizado profundo) vêm sendo amplamente utilizadas, na área de eletrofisiologia clínica, para criar modelos matemáticos capazes de detectar automaticamente padrões que caracterizam a presença ou ausência de patologias cardíacas. Contudo, apesar do bom desempenho alcançado, poucos trabalhos apresentados até então utilizam conjuntos de dados públicos no treinamento das arquiteturas, dificultando assim a validação dos resultados, comparações de desempenho e proposição de melhorias. Diante disso, o presente trabalho foca sobre o desenvolvimento de uma ferramenta computacional para classificação automática multirrótulos de 5 superclasses de patologias cardíacas. Especificamente, as arquiteturas introduzidas em (RAJPURKAR et al., 2017) e (RIBEIRO et al., 2020) são implementadas, utilizando linguagem Python juntamente as bibliotecas Tensorflow e Keras, e treinadas considerando um conjunto de dados público já rotulado por cardiologistas, denominado PTB-XL (WAGNER et al., 2020). O desempenho dos modelos obtidos (após o treinamento) é avaliado através da matriz de confusão multirrótulos (MLCM), a partir da qual as métricas precisão, sensibilidade e F1-score assim como seus valores médios micro, macro e weighted podem ser computados, conforme metodologia introduzida em (HEYDARIAN; DOYLE; SAMAVI, 2022). Ainda, comparações envolvendo o custo computacional para a implementação e treinamento das arquiteturas são apresentadas, abordando a quantidade de parâmetros, o número de épocas e o tempo transcorrido até o encerramento do treinamento. Os resultados obtidos mostraram que o modelo de (RIBEIRO et al., 2020) apresenta um desempenho similar, ou ainda ligeiramente melhor, para as diferentes métricas consideradas do que aquele obtido a partir da arquitetura de (RAJPURKAR et al., 2017); sobretudo, ao se levar em conta o custo computacional envolvido.
