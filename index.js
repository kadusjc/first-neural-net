import tf from '@tensorflow/tfjs-node';


async function _trainModel(xs, ys) {
    // Criamos um modelo sequencial, que é uma pilha de camadas
    const model = tf.sequential();

    //Primeira camada da rede: (camada oculta)
    //Entrada de 7 posições (idade normalizada + 3 cores + 3 localizações) e 80 neurônios, com função de ativação ReLU

    //80 neuronios, pois tem pouca base de treinamento
    //Quanto mais neurônios, mais complexa a rede, mas também mais propensa a aprender e a overfitting (ajuste excessivo aos dados de treinamento)
    //ReLu é uma função de ativação que ajuda a rede a aprender padrões complexos, introduzindo não linearidade. Ela retorna 0 para entradas negativas e o valor da entrada para entradas positivas, o que ajuda a rede a aprender relações mais complexas entre os dados.
    //Relu é como um filtro que permite que apenas os sinais positivos (dados interessantes) passem, ajudando a rede a focar em padrões relevantes e a evitar o problema de gradientes desaparecendo, onde os pesos param de ser atualizados durante o treinamento. Isso torna a ReLU uma escolha popular para camadas ocultas em redes neurais.
    //Se a informação chegou nesse neurônio e é positiva, passa para frente, se for 0 ou negativa, pode jogar fora. Não serve para nada
    model.add(tf.layers.dense({inputShape: [7], units: 80, activation: 'relu'}));


    //Saída da rede: 3 neurônios (premium, medium, basic) e função de ativação softmax
    //Softmax é uma função de ativação que transforma os valores de saída em probabilidades, ou seja, cada neurônio na camada de saída representará a probabilidade de pertencer a uma classe específica (premium, medium, basic). A soma das saídas da camada de saída será igual a 1, o que facilita a interpretação dos resultados como probabilidades.
    model.add(tf.layers.dense({units: 3, activation: 'softmax'}));


    //Compilamos o modelo, definindo a função de perda e o otimizador
    //A função de perda 'categoricalCrossentropy' é usada para problemas de classificação multiclasse, onde as classes são mutuamente exclusivas (como premium, medium, basic). Ela mede a diferença entre as distribuições de probabilidade previstas pelo modelo e as distribuições reais (labels), penalizando mais fortemente as previsões incorretas.
    //O otimizador 'adam' (Adaptive Moment Estimation) é um algoritmo de otimização que ajusta os pesos do modelo durante o treinamento para minimizar a função de perda. Ele combina as vantagens dos otimizadores AdaGrad e RMSProp, adaptando a taxa de aprendizado para cada peso individualmente, o que pode levar a uma convergência mais rápida e eficiente.
    //Vai aprender com o histórico de erros e acertos, ajustando os pesos para melhorar a precisão das previsões ao longo do tempo.
    //loss: categoricalCrossentropy, Compara o que o modelo acha (os scores de cada categoria) com a resposta correta (labels) e calcula o erro. O modelo tenta minimizar esse erro durante o treinamento, ajustando seus pesos para melhorar a precisão das previsões.
    //a categoria premium será sempre [1, 0, 0]

    //Quanto mais distante da previsão correta, maior será a perda, e o modelo tentará ajustar seus pesos para reduzir essa perda ao longo do tempo, melhorando assim a precisão das previsões.
    //Exemplo classico: classificação de imagens, onde cada imagem é classificada em uma categoria específica (gato, cachorro, etc.). A função de perda ajuda a medir o quão bem o modelo está aprendendo a classificar as imagens corretamente. Recomendação e categorização de usuário. 
    //Qualquer coisa em que a resposta certa seja apenas entre várias opções possíveis. Como classificação de texto, análise de sentimentos, etc.
    model.compile({ 
        optimizer: 'adam', 
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'] // Métrica para avaliar a precisão do modelo durante o treinamento
    });


    //Treinamos o modelo usando os dados de entrada (xs) e as labels (ys)
    //O método fit é assíncrono, então usamos await para esperar o treinamento ser concluído antes de continuar.
    //O número de épocas (epochs) define quantas vezes o modelo passará por todo o conjunto de dados durante o treinamento. Um número maior de épocas pode levar a um modelo mais preciso, mas também pode aumentar o risco de overfitting (ajuste excessivo aos dados de treinamento). É importante monitorar a perda e a precisão durante o treinamento para determinar quando parar.
    await model.fit(xs, ys, {
        verbose: 0, // Configura o nível de detalhes do log durante o treinamento. 0 = sem logs, 1 = barra de progresso, 2 = uma linha por época. Usar 0 para evitar poluição do console.
        epochs: 100, // Número de vezes que o modelo passará por todo o conjunto de dados. Número de vezes de treinamento. Quanto mais épocas, mais o modelo aprende, mas também aumenta o risco de overfitting (ajuste excessivo aos dados de treinamento). É importante monitorar a perda e a precisão durante o treinamento para determinar quando parar.
        shuffle: true, // Embaralha os dados a cada época (cada treinamento que ele fizer) para melhorar a generalização do modelo - Para não ficar algoritmo viciado (para evitar viés)
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, loss percentage = ${(logs.loss * 100).toFixed(2)}%, accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    });

    return model;
}

async function _predict(model, pessoa) {
    //transformar array js para o tensor do tensorflow
    //input é um tensor 2D, mesmo formato dos dados de treinamento
    const tfInput = tf.tensor2d(pessoa);

    // Output - A previsão é um tensor de 3 possibilidades
    const predict = await model.predict(tfInput);
    const predictedValuesArray = await predict.array(); // Converte o tensor de previsão para um array JavaScript
    //console.log("Probabilidades previstas para cada categoria (premium, medium, basic):", predictedValuesArray[0]);
    const results = predictedValuesArray[0].map((probabilidade, index) => { return { probabilidade, index } }); 
    return results;
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

//inputXs.print();
//outputYs.print();

//Etapa de Treinamento - quanto mais dado melhor. Assim o algoritmo consegue entender melhor os padrões e fazer previsões mais precisas. O modelo aprende a partir dos dados de entrada (inputXs) e das labels (outputYs) para ajustar seus pesos e melhorar suas previsões ao longo do tempo. Quanto mais dados de treinamento, melhor o modelo pode aprender a generalizar e fazer previsões precisas em novos dados que ele nunca viu antes.
const model = await _trainModel(inputXs, outputYs);

//Já temos o modelo treinado, agora podemos usar ele para fazer previsões com novos dados de entrada (novas pessoas) e ver a probabilidade de cada categoria (premium, medium, basic) para cada pessoa.
// Mais proximo do Carlos
const pessoa = { nome: 'José Toalha', idade: 28, cor: 'verde', localizacao: 'Curitiba' }
//Normalizando a idade da pessoa com mesmo padrão de treino
//idadeMin=25 idadeMax=40 = (idade - idadeMin)/ (idadeMax-idadeMin) = 0.2
const idadeNormalizada = (pessoa.idade - 25) / (40 - 25); // Normaliza a idade para o intervalo [0, 1]
const corOneHot = [0, 0, 1]; // Verde
const localizacaoOneHot = [0, 0, 1]; // Curitiba

//Criando o vetor de entrada para a pessoa
const pessoaNormalizada = [
    [
        idadeNormalizada,
        ...corOneHot,
        ...localizacaoOneHot
    ]
];

/*array ficara
[   
    0.2, //idade
    0, //cor azul
    0, //cor vermelho
    1, //cor verde
    0, //localizacao SP
    0, //localizacao Rio
    1, //localizacao Curitiba
]
*/

const predictions = await _predict(model, pessoaNormalizada);
const results = predictions
    .sort((a, b) => b.probabilidade - a.probabilidade) // Ordena as previsões por probabilidade (do mais provável para o menos provável )
    .map((pred) => `\n${labelsNomes[pred.index]}: ${(pred.probabilidade * 100).toFixed(2)}%`)
    .join('\n') // Mapeia as previsões para o formato "categoria: probabilidade%"

console.log("********** Probabilidades previstas para cada categoria (premium, medium, basic):", results);



