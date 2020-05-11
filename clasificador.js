async function trainModel(xTrain, yTrain, xTest, yTest){
    const model = tf.sequential();
    const learningRate=parseFloat(document.getElementById("learningRate").value);
    const numberOfEpochs=parseInt(document.getElementById("numberOfEpochs").value);
    const optimizer= tf.train.adam(learningRate);
    model.add(tf.layers.dense({units:10,activation:'sigmoid',inputShape:[xTrain.shape[1]]}));
    model.add(tf.layers.dense({units:3,activation:'softmax'}));
    //fit
    model.compile(
        {optimizer:optimizer,loss:'categoricalCrossentropy', metrics:['accuracy']});
    const history=await model.fit(xTrain, yTrain,
        {epochs:numberOfEpochs,validationData:[xTest,yTest],
        callbacks:tfvis.show.fitCallbacks({ name: 'Train',tab:'Train'}, ['loss', 'acc'])
        });
    return model;
}
async function fitModel(){
    const [xTrain, yTrain, xTest, yTest]= getData(.2);
    model = await trainModel(xTrain, yTrain, xTest, yTest);
    tfvis.show.modelSummary({name: 'Model Summary',tab:"Model Summary",styles: {
        width: 1000
     }}, model); 
    //download model
    await model.save('downloads://my-model');
    ///probar el algoritmo
    const yTrue=yTest.argMax(-1).dataSync();
    const predictions=model.predict(xTest);
    const yPred=predictions.argMax(-1).dataSync();
    //Creando tensores para la evaluación de metricas
    const yLabels = tf.tensor1d(yTrue);
    const yPredictions = tf.tensor1d(yPred);
    const accuracy = await tfvis.metrics.accuracy(yLabels,yPredictions);
    console.log("accuracy: "+accuracy)
    const ConfusionMatrix = await tfvis.metrics.confusionMatrix(yLabels,yPredictions);
    console.log("Confusion Matrix")
    console.log(ConfusionMatrix)
    // Render to visor
    const surface = { name: 'Confusion Matrix', tab: 'Matriz de confusión' };
    const data={values:ConfusionMatrix}
    tfvis.render.confusionMatrix(surface, data);


    
}

function PredictionErrorRate(d) {
    var correct=0;
    var wrong=0;
    for(var i=0; i<result.yTrue.length;i++){
        if(d.yTrue[i]==d.yPred[i]){
            correct++;
        }
        else{
            wrong++;
        }
    }
    console.log("Prediction error rate: "+(wrong/yTrue.length))
   return {correct,wrong,error:(wrong/yTrue.length)};
}





async function doClassPerfilPrefit(inputRaw){
    const model = await tf.loadLayersModel('modelTrained/perfiles.json');
    const input=tf.tensor2d(inputRaw,[1,7]);
    const predictionWithArgMax=model.predict(input).argMax(-1).dataSync();
    console.log(predictionWithArgMax)

}
