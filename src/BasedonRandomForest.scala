import BasedonDecisionTree.getMetrics
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.Matrix

object BasedonRandomForest {
  def getMetrics(predictionAndLabels: RDD[(Double, Double)]): Seq[(String, String)] = {
    val metrics = new MulticlassMetrics(predictionAndLabels)
    Seq(
      ("Accuracy",
        (metrics.accuracy * 100).formatted("%.2f")),
      ("Precision",
        ((metrics.precision(0.0) + metrics.precision(1.0)) / 2 * 100).formatted("%.2f")),
      ("Recall",
        ((metrics.recall(0.0) + metrics.recall(1.0)) / 2 * 100).formatted("%.2f")),
      ("FPR",
        ((metrics.falsePositiveRate(0.0) + metrics.falsePositiveRate(1.0)) / 2 * 100).formatted("%.2f")),
      ("F1-Score",
        ((metrics.fMeasure(0.0) + metrics.fMeasure(1.0)) / 2 * 100).formatted("%.2f"))
    )
  }

  def run(args: Array[String],
          dataset: DataFrame,
          stringCols: Array[String],
          numericCols: Array[String],
          spark: SparkSession
         ) : (DataFrame, DataFrame, Matrix) = {

    import spark.implicits._
    println("Random Forest")
    /* Splitting training, test data */
    val splits = dataset.randomSplit(Array(0.7, 0.3), seed = 36L)
    val (trainingData, testData) = (splits(0), splits(1))
    trainingData.groupBy("label").count.show()
    testData.groupBy("label").count.show()
    trainingData.printSchema()

    /* Transform string columns to categorical columns */
    val stringIndexers = stringCols.map(col =>
      new StringIndexer()
        .setInputCol(col)
        .setOutputCol("Indexed" + col)
        .setHandleInvalid("keep")
    )

    /* Assemble features */
    val featureCols = stringCols.map("Indexed" + _) ++ numericCols
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    /* Init a estimator */
    val estimator = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxBins(1000)

    /* Initial a pipeline */
    val steps = stringIndexers ++ Array(assembler, estimator)
    val pipeline = new Pipeline().setStages(steps)

    /* Initial a evaluator */
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    /* Initial a Cross Validator */
    val validator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
    /*.setNumFolds(agrs(1).toInt)*/

    if (args(0) == "Parameters tuning") {
      /* Parameters tuning with CrossValidator and ParamGridBuilder */
      println(args(0))
      val paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxBins,  Array(10000, 11000))
        .addGrid(estimator.maxDepth, Array(5, 10, 20))
        .addGrid(estimator.impurity, Array("entropy", "gini"))
        .build()

      /* Add paramGrid into Cross Validation */
      validator.setEstimatorParamMaps(paramGrid)
    }
    else {
      val paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxBins,  Array(10000))
        .build()
      /* Add paramGrid into Cross Validation */
      validator.setEstimatorParamMaps(paramGrid)
    }

    /* Training pipeline */
    val model = validator.fit(trainingData)

    /* Test */
    val predictionDF = model
      .transform(testData)
      .select("label", "probability", "prediction")

    /* Random Forest features important */
    //    if (args(1) == "Features Importance") {
    println(args(1))
    val featuresImportance = model
      .bestModel
      .asInstanceOf[PipelineModel]
      .stages(stringIndexers.length + 1) // (stringIndexers.size + 1)'th transformer of PipelineModel is "rf" (RandomForest)
      .asInstanceOf[RandomForestClassificationModel]
      .featureImportances

    val feaImpDF = assembler.getInputCols.zip(featuresImportance.toArray)
      .sortBy(-_._2)
      .toSeq
      .toDF("name", "importance")

    // Compute raw scores on the test set
    val predictionAndLabels: RDD[(Double, Double)] = predictionDF
      .map(row => (row.getDouble(2), row.getDouble(0)))
      .rdd
    val multiclassMetrics = new MulticlassMetrics(predictionAndLabels)
    val scoreAndLabels = predictionDF
      .select("probability", "label")
      .map { case Row(prob: Vector, label: Double) => (prob(1), label) }
      .rdd
    // Instantiate metrics object
    val binaryMetrics = new BinaryClassificationMetrics(scoreAndLabels, numBins = 1000)
    val auPRC = binaryMetrics.areaUnderPR // AUPRC
    val auROC = binaryMetrics.areaUnderROC // AUROC
    val roc = binaryMetrics.roc // ROC Curve
    val PRC = binaryMetrics.pr // Precision-Recall Curve
    val metricsDF = (getMetrics(predictionAndLabels) ++ Seq(
      ("areaUnderPR", (auPRC * 100).formatted("%.2f")),
      ("areaUnderROC", (auROC * 100).formatted("%.2f"))
    )).toDF("Name", "Score%")
    (feaImpDF, metricsDF, multiclassMetrics.confusionMatrix)
  }
}