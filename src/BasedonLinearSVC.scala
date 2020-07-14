import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object BasedonLinearSVC {
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
         ): (DataFrame, Matrix) = {

    import spark.implicits._
    println("Linear SVC")

    /* Splitting training, test data */
    val splits = dataset.randomSplit(Array(0.7, 0.3), seed = 36L)
    val (trainingData, testData) = (splits(0).cache(), splits(1).cache())
    println("Total number of samples for training")
    trainingData.groupBy("label").count.show()
    println("Total number of samples for test")
    testData.groupBy("label").count.show()

    /* Transform string columns to categorical columns */
    val stringIndexers = stringCols.map(col =>
      new StringIndexer()
        .setInputCol(col)
        .setOutputCol("Indexed" + col)
        .setHandleInvalid("keep")
    )

    /* Assemble features */
    val featureCols = stringCols.map(col => "Indexed" + col) ++ numericCols
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    /* Initial a pipeline */
    val steps = stringIndexers ++ Array(assembler)
    val pipeline = new Pipeline().setStages(steps)

    val estimator = new LinearSVC()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(100)
      .setRegParam(0.01)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    /* Initial a Cross Validator */
    val validator = new CrossValidator()
      .setEstimator(estimator)
      .setEvaluator(evaluator)
    /*.setNumFolds(agrs(1).toInt)*/

    if (args(0) != "Parameters tuning") {
      val paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxIter, Array(100))
        .addGrid(estimator.regParam, Array(0.01))
        .build()
      /* Add paramGrid into Cross Validation */
      validator.setEstimatorParamMaps(paramGrid)
    }
    else {
      /* Parameters tuning with CrossValidator and ParamGridBuilder */
      //      println("Parameters tuning")
      val paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxIter, Array(100, 200, 300))
        .addGrid(estimator.regParam, Array(0.1, 0.01, 0.001))
        .build()
      /* Add paramGrid into Cross Validation */
      validator.setEstimatorParamMaps(paramGrid)
    }

    /* */
    val trainingDataAndFea = pipeline.fit(trainingData).transform(trainingData)
    val testDataAndFea = pipeline.fit(testData).transform(testData)

    /* Training */
    val svcModel = validator.fit(trainingDataAndFea)

    /* Test */
    val scoreAndLabels = testDataAndFea
      .select("label", "features")
      .map { case Row(label: Double, v: Vector) =>
        val prediction = svcModel
          .bestModel
          .asInstanceOf[LinearSVCModel]
          .predict(v)
        (label, prediction)
      }.rdd

    // Instantiate metrics object
    val multiclassMetrics = new MulticlassMetrics(scoreAndLabels)
    val binaryMetrics = new BinaryClassificationMetrics(scoreAndLabels, numBins = 1000)
    val auPRC = binaryMetrics.areaUnderPR // AUPRC
    val auROC = binaryMetrics.areaUnderROC // AUROC
    val roc = binaryMetrics.roc // ROC Curve
    val PRC = binaryMetrics.pr // Precision-Recall Curve
    val metricsDF = (getMetrics(scoreAndLabels) ++ Seq(
      ("areaUnderPR", (auPRC * 100).formatted("%.2f")),
      ("areaUnderROC", (auROC * 100).formatted("%.2f"))
    )).toDF("Name", "Score%")

    (metricsDF, multiclassMetrics.confusionMatrix)
  }
}
