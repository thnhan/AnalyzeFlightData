import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object BasedonNaiveBayes {
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
         ) :(DataFrame, Matrix) = {

    import spark.implicits._

    /* Splitting training, test data */
    val splits = dataset.randomSplit(Array(0.7, 0.3), seed = 36L)
    val (trainingData, testData) = (splits(0), splits(1))
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
    val featureCols = stringCols.map("Indexed" + _) ++ numericCols
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    /* Init a estimator */
    val estimator = new NaiveBayes()
      .setLabelCol("label")
      .setFeaturesCol("features")

    /* Initial a pipeline */
    val steps = stringIndexers ++ Array(assembler, estimator)
    val pipeline = new Pipeline().setStages(steps)

    /* Initial a evaluator */
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    /* Initial a Cross Validator */
    /*val validator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
    /*.setNumFolds(agrs(1).toInt)*/

    if (args(0) == "Parameters tuning") {
      /* Parameters tuning with CrossValidator and ParamGridBuilder */
      println("Parameters tuning")
      val paramGrid = new ParamGridBuilder()
        .build()

      /* Add paramGrid into Cross Validation */
      validator.setEstimatorParamMaps(paramGrid)
    }
    else {
      val paramGrid = new ParamGridBuilder()
        .build()
      /* Add paramGrid into Cross Validation */
      validator.setEstimatorParamMaps(paramGrid)
    }*/

    /* Training pipeline */
    val model = pipeline.fit(trainingData)

    /* Test */
    val predictionDF = model
      .transform(testData)
      .select("label", "probability", "prediction")

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
    (metricsDF, multiclassMetrics.confusionMatrix)
  }
}

// Precision by threshold
/*val precision = binaryMetrics.precisionByThreshold
precision.foreach { case (t, p) =>
  println(s"Threshold: $t, Precision: $p")
}

// Recall by threshold
val recall = binaryMetrics.recallByThreshold
recall.foreach { case (t, r) =>
  println(s"Threshold: $t, Recall: $r")
}

// F-measure
val f1Score = binaryMetrics.fMeasureByThreshold
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 1")
}



val beta = 0.5
val fScore = binaryMetrics.fMeasureByThreshold(beta)
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 0.5")
}

// AUPRC
val auPRC = binaryMetrics.areaUnderPR
//    println("Area under precision-recall curve = " + auPRC)
// AUROC
val auROC = binaryMetrics.areaUnderROC
//    println("Area under ROC = " + auROC)

//    // Compute thresholds used in ROC and PR curves
//    val thresholds = precision.map(_._1)
val beta = 0.5
    val fScore = binaryMetrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }
// Compute thresholds used in ROC and PR curves
//    val thresholds = precision.map(_._1)
val roc = binaryMetrics.roc // ROC Curve

val PRC = binaryMetrics.pr // Precision-Recall Curve*/
