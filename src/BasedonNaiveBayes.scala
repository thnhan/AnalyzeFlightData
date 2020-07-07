import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object BasedonNaiveBayes {
  def getMetrics(predictionAndLabels: RDD[(Double, Double)]): Seq[(String, String)] = {
    val metrics = new MulticlassMetrics(predictionAndLabels)
    Seq(
      ("Accuracy", metrics.accuracy.formatted("%.4f")),
      ("Precision (Label 0)", metrics.precision(0.0).formatted("%.4f")),
      ("Precision (Label 1)", metrics.precision(1.0).formatted("%.4f")),
      ("Recall (Label 0)", metrics.recall(0.0).formatted("%.4f")),
      ("Recall (Label 1)", metrics.recall(1.0).formatted("%.4f")),
      ("FPR (Label 0)", metrics.falsePositiveRate(0.0).formatted("%.4f")),
      ("FPR (Label 1)", metrics.falsePositiveRate(1.0).formatted("%.4f")),
      ("F1-Score (Label 0)", metrics.fMeasure(0.0).formatted("%.4f")),
      ("F1-Score (Label 1)", metrics.fMeasure(1.0).formatted("%.4f")),
      ("Weighted precision", metrics.weightedPrecision.formatted("%.4f")),
      ("Weighted recall", metrics.weightedRecall.formatted("%.4f")),
      ("Weighted F1 score", metrics.weightedFMeasure.formatted("%.4f")),
      ("Weighted false positive rate", metrics.weightedFalsePositiveRate.formatted("%.4f"))
    )
  }

  def run(args: Array[String],
          dataset: DataFrame,
          stringCols: Array[String],
          numericCols: Array[String],
          spark: SparkSession
         ): Unit = {
    import spark.implicits._

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
    val predictionDF = model
      .transform(testData)
      .select("label", "probability", "prediction")

    /* Measure the accuracy */
    predictionDF.show(truncate = false)
    val accuracy = (evaluator.evaluate(predictionDF) * 100).formatted("%.2f")
    println(s"ACCURACY: $accuracy%")


    // Compute raw scores on the test set
    val predictionAndLabels: RDD[(Double, Double)] = predictionDF
      .map(row => (row.getDouble(2), row.getDouble(0)))
      .rdd
    val multiclassMetrics = new MulticlassMetrics(predictionAndLabels)
    val metricsDF = getMetrics(predictionAndLabels).toDF("Name", "Score")
    metricsDF.show()

    // Confusion matrix
    println("Confusion matrix:")
    println(multiclassMetrics.confusionMatrix)

    val scoreAndLabels = predictionDF
      .select("probability", "label")
      .map { case Row(prob: Vector, label: Double) => (prob(1), label) }
      .rdd

    // Instantiate metrics object
    val binaryMetrics = new BinaryClassificationMetrics(scoreAndLabels, numBins = 2)

    // Precision by threshold
    val precision = binaryMetrics.precisionByThreshold
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

/*    val beta = 0.5
    val fScore = binaryMetrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }*/

    // AUPRC
    val auPRC = binaryMetrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)
    // AUROC
    val auROC = binaryMetrics.areaUnderROC
    println("Area under ROC = " + auROC)

//    // Compute thresholds used in ROC and PR curves
//    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = binaryMetrics.roc
    // Precision-Recall Curve
    val PRC = binaryMetrics.pr

    return roc
  }
}