import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object BasedonDecisionTree {
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
    println("-------")
    /* Assemble features */
    val featureCols = stringCols.map("Indexed" + _) ++ numericCols
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    /* Init a estimator */
    val estimator = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
//      .setMaxBins(10000)

    /* Initial a pipeline */
    val steps = stringIndexers ++ Array(assembler, estimator)
    val pipeline = new Pipeline().setStages(steps)

    /* Initial a evaluator */
    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    /* Initial a Cross Validator */

    val validator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
    /*.setNumFolds(agrs(1).toInt)*/
    println("-------")

    if (args(0) == "Parameters tuning") {
      /* Parameters tuning with CrossValidator and ParamGridBuilder */
      println("Parameters tuning")
      val paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxBins,  Array(10000, 11000))
        .addGrid(estimator.maxDepth, Array(2, 5, 10))
        .addGrid(estimator.impurity, Array("entropy", "gini"))
        .build()

      /* Add paramGrid into Cross Validation */
      validator.setEstimatorParamMaps(paramGrid)
    }
    else {
      val paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxBins,  Array(10000))
        .addGrid(estimator.impurity, Array("gini"))
        .build()
      /* Add paramGrid into Cross Validation */
      validator.setEstimatorParamMaps(paramGrid)
    }

    /* Training pipeline */
    println("train")
    val model = validator.fit(trainingData)
    val predictionDF = model
      .transform(testData)
      .select("label", "probability", "prediction")
      .cache()
    predictionDF.show(truncate = false)
    println("finish training")

    /* Measure the accuracy */
    val accuracy = (evaluator.evaluate(predictionDF) * 100).formatted("%.2f")
    if (args(0) == "Parameters tuning") {
      println(s"ACCURACY with parameters tuning: $accuracy%")
    }
    else {
      println(s"ACCURACY without parameters tuning: $accuracy%")
    }

    // Compute raw scores on the test set
    val predictionAndLabels: RDD[(Double, Double)] = predictionDF
      .map(row => (row.getDouble(2), row.getDouble(0)))
      .rdd
    val multiclassMetrics = new MulticlassMetrics(predictionAndLabels)
    val metricsDF = getMetrics(predictionAndLabels).toDF("Name", "Score")
    metricsDF.show()

    /* Random Forest features important */
    if (args(1) == "Features Important") {
      println("Features Important")
      val featuresImportant = model
        .bestModel
        .asInstanceOf[PipelineModel]
        .stages(stringIndexers.length + 1) // (stringIndexers.size + 1)'th transformer of PipelineModel is "rf" (RandomForest)
        .asInstanceOf[DecisionTreeClassificationModel]
        .featureImportances

      assembler.getInputCols.zip(featuresImportant.toArray)
        .sortBy(-_._2)
        .toSeq
        .toDF("name", "important")
        .show()
    }

    // Confusion matrix
    println("Confusion matrix:")
    println(multiclassMetrics.confusionMatrix)

    val scoreAndLabels = predictionDF
      .select("probability", "label")
      .map { case Row(prob: Vector, label: Double) => (prob(1), label) }
      .rdd
    println("No Features Important")
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

    /*val beta = 0.5
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

    /*// Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)*/

    // ROC Curve
    val roc = binaryMetrics.roc
    // Precision-Recall Curve
    val PRC = binaryMetrics.pr

    return roc
  }
}