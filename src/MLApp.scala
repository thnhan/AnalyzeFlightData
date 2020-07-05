import org.apache.spark.ml.{Pipeline, PipelineModel, classification}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, GBTClassifier, LinearSVC, LinearSVCModel, MultilayerPerceptronClassifier, NaiveBayes, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{Bucketizer, LabeledPoint, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils


object MLApp {
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

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("MLApp")
      .getOrCreate

    import spark.implicits._
    val dataFrame = spark
      .read
      .format("csv")
      .option("header", value = true)
      .schema(Global.schema)
      .load("C:/Users/THNHAN/IdeaProjects/Dataset/Flight data for delay analysis/Delayed_Flights.csv")
    .as[Global.FlightDelay]

    /*    println(dataFrame.count())

    dataFrame.na.drop()
    dataFrame.show()
    println(dataFrame.count())*/

    val dataset = dataFrame
      .select("No", "DayOfWeek", "DepTime", "CRSDepTime", "ArrTime", "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "ActualElapsedTime", "CRSElapsedTime", "AirTime", "ArrDelay", "DepDelay", "Origin", "Dest", "Distance")

    /* Dataset */
    dataset.printSchema()
    dataset.show(truncate = false)
    dataset.createOrReplaceTempView("dataset")

    val bucketed = new Bucketizer()
      .setInputCol("DepDelay")
      .setOutputCol("label")
      .setSplits(Array(0.0, 40.0, Double.PositiveInfinity))

    val datasetWithLabel = bucketed.transform(dataset)
    datasetWithLabel.groupBy("label").count().show()

    var frac_pos_neg = datasetWithLabel
      .filter("label = 1.0")
      .count().toDouble

    frac_pos_neg = frac_pos_neg / (datasetWithLabel.count().toDouble - frac_pos_neg)
    println(frac_pos_neg)
    val frac = Map(0.0 -> Math.min(1.0, frac_pos_neg), 1.0 -> Math.min(1.0, frac_pos_neg))
    val balanceDataset = datasetWithLabel
      .stat.sampleBy(col = "label", frac, seed = 36L)

    val categoricalCols = Array(
      /*"UniqueCarrier",
      "Origin",
      "Dest",*/
      "FlightNum",
      "TailNum")

    val stringIndexers = categoricalCols.map(col =>
      new StringIndexer()
        .setInputCol(col)
        .setOutputCol(col + "Ix")
        .setHandleInvalid("keep")
    )

    val featureCols = categoricalCols.map(_ + "Ix") ++
      Array(
        /*"DayOfWeek",
                "ActualElapsedTime",
                "CRSElapsedTime",
                "AirTime",
                "ArrDelay",
                "Distance",*/
        "DepTime",
        "CRSDepTime",
        "ArrTime",
        "CRSArrTime"
      )

    val splits = balanceDataset.randomSplit(Array(0.7, 0.3), seed = 36L)
    val (trainingData, testData) = (splits(0).cache(), splits(1))
    trainingData.groupBy("label").count.show()
    testData.groupBy("label").count.show()
    trainingData.printSchema()

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val estimator = new LinearSVC()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(200)
      .setRegParam(0.001)
      /*.setMaxBins(10000)*/

    val steps1 = stringIndexers ++ Array(assembler) //, estimator)
    val pipeline1 = new Pipeline().setStages(steps1)
//      .asInstanceOf[PipelineModel]
    val testDataAndFea = pipeline1
      .fit(testData)
      .transform(testData)
    val trainingDataAndFea = pipeline1
      .fit(trainingData)
      .transform(trainingData)

    val svcModel = estimator.fit(trainingDataAndFea)
    val predLabels = testDataAndFea.select("label", "features").map { case Row(label: Double, v: Vector) =>
      val pred = svcModel.predict(v)
      (label, pred)
    }.toDF("label", "prediction")
    predLabels.show()


    /*/*val data = MLUtils.loadLibSVMFile(sparkContext(),
      "/home/vagrant/spark/data/mllib/sample_libsvm_data.txt")*/
    val labeledPoints = testDataAndFea.select("label", "features")
      .map {  case Row(label: Double, v: Vector) =>  LabeledPoint(label, v)
      }.rdd
    labeledPoints.take(10).foreach(println)
    val svmModel = SVMWithSGD.train(MLUtils.tlabeledPoints, 10)
//    svmWithSGD
*/
    val evaluator1 = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    /* Measure the accuracy */
    val accWithoutTuning1 = (evaluator1.evaluate(predLabels) * 100).formatted("%.2f")
    println(s"ACCURACY without parameters tuning: $accWithoutTuning1%")
    return  0

//    return  0
//    val predictionDF = estimator
//      .transform(testData)
//      .select($"label", $"probability", $"prediction")
//    predictionDF.show(truncate = false)

    return 0

    /* Training pipeline */
    val steps = stringIndexers ++ Array(assembler, estimator)
    val pipeline = new Pipeline().setStages(steps)
    val modelWithoutTuning = pipeline.fit(trainingData)

    val predictionDF = modelWithoutTuning
      .transform(testData)
      .select($"label", $"probability", $"prediction")
    predictionDF.show(truncate = false)
    return 0

    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    /* Measure the accuracy */
    val accWithoutTuning = (evaluator.evaluate(predictionDF) * 100).formatted("%.2f")
    println(s"ACCURACY without parameters tuning: $accWithoutTuning%")
    return  0

    /* Random Forest features important */
    /*val featureImportances = modelWithoutTuning
      //      .bestModel
      //      .asInstanceOf[PipelineModel]
      .stages(stringIndexers.length + 1) // (stringIndexers.size + 1)'th transformer of PipelineModel is "rf" (RandomForest)
      .asInstanceOf[DecisionTreeClassificationModel]
      .featureImportances

    assembler.getInputCols.zip(featureImportances.toArray)
      .sortBy(-_._2)
      .toSeq
      .toDF("name", "important")
      .show()*/

    // Compute raw scores on the test set
    val predictionAndLabels: RDD[(Double, Double)] = predictionDF
      .map(row => (row.getDouble(2), row.getDouble(0))).rdd

    val multiclassMetrics = new MulticlassMetrics(predictionAndLabels)

    val metricsDF = getMetrics(predictionAndLabels).toDF("Name", "Score")
    metricsDF.show()

    // Confusion matrix
    println("Confusion matrix:")
    println(multiclassMetrics.confusionMatrix)

    // Compute raw scores on the test set
    /*    val scoreAndLabels: RDD[(Double, Double)] = predictionDF
      .map(row => (row.getDouble(2), row.getDouble(0))).rdd
    predictionDF.printSchema()*/

    val scoreAndLabels = predictionDF
      .select("probability", "label")
      .map { case Row(prob: Vector, label: Double) => (prob(1), label) }
      .rdd

    // Instantiate metrics object
    val binaryMetrics = new BinaryClassificationMetrics(scoreAndLabels, numBins = 100)

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

    // Precision-Recall Curve
    val PRC = binaryMetrics.pr

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
    println("Area under precision-recall curve = " + auPRC)

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = binaryMetrics.roc

    // AUROC
    val auROC = binaryMetrics.areaUnderROC
    println("Area under ROC = " + auROC)

    /* Parameters tuning with CrossValidator and ParamGridBuilder */
/*
    val paramGrid = new ParamGridBuilder()
      .addGrid(estimator.maxBins, Array(10000, 11000))
      .addGrid(estimator.maxDepth, Array(2, 5, 10))
      .addGrid(estimator.numTrees, Array(100, 200, 300))
      .addGrid(estimator.impurity, Array("entropy", "gini"))
      .build()

    val validator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)

    val modelWithTuning = validator.fit(trainingData)
    val predictions = modelWithTuning.transform(testData)
    val accWithTuning = evaluator.evaluate(predictions)
    println(s"ACCURACY with parameters tuning: $accWithTuning%")
*/

    spark.stop()
  }
}