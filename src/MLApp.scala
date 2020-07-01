import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{Bucketizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Dataset, SparkSession}

object MLApp {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("GuiApp").setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("MLApp")
      .getOrCreate

    import spark.implicits._
    val dataFrame: Dataset[Global.FlightDelay] = spark
      .read
      .format("csv")
      .option("header", value = true)
      .schema(Global.schema)
      .load("C:/Users/THNHAN/IdeaProjects/Dataset/Flight data for delay analysis/Delayed_Flights.csv")
      .as[Global.FlightDelay]

    println(dataFrame.count())

    dataFrame.na.drop()
    dataFrame.show()
    println(dataFrame.count())

    val dataset = dataFrame
      .select("No", "DayOfWeek", "DepTime", "CRSDepTime", "ArrTime", "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "ActualElapsedTime", "CRSElapsedTime", "AirTime", "ArrDelay", "DepDelay", "Origin", "Dest", "Distance")

    dataset.printSchema()
    dataset.show(truncate = false)
    dataset.createOrReplaceTempView("dataset")

    val bucketizer = new Bucketizer()
      .setInputCol("DepDelay")
      .setOutputCol("label")
      .setSplits(Array(0.0, 40.0, Double.PositiveInfinity))
    val datasetWithLabel = bucketizer.transform(dataset)
    datasetWithLabel.groupBy("label").count().show()
    var frac_pos_neg = datasetWithLabel.filter("label = 1.0").count().toDouble
    frac_pos_neg = frac_pos_neg / (datasetWithLabel.count().toDouble - frac_pos_neg)
    println(frac_pos_neg)
    val frac = Map(0.0 -> Math.min(1.0, frac_pos_neg), 1.0 -> Math.min(1.0, frac_pos_neg))
    val balanceDataset = datasetWithLabel
      .stat.sampleBy(col = "label", frac, seed = 36L)

    val catogoricalCols = Array(
      "UniqueCarrier",
      "FlightNum",
      "TailNum",
      "Origin",
      "Dest"
    )

    val stringIndexers = catogoricalCols.map(col =>
      new StringIndexer()
        .setInputCol(col)
        .setOutputCol(col + "Ix")
        .setHandleInvalid("keep")
    )

    val featureCols = Array(
      "DayOfWeek",
      "DepTime",
      "CRSDepTime",
      "ArrTime",
      "CRSArrTime",
      "UniqueCarrierIx",
      "FlightNumIx",
      "TailNumIx",
      "ActualElapsedTime",
      "CRSElapsedTime",
      "AirTime",
      "ArrDelay",
      "OriginIx",
      "DestIx",
      "Distance"
    )

    val splits = balanceDataset.randomSplit(Array(0.7, 0.3), seed = 36L)
    val (trainingData, testData) = (splits(0), splits(1))
    trainingData.groupBy("label").count.show()
    testData.groupBy("label").count.show()
    trainingData.printSchema()

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxBins(10000)

    /* Training pipeline */
    val steps = stringIndexers ++ Array(assembler, rf)
    val pipeline = new Pipeline().setStages(steps)
    val pipelineModel = pipeline.fit(trainingData)
    val predictionDF = pipelineModel.transform(testData.na.drop()).select("label", "probability", "prediction")
    predictionDF.show(truncate = false)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    /* Measure the accuracy */
    val acc = (evaluator.evaluate(predictionDF) * 100).formatted("%.2f")
    println(s"ACCURACY: $acc%")

    /* Parameters tuning with CrossValidator and ParamGridBuilder */
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxBins, Array(10000, 11000))
//      .addGrid(rf.maxDepth, Array(2, 5, 10))
      .addGrid(rf.numTrees, Array(100, 200, 300))
      .addGrid(rf.impurity, Array("entropy", "gini"))
      .build()

    val validator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
    //.setNumFolds(3)
    val pipelineGrid = validator.fit(trainingData)
    val predictions = pipelineGrid.transform(testData)
    val aucROC = evaluator.evaluate(predictions)
    println(aucROC)
    pipeline.fit(trainingData)

    /* Calculating TN, FN, TP, FP */
/*    val correct = ip.filter($"label" === $"prediction").count()
    ip.printSchema()
    ip.show(374048)
    ip.filter($"label".isNaN).show()
    print(ip.filter("label = prediction").count())
    println(correct.toDouble / totalpre.toDouble)*/
  }
}