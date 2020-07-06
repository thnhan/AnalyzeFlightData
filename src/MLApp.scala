import org.apache.spark.ml.{Pipeline, PipelineModel, classification}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, GBTClassifier, LinearSVC, LinearSVCModel, MultilayerPerceptronClassifier, NaiveBayes, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{Bucketizer, LabeledPoint, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.log4j.{Level, Logger}

object MLApp {
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

    val dataset = dataFrame
      .select("No", "DayOfWeek", "DepTime", "CRSDepTime", "ArrTime", "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "ActualElapsedTime", "CRSElapsedTime", "AirTime", "ArrDelay", "DepDelay", "Origin", "Dest", "Distance")

    /* Dataset */
    println(s"Flight Data:")
    dataFrame.show()
    println(s"Number of rows = ${dataFrame.count()}")
    println(s"Schema:")
    dataset.printSchema()
    dataset.show(truncate = false)
    //    dataset.createOrReplaceTempView("dataset")

    /* Creating Dataset */
    val bucketed = new Bucketizer()
      .setInputCol("DepDelay")
      .setOutputCol("label")
      .setSplits(Array(0.0, 40.0, Double.PositiveInfinity))

    val datasetWithLabel = bucketed.transform(dataset)
    datasetWithLabel.groupBy("label").count().show()

    var d = datasetWithLabel
      .filter("label = 1.0")
      .count().toDouble

    d = d / (datasetWithLabel.count().toDouble - d)
    println(d)
    val frac = Map(0.0 -> Math.min(1.0, d), 1.0 -> Math.min(1.0, d))
    val balanceDataset = datasetWithLabel
      .stat
      .sampleBy(col = "label", frac, seed = 36L)

    /* */
    val stringCols = Array(
//      "UniqueCarrier",
//      "Origin",
//      "Dest",
      "FlightNum",
      "TailNum")

    val numbericCols = Array(
//      "DayOfWeek",
//      "ActualElapsedTime",
//      "CRSElapsedTime",
//      "AirTime",
//      "ArrDelay",
//      "Distance",
      "DepTime",
      "CRSDepTime",
      "ArrTime",
      "CRSArrTime"
    )

    /* Run Linear SVC */
    BasedonSVC.run(
      Array(""),
      balanceDataset,
      stringCols,
      numbericCols,
      spark
    )

    /* Run Gradient-Boost Tree */
    BasedonGDBTree.run(
      Array(""),
      balanceDataset,
      stringCols,
      numbericCols,
      spark
    )

    /* Run DecisionTree */
    BasedonDecisionTree.run(
      Array("", "Features Important"),
      balanceDataset,
      stringCols,
      numbericCols,
      spark
    )
    spark.stop()
  }
}