import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object BasedonSVC {
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

    /* Initial a pipeline */
    val steps = stringIndexers ++ Array(assembler)
    val pipeline = new Pipeline().setStages(steps)

    val trainingDataAndFea = pipeline.fit(trainingData).transform(trainingData)
    val testDataAndFea = pipeline.fit(testData).transform(testData)

    val estimator = new LinearSVC()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(200)
      .setRegParam(0.0001)

    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    /* Initial a Cross Validator */
    val validator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
    /*.setNumFolds(agrs(1).toInt)*/

    if (args(0) != "Parameters tuning") {
      /* Training pipeline */
      val modelWithoutTuning = pipeline.fit(trainingData)
      val predictionDF = modelWithoutTuning
        .transform(testData)
        .select("label", "probability", "prediction")

      /* Measure the accuracy */
      predictionDF.show(truncate = false)
      val accWithoutTuning = (evaluator.evaluate(predictionDF) * 100).formatted("%.2f")
      println(s"ACCURACY without parameters tuning: $accWithoutTuning%")
    }
    else {
      /* Parameters tuning with CrossValidator and ParamGridBuilder */
      val paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxIter, Array(100, 200, 300))
        .addGrid(estimator.regParam, Array(0.01, 0.001, 0.0001))
        .build()

      /* Add paramGrid into Cross Validation */
      validator.setEstimatorParamMaps(paramGrid)
    }

    val svcModel = validator.fit(trainingDataAndFea)
    val predLabels = testDataAndFea
      .select("label", "features")
      .map { case Row(label: Double, v: Vector) =>
        val prediction = svcModel
          .bestModel
          .asInstanceOf[LinearSVCModel]
          .predict(v)
        (label, prediction)
      }.toDF("label", "prediction")
    predLabels.show()

    /* Measure the accuracy */
    val accuracy = (evaluator.evaluate(predLabels) * 100).formatted("%.2f")
    println(s"ACCURACY without parameters tuning: $accuracy%")
  }
}
