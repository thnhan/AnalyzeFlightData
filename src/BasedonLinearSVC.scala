import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object BasedonLinearSVC {
  def run(args        : Array[String],
          dataset     : DataFrame,
          stringCols  : Array[String],
          numericCols : Array[String],
          spark: SparkSession
         ): Unit = {
    import spark.implicits._

    /* Splitting training, test data */
    val splits = dataset.randomSplit(Array(0.7, 0.3), seed = 36L)
    val (trainingData, testData) = (splits(0).cache(), splits(1).cache())
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
      .setMaxIter(200)
      .setRegParam(0.001)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    /* Initial a Cross Validator */
    val validator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
    /*.setNumFolds(agrs(1).toInt)*/

    if (args(0) != "Parameters tuning") {
      val paramGrid = new ParamGridBuilder()
        .build()
      /* Add paramGrid into Cross Validation */
      validator.setEstimatorParamMaps(paramGrid)
    }
    else {
      /* Parameters tuning with CrossValidator and ParamGridBuilder */
      val paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxIter, Array(100, 200, 300))
        .addGrid(estimator.regParam, Array(0.1, 0.01, 0.001))
        .build()
      /* Add paramGrid into Cross Validation */
      validator.setEstimatorParamMaps(paramGrid)
    }

    val trainingDataAndFea = pipeline.fit(trainingData).transform(trainingData)
    val testDataAndFea = pipeline.fit(testData).transform(testData)

    val svcModel = estimator.fit(trainingDataAndFea)
    val predLabels = testDataAndFea
      .select("label", "features")
      .map { case Row(label: Double, v: Vector) =>
        val prediction = svcModel
//          .bestModel
//          .asInstanceOf[LinearSVCModel]
          .predict(v)
        (label, prediction)
      }.toDF("label", "prediction")
    predLabels.show()

    /* Measure the accuracy */
    val accuracy = (evaluator.evaluate(predLabels) * 100).formatted("%.2f")
    if (args(0) != "Parameters tuning")
      println(s"ACCURACY without parameters tuning: $accuracy%")
    else
      println(s"ACCURACY with parameters tuning: $accuracy%")
  }
}
