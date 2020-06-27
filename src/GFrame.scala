import org.apache.spark.{SparkConf, SparkContext}
import org.graphframes._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StringType, StructType}

object GFrame {
  def main(args: Array[String]): Unit = {
    // Vertex DataFrame
    val conf = new SparkConf().setAppName("GuiApp").setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("GFrame")
      .getOrCreate

    val schema = new StructType()
      .add("YEAR", IntegerType)
      .add("MONTH", IntegerType)
      .add("DAY", IntegerType)
      .add("DAY_OF_WEEK", IntegerType)
      .add("AIRLINE", StringType)
      .add("FLIGHT_NUMBER", StringType)
      .add("TAIL_NUMBER", StringType)
      .add("id", StringType, nullable = false)
      .add("DESTINATION_AIRPORT", StringType, nullable = false)
      .add("SCHEDULED_DEPARTURE", StringType)
      .add("DEPARTURE_TIME", StringType)
      .add("DEPARTURE_DELAY", StringType)
      .add("TAXI_OUT", StringType)
      .add("WHEELS_OFF", StringType)
      .add("SCHEDULED_TIME", StringType)
      .add("ELAPSED_TIME", StringType)
      .add("AIR_TIME", StringType)
      .add("DISTANCE", IntegerType, nullable = false)
      .add("WHEELS_ON", StringType)
      .add("TAXI_IN", StringType)
      .add("SCHEDULED_ARRIVAL", StringType)
      .add("ARRIVAL_TIME", StringType)
      .add("ARRIVAL_DELAY", StringType)
      .add("DIVERTED", StringType)
      .add("CANCELLED", StringType)
      .add("CANCELLATION_REASON", StringType)
      .add("AIR_SYSTEM_DELAY", StringType)
      .add("SECURITY_DELAY", StringType)
      .add("AIRLINE_DELAY", StringType)
      .add("LATE_AIRCRAFT_DELAY", StringType)
      .add("WEATHER_DELAY", StringType)


    val data_df = spark
      .read
      .format("csv")
      .option("header", value = true).schema(schema)
      .load("data/flights_reduce.csv")
//    data_df = data_df.withColumnRenamed("ORIGIN_AIRPORT", "id")
    data_df.printSchema()
    val v = data_df.dropDuplicates("id")
    println(v.count())

    //    data_df.show()
    val sqlContext = spark.sqlContext
    //    val v = sqlContext.createDataFrame(List(
    //    ("a", "Alice", 34),
    //    ("b", "Bob", 36),
    //    ("c", "Charlie", 30),
    //    ("d", "David", 29),
    //    ("e", "Esther", 32),
    //    ("f", "Fanny", 36),
    //    ("g", "Gabby", 60)
    //    )).toDF("id", "name", "age")
    //    // Edge DataFrame
    val e = sqlContext.createDataFrame(List(
      ("a", "b", "friend"),
      ("b", "c", "follow"),
      ("c", "b", "follow"),
      ("f", "c", "follow"),
      ("e", "f", "follow"),
      ("e", "d", "friend"),
      ("d", "a", "friend"),
      ("a", "e", "friend")
    )).toDF("src", "dst", "relationship")
    //
    //    v.show()
    //    e.show()
    //
    val g = GraphFrame(v, e)
//    val w = Window.orderBy("id")
//    val result = g.vertices
    g.vertices.show()
  }
}
