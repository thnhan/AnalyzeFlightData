//import breeze.plot.{Figure, plot}
//import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{ByteType, DoubleType, IntegerType, StringType, StructField, StructType}

case object Global {
  val num_defaut_label: String = "10"
  val path_file_data: String = "data/flights_reduce.csv"

  val list = List(
    "Show the FLIGHTS data",
    "Show information about the number of AIRPORTS and ROUTES",
    "List all VERTICES and EDGES of the graph",
    "List all EDGES which have distance longer than ...",
    "Find the airport with the highest incoming flights",
    "Find the airport with the highest outgoing flights",
    "Find the most important airports",
    "Find paths with the lowest flight costs",
    "Find the airport with the lowest flight costs"
  )

  case class FlightDelay
  (
    num: Int,
    no: Int,
    year: Int,
    month: Integer,
    dayofmonth: Byte,
    dayofweek: Byte,
    deptime: Double,
    crsdepTime: Double,
    arrtime: Double,
    crsarrtime: Double,
    uniquecarrier: String,
    flightnum: String,
    tailnum: String,
    actualelapsedtime: Double,
    CRSElapsedTime: Double,
    AirTime: Double,
    ArrDelay: Double,
    DepDelay: Double,
    Origin: String,
    Dest: String,
    Distance: Double,
    TaxiIn: String,
    TaxiOut: String,
    Cancelled: String,
    CancellationCode: String,
    Diverted: String,
    CarrierDelay: String,
    WeatherDelay: String,
    NASDelay: String,
    SecurityDelay: String,
    LateAircraftDelay: String
  ) extends Serializable

  val schema: StructType = StructType(Array(
    StructField("num", IntegerType),
    StructField("no", IntegerType),
    StructField("year", IntegerType),
    StructField("month", IntegerType),
    StructField("dayofmonth", ByteType),
    StructField("dayofweek", ByteType),
    StructField("DepTime", DoubleType),
    StructField("CRSDepTime", DoubleType),
    StructField("ArrTime", DoubleType),
    StructField("CRSArrTime", DoubleType),
    StructField("UniqueCarrier", StringType),
    StructField("FlightNum", StringType),
    StructField("TailNum", StringType),
    StructField("ActualElapsedTime", DoubleType),
    StructField("CRSElapsedTime", DoubleType),
    StructField("AirTime", DoubleType),
    StructField("ArrDelay", DoubleType),
    StructField("DepDelay", DoubleType),
    StructField("Origin", StringType),
    StructField("Dest", StringType),
    StructField("Distance", DoubleType),
    StructField("TaxiIn", StringType),
    StructField("TaxiOut", StringType),
    StructField("Cancelled", StringType),
    StructField("CancellationCode", StringType),
    StructField("Diverted", StringType),
    StructField("CarrierDelay", StringType),
    StructField("WeatherDelay", StringType),
    StructField("NASDelay", StringType),
    StructField("SecurityDelay", StringType),
    StructField("LateAircraftDelay", StringType)
  ))

  //Creating a Case Class Flight
  case class Flight(
                     YEAR: String,
                     MONTH: String,
                     DAY: String,
                     DAY_OF_WEEK: String,
                     AIRLINE: String,
                     FLIGHT_NUMBER: String,
                     TAIL_NUMBER: String,
                     ORIGIN_AIRPORT: String,
                     DESTINATION_AIRPORT: String,
                     SCHEDULED_DEPARTURE: String,
                     DEPARTURE_TIME: String,
                     DEPARTURE_DELAY: String,
                     TAXI_OUT: String,
                     WHEELS_OFF: String,
                     SCHEDULED_TIME: String,
                     ELAPSED_TIME: String,
                     AIR_TIME: String,
                     DISTANCE: Int,
                     WHEELS_ON: String,
                     TAXI_IN: String,
                     SCHEDULED_ARRIVAL: String,
                     ARRIVAL_TIME: String,
                     ARRIVAL_DELAY: String,
                     DIVERTED: String,
                     CANCELLED: String,
                     CANCELLATION_REASON: String,
                     AIR_SYSTEM_DELAY: String,
                     SECURITY_DELAY: String,
                     AIRLINE_DELAY: String,
                     LATE_AIRCRAFT_DELAY: String,
                     WEATHER_DELAY: String
                   )

  case class Flight1(
                      YEAR: String,
                      MONTH: String,
                      DAY: String,
                      DAY_OF_WEEK: String,
                      ORIGIN_AIRPORT: String,
                      DESTINATION_AIRPORT: String,
                      DISTANCE: Int
                    )

  /*
    def parseFlight1(str: String): Flight1 = {
      val line = str.split(",")
      new Flight1(
        line(0).toByte,
        line(1).toByte,
        line(2).toByte,
        line(3).toByte,
        line(4),
        line(5),
        line(6).toInt
      )
    }
    Defining a Parse String function to parse input into Flight class
    function to parse input into 'Flight' class
   */


  /*def plottingROC(rocDF: DataFrame) = {
    val fig = Figure()
    val chart = fig.subplot(0)
    chart.title = "ROC curve"
    chart.xlim(2,100)
    chart.ylim(0,100000)

    val plotX = Array(1,2,3,4)//rocDF.collect().map(_.getDouble(0))
    val plotY = Array(1,2,3,4)//rocDF.collect().map(_.getDouble(1))
    chart += plot(plotX, plotY)
    chart.xlabel = "FPR"
    chart.ylabel = "TPR"
    chart.refresh()
  }*/
}