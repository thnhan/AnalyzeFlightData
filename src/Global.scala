/*import breeze.plot.{Figure, plot}
import org.apache.spark.sql.DataFrame*/
import org.apache.spark.sql.types.{ByteType, DoubleType, IntegerType, StringType, StructType}

case object Global {
  val num_defaut_label: String = "10"
//  val path_file_data: String = "data/flights_reduce.csv"

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

  var schema: StructType = new StructType()
    .add("num", IntegerType, nullable = true)
    .add("no", IntegerType, nullable = true)
    .add("year", IntegerType)
    .add("month", IntegerType)
    .add("dayofmonth", ByteType)
    .add("dayofweek", ByteType)
    .add("DepTime", DoubleType)
    .add("CRSDepTime", DoubleType)
    .add("ArrTime", DoubleType)
    .add("CRSArrTime", DoubleType)
    .add("UniqueCarrier", StringType)
    .add("FlightNum", StringType)
    .add("TailNum", StringType)
    .add("ActualElapsedTime", DoubleType)
    .add("CRSElapsedTime", DoubleType)
    .add("AirTime", DoubleType)
    .add("ArrDelay", DoubleType)
    .add("DepDelay", DoubleType)
    .add("Origin", StringType)
    .add("Dest", StringType)
    .add("Distance", DoubleType)
    .add("TaxiIn", StringType)
    .add("TaxiOut", StringType)
    .add("Cancelled", StringType)
    .add("CancellationCode", StringType)
    .add("Diverted", StringType)
    .add("CarrierDelay", StringType)
    .add("WeatherDelay", StringType)
    .add("NASDelay", StringType)
    .add("SecurityDelay", StringType)
    .add("LateAircraftDelay", StringType)

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