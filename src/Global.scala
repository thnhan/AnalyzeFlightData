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
  //  def parseFlight1(str: String): Flight1 = {
  //    val line = str.split(",")
  //    new Flight1(
  //      line(0).toByte,
  //      line(1).toByte,
  //      line(2).toByte,
  //      line(3).toByte,
  //      line(4),
  //      line(5),
  //      line(6).toInt
  //    )
  //  }
  //  Defining a Parse String function to parse input into Flight class
  //  function to parse input into 'Flight' class
   */
}