import org.apache.spark.graphx._

object AirportGraph {
  /*def parseFlight(line: Array[String]): Global.Flight = {
    Global.Flight(
      line(0),
      line(1),
      line(2),
      line(3),
      line(4),
      line(5),
      line(6),
      line(7),
      line(8),
      line(9),
      line(10),
      line(11),
      line(12),
      line(13),
      line(14),
      line(15),
      line(16),
      line(17).toInt,
      line(18),
      line(19),
      line(20),
      line(21),
      line(22),
      line(23),
      line(24),
      line(25),
      line(26),
      line(27),
      line(28),
      line(29),
      line(30)
    )
  }*/

  /*val schema: StructType = new StructType()
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
    .add("WEATHER_DELAY", StringType)*/

  /*def parseFlight1(line: Array[String]): Global.Flight1 = {
    Global.Flight1(line(0), line(1), line(2), line(3), line(4), line(5), line(6).toInt)
  }*/

  case class FlightDelay
  (
    no: Int,
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
    Distance: Double
  ) extends Serializable

  def parseFlight(line: Array[String]): FlightDelay = {
    FlightDelay(
      line(0).toInt,
      line(1).toByte,
      line(2).toDouble,
      line(3).toDouble,
      line(4).toDouble,
      line(5).toDouble,
      line(6),
      line(7),
      line(8),
      line(9).toDouble,
      line(10).toDouble,
      line(11).toDouble,
      line(12).toDouble,
      line(13).toDouble,
      line(14),
      line(15),
      line(16).toDouble)
  }

  def findMaxIncoming(graph: Graph[String, Int],
                      airportsMap: Map[VertexId, String]
                     ): Array[(String, Int)] = {
    val maxIncoming = graph.inDegrees.collect.sortWith(_._2 > _._2).map(x => (airportsMap(x._1), x._2))
    maxIncoming
  }

  def dijktra(graph: Graph[String, Double],  sourceId: Int): Graph[Double, Double] = {
    val gg = graph.mapEdges(e => 50.toDouble + e.attr / 2)
    val initialGraph = gg.mapVertices(
      (id, _) => if (id == sourceId) 0.0 else Double.PositiveInfinity
    )
    val sssp = initialGraph.pregel(Double.PositiveInfinity)(
      (_, dist, newDist) => math.min(dist, newDist),
      triplet => {
        if (triplet.srcAttr + triplet.attr < triplet.dstAttr) {
          Iterator((triplet.dstId, triplet.srcAttr + triplet.attr))
        }
        else {
          Iterator.empty
        }
      },
      (a, b) => math.min(a, b)
    )
    sssp
  }

  def getMaxDegree(graph: Graph[String, Double]): Array[(VertexId, PartitionID)] = {
    def maxVertex(a: (VertexId, Int), b: (VertexId, Int)): (VertexId, Int) = {
      if (a._2 > b._2) a else b
    }
    val maxIndegree: (VertexId, Int) = graph.inDegrees.reduce(maxVertex)
    val maxOutDegree: (VertexId, Int) = graph.outDegrees.reduce(maxVertex)
    val maxDegrees: (VertexId, Int) = graph.degrees.reduce(maxVertex)
    Array(maxIndegree, maxOutDegree, maxDegrees)
  }
}