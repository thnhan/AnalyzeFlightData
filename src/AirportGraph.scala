import org.apache.spark.graphx._

object AirportGraph {
  def parseFlight(line: Array[String]): Global.Flight = {
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
  }

  def parseFlight1(line: Array[String]): Global.Flight1 = {
    Global.Flight1(line(0), line(1), line(2), line(3), line(4), line(5), line(6).toInt)
  }

  def findMaxIncoming(graph: Graph[String, Int],
                      airportsMap: Map[VertexId, String]
                     ): Array[(String, Int)] = {
    val maxIncoming = graph.inDegrees.collect.sortWith(_._2 > _._2).map(x => (airportsMap(x._1), x._2))
    maxIncoming
  }

  def dijktra(graph: Graph[String, Int],  sourceId: Int): Graph[Double, Double] = {
    val gg = graph.mapEdges(e => 50.toDouble + e.attr.toDouble / 2)
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

  def getMaxDegree(graph: Graph[String, Int]) = {
    def maxVertex(a: (VertexId, Int), b: (VertexId, Int)): (VertexId, Int) = {
      if (a._2 > b._2) a else b
    }
    val maxIndegree: (VertexId, Int) = graph.inDegrees.reduce(maxVertex)
    val maxOutDegree: (VertexId, Int) = graph.outDegrees.reduce(maxVertex)
    val maxDegrees: (VertexId, Int) = graph.degrees.reduce(maxVertex)
    Array(maxIndegree, maxOutDegree, maxDegrees)
  }
}