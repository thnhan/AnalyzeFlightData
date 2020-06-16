import org.apache.spark.graphx.{EdgeDirection, Graph, VertexId}

object ShortestOutgoing {
  def run(graph: Graph[String, Double], sourceId: Int): Graph[(Double, List[VertexId]), Double] = {
    val initialGraph = graph.mapVertices(
      (id, _) =>
        if (id == sourceId) (0.0, List[VertexId](sourceId))
        else (Double.PositiveInfinity, List[VertexId]())
    )

    val sssp = initialGraph.pregel((Double.PositiveInfinity, List[VertexId]()), Int.MaxValue, EdgeDirection.Out)(
      // Vertex Program
      (id, dist, newDist) => if (dist._1 < newDist._1) dist else newDist,
      // Send Message
      triplet => {
        if (triplet.srcAttr._1 < triplet.dstAttr._1 - triplet.attr) {
          Iterator((triplet.dstId, (triplet.srcAttr._1 + triplet.attr, triplet.srcAttr._2 :+ triplet.dstId)))
        } else {
          Iterator.empty
        }
      },
      //Merge Message
      (a, b) => if (a._1 < b._1) a else b)
    return sssp
  }

}
