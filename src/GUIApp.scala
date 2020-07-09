import scala.swing._
import java.awt.Font

import org.apache.commons.io.output.ByteArrayOutputStream
import org.apache.spark.graphx.{Edge, Graph, VertexId}
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Try

class UI extends MainFrame {
  /* Global variables */
  val num_defaut_label: Label = new Label(Global.num_defaut_label)
//  val path_file_data: String = Global.path_file_data
  var strCols: String = ""

  def restrictHeight(s: Component) {
    s.maximumSize = new Dimension(Short.MaxValue, s.preferredSize.height)
  }

  title = "Analyze Flight Data"

  /*val filenameField: TextField = new TextField {
    columns = 80
  }*/
  val filenameField = new ComboBox(List("DATA/Flights.csv", "DATA/Flights_small.csv"))

  val paraTuning = new CheckBox("Parameters Tuning")
  paraTuning.selected = false
  val status1 = new RadioButton("Naive Bayes")
  val status2 = new RadioButton("Decision Tree")
  val status3 = new RadioButton("Linear SVC")
  val status4 = new RadioButton("Random Forest")
  val status5 = new RadioButton("Gradient-Boosted Tree")
  status3.selected = true
  val statusGroup = new ButtonGroup(status1, status2, status3, status4, status5)
  //  val gender = new ComboBox(List("don't know", "female", "male"))
  val queryGraphX = new ComboBox(Global.list)

  /* Initial Comment Field */
  val commentField: TextArea = new TextArea {
    rows = 40
    columns = 80
    lineWrap = false
    wordWrap = false
  }
  //  val pressMe = new ToggleButton("Press me!")
  //  pressMe.selected = true
  //  restrictHeight(nameField)
  //  restrictHeight(query)

  contents = new BoxPanel(Orientation.Vertical) {
    contents += new BoxPanel(Orientation.Horizontal) {
      contents += new Label("Flights Data")
      contents += Swing.HStrut(5)
      contents += filenameField
      filenameField.item = "DATA/Flights_small.csv"
    }

    contents += Swing.VStrut(5)
    contents += new BoxPanel(Orientation.Horizontal) {
      contents += status1
      contents += Swing.HStrut(10)
      contents += status2
      contents += Swing.HStrut(10)
      contents += status3
      contents += Swing.HStrut(10)
      contents += status4
      contents += Swing.HStrut(10)
      contents += status5
    }
    contents += Swing.VStrut(5)
    contents += paraTuning

    contents += Swing.VStrut(5)
    contents += new BoxPanel(Orientation.Horizontal) {
      contents += new Label("Query - GrahpX")
      contents += Swing.HStrut(20)
      contents += queryGraphX
    }

    contents += Swing.VStrut(5)
    contents += new Label("Result")
    contents += Swing.VStrut(3)
    contents += new ScrollPane(commentField)
    contents += Swing.VStrut(5)
    contents += new BoxPanel(Orientation.Horizontal) {
      contents += Button("Train & Evaluate model") {
        trainAndEvaluate()
      }

      contents += Swing.HGlue
      //      contents += Swing.VStrut(5)
      contents += Button("Process the query") {
        pressMeTwo()
      }
      contents += Swing.VStrut(5)
      contents += Button("Close") {
        reportAndClose()
      }
    }

    for (e <- contents)
      e.xLayoutAlignment = 0.0
    border = Swing.EmptyBorder(10, 10, 10, 10)
  }

  var graph: Graph[String, Double] = createGraph()
  val airportsMap: Map[VertexId, String] = graph.vertices.map {
    case (id, code) => id -> code
  }.collect.toList.toMap

  def reportAndClose() {
    println("Goodbye TRAN HOAI NHAN")
    sys.exit(0)
  }

  def pressMeTwo() {
    queryGraphX.selection.item match {
      case "Show the FLIGHTS data" =>
        /* Initial */
        val r = Dialog.showInput(contents.head, "Input number of airports", initial = num_defaut_label.text)
        var num = 5
        r match {
          case Some(x) => Try(num = x.toInt)
          case None =>
        }
        showFlightsData(num)

      case "Show information about the number of AIRPORTS and ROUTES" =>
        val numAirport = graph.numVertices
        val numRoutes = graph.numEdges
        val maxdegrees = AirportGraph.getMaxDegree(graph)

        var lines = "The total number of airports and routes:\n"
        lines += "+------------------+------------------+\n"
        lines += "|Number of airports|  Number of routes|\n"
        lines += "+------------------+------------------+\n"
        var info1 = numAirport.toString
        for (_ <- 1 to (18 - info1.length())) {
          info1 = " " + info1
        }
        var info2 = numRoutes.toString
        for (_ <- 1 to (18 - info2.length())) {
          info2 = " " + info2
        }
        lines += s"|$info1|$info2|\n"
        lines += "+------------------+------------------+\n\n"

        lines += "The airport has the largest number of incoming flights:\n"
        lines += "+------------------+------------------+\n"
        lines += "|           Airport| Number of flights|\n"
        lines += "+------------------+------------------+\n"
        info1 = airportsMap(maxdegrees(0)._1)
        for (_ <- 1 to (18 - info1.length())) {
          info1 = " " + info1
        }
        info2 = maxdegrees(0)._2.toString
        for (_ <- 1 to (18 - info2.length())) {
          info2 = " " + info2
        }
        lines += s"|$info1|$info2|\n"
        lines += "+------------------+------------------+\n\n"

        lines += "The airport has the largest number of outgoing flights:\n"
        lines += "+------------------+------------------+\n"
        lines += "|           Airport| Number of flights|\n"
        lines += "+------------------+------------------+\n"
        info1 = airportsMap(maxdegrees(1)._1)
        for (_ <- 1 to (18 - info1.length())) {
          info1 = " " + info1
        }
        info2 = maxdegrees(1)._2.toString
        for (_ <- 1 to (18 - info2.length())) {
          info2 = " " + info2
        }
        lines += s"|$info1|$info2|\n"
        lines += "+------------------+------------------+\n\n"

        lines += "The airport has the largest number of flights:\n"
        lines += "+------------------+------------------+\n"
        lines += "|           Airport| Number of flights|\n"
        lines += "+------------------+------------------+\n"
        info1 = airportsMap(maxdegrees(2)._1)
        for (_ <- 1 to (18 - info1.length())) {
          info1 = " " + info1
        }
        info2 = maxdegrees(2)._2.toString
        for (_ <- 1 to (18 - info2.length())) {
          info2 = " " + info2
        }
        lines += s"|$info1|$info2|\n"
        lines += "+------------------+------------------+\n\n"
        commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
        commentField.text = lines

      case "Find the airport with the highest incoming flights" =>
        /* Initial */
        val r = Dialog.showInput(contents.head, "Input number of airports", initial = num_defaut_label.text)
        var num = 5
        r match {
          case Some(x) => Try(num = x.toInt)
          case None =>
        }

        val newIn = graph.inDegrees.collect.sortWith(_._2 > _._2).map(x => (airportsMap(x._1), x._2))
        var lines = s"Top $num the airport with the highest incoming flights:\n"
        lines += "+------+--------------+--------------------------+\n"
        lines += "|    No|  AIRPORT CODE|Number of incoming flights|\n"
        lines += "+------+--------------+--------------------------+\n"
        var index = 0
        for (x <- newIn.filter(x => x._1 != "nowhere").take(num)) {
          index += 1
          var len = 0

          var inf1 = index.toString
          len = inf1.length
          for (i <- 1 to (6 - len)) {
            inf1 = " " + inf1
          }

          var inf2 = x._1
          len = inf2.length
          for (i <- 1 to (14 - len)) {
            inf2 = " " + inf2
          }

          var inf3 = x._2.toString
          len = inf3.length
          for (i <- 1 to (26 - len)) {
            inf3 = " " + inf3
          }
          lines += s"|$inf1|$inf2|$inf3|\n"
        }
        lines += "+------+--------------+--------------------------+\n"
        commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
        commentField.text = lines

      case "Find the airport with the highest outgoing flights" =>
        /* Initial */
        val r = Dialog.showInput(contents.head, "Input number of airports", initial = num_defaut_label.text)
        var num = 5
        r match {
          case Some(x) => Try(num = x.toInt)
          case None =>
        }

        val maxOut = graph.outDegrees.collect.sortWith(_._2 > _._2).map(x => (airportsMap(x._1), x._2))
        var lines = s"Top $num the airport with the highest outgoing flights:\n"
        lines += "+------+--------------+--------------------------+\n"
        lines += "|    No|  AIRPORT CODE|Number of outgoing flights|\n"
        lines += "+------+--------------+--------------------------+\n"
        var index = 0
        for (x <- maxOut.filter(x => x._1 != "nowhere").take(num)) {
          index += 1
          var len = 0

          var inf1 = index.toString
          len = inf1.length
          for (_ <- 1 to (6 - len)) {
            inf1 = " " + inf1
          }

          var inf2 = x._1
          len = inf2.length
          for (_ <- 1 to (14 - len)) {
            inf2 = " " + inf2
          }

          var inf3 = x._2.toString
          len = inf3.length
          for (i <- 1 to (26 - len)) {
            inf3 = " " + inf3
          }
          lines += s"|$inf1|$inf2|$inf3|\n"
        }
        lines += "+------+--------------+--------------------------+\n"
        commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
        commentField.text = lines

      case "Find the most important airports" =>
        /* Initial */
        val r = Dialog.showInput(contents.head, "Input number of airports", initial = num_defaut_label.text)
        var num = 5
        r match {
          case Some(x) => Try(num = x.toInt)
          case None =>
        }

        /* Use PageRank algorithm to score airports */
        val ranks = graph.pageRank(0.1).vertices
        //.filter(v => airportsMap(v._1) != "nowhere")
        val temp = ranks.join(graph.vertices)
        val temp2 = temp.sortBy(_._2, ascending = false)
        val impotant = temp2.collect

        /* Print */
        var lines = s"Top $num the most important airports:\n"
        lines += "+------+--------------+\n"
        lines += "|  Rank|  AIRPORT CODE|\n"
        lines += "+------+--------------+\n"
        var index = 0
        for (x <- impotant.filter(v => airportsMap(v._1) != "nowhere").take(num)) {
          index += 1
          var inf1 = index.toString
          var len = inf1.length
          for (i <- 1 to (6 - len)) {
            inf1 = " " + inf1
          }
          var inf2 = airportsMap(x._1)
          len = inf2.length
          for (i <- 1 to (14 - len)) {
            inf2 = " " + inf2
          }
          lines += s"|$inf1|$inf2|\n"
        }
        lines += "+------+--------------+\n"
        commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
        commentField.text = lines

      case "Find paths with the lowest flight costs" =>
        val r = Dialog.showInput(contents.head, "Input the airport code (LAX)", initial = "LAX")
        var sourceAirport = "LAX"
        r match {
          case Some(x) => Try(sourceAirport = x)
          case None =>
        }

        /* Initial Graph */
        val graph_cost = graph.mapEdges(e => 50.toDouble + e.attr.toDouble / 2)
        /* Find all shortest paths form source vertex */
        val result = ShortestOutgoing.run(graph_cost, sourceAirport.hashCode)

        var lines = s"5 paths from $sourceAirport airport and their lowest flight cost:\n"
        lines += "+------+-------------------------------+----------+\n"
        lines += "|    No|                           Path|      Cost|\n"
        lines += "+------+-------------------------------+----------+\n"
        var index = 0
        for (x <- result.vertices.filter(
          v => v._2._1 < Double.PositiveInfinity && v._2._2.length > 2
        ).take(10)) {
          index += 1
          var len = 0

          var inf1 = index.toString
          len = inf1.length
          for (_ <- 1 to (6 - len)) {
            inf1 = " " + inf1
          }

          var inf2 = airportsMap(x._2._2.head)
          for (v <- x._2._2.takeRight(x._2._2.length - 1)) {
            if (airportsMap(v) != "nowhere") {
              inf2 += "->" + airportsMap(v)
            }
          }
          len = inf2.length
          for (_ <- 1 to (31 - len)) {
            inf2 = " " + inf2
          }

          var inf3 = x._2._1.toString
          len = inf3.length
          for (i <- 1 to (10 - len)) {
            inf3 = " " + inf3
          }
          lines += s"|$inf1|$inf2|$inf3|\n"
        }
        lines += "+------+-------------------------------+----------+\n"
        commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
        commentField.text = lines

      case "Find the airport with the lowest flight costs" =>
        /* Get input and prepare */
        val r = Dialog.showInput(contents.head, "Input the airport code (LAX)", initial = "LAX")
        var sourceAirport = "LAX"
        r match {
          case Some(x) => Try(sourceAirport = x)
          case None =>
        }

        /* Find shortest path and Print */
        val sssp = AirportGraph.dijktra(graph, sourceAirport.hashCode)
        var lines = "The Routes with the lowest flight costs:\n"
        lines += "+--------------+-------------------+---------+\n"
        lines += "|ORIGIN AIRPORT|DESTINATION AIRPORT|     Cost|\n"
        lines += "+--------------+-------------------+---------+\n"
        for (x <- sssp.edges.sortBy(_.attr).take(10)) {
          if (airportsMap(x.srcId) != "nowhere" && airportsMap(x.dstId) != "nowhere") {
            var inf1: String = airportsMap(x.srcId)
            var inf2: String = airportsMap(x.dstId)
            var inf3: String = x.attr.toString

            var len = inf1.length
            for (i <- 1 to (14 - len)) {
              inf1 = " " + inf1
            }
            len = inf2.length
            for (i <- 1 to (19 - len)) {
              inf2 = " " + inf2
            }
            len = inf3.length
            for (_ <- 1 to (9 - len)) {
              inf3 = " " + inf3
            }
            lines += s"|$inf1|$inf2|$inf3|\n"
          }
        }
        lines += "+--------------+-------------------+---------+\n\n"

        lines += "The airports and their lowest flight costs\n"
        lines += "+--------------+---------+\n"
        lines += "|  Airport code|     Cost|\n"
        lines += "+--------------+---------+\n"
        for (x <- sssp.vertices.sortBy(_._2).take(10)) {
          if (airportsMap(x._1) != "nowhere") {
            var inf1: String = airportsMap(x._1)
            var inf2: String = x._2.toString
            var len = inf1.length
            for (i <- 1 to (14 - len)) {
              inf1 = " " + inf1
            }
            len = inf2.length
            for (i <- 1 to (9 - len)) {
              inf2 = " " + inf2
            }
            lines += s"|$inf1|$inf2|\n"
          }
        }
        lines += "+--------------+---------+\n"
        commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
        commentField.text = lines

      case "List all VERTICES and EDGES of the graph" =>
        val vertices = graph.vertices
          .filter(v => v._2 != "nowhere")
          .collect
        var lines = "List all vertices of the graph:\n"
        lines += "+------+--------------+--------------+\n"
        lines += "|    No|      VertexID|  Airport code|\n"
        lines += "+------+--------------+--------------+\n"
        var No = 0
        for (x <- vertices) {
          No += 1
          var inf1: String = No.toString
          var inf2: String = x._1.toString
          var inf3: String = x._2
          var len = inf1.length
          for (i <- 1 to (6 - inf1.length)) {
            inf1 = " " + inf1
          }
          for (i <- 1 to (14 - inf2.length)) {
            inf2 = " " + inf2
          }
          for (i <- 1 to (14 - inf3.length)) {
            inf3 = " " + inf3
          }
          lines += s"|$inf1|$inf2|$inf3|\n"
        }
        lines += "+------+--------------+--------------+\n\n"

        val edges = graph.triplets
          .filter(e => e.srcAttr != "nowhere" && e.dstAttr != "nowhere")
          .collect
        lines += "List all edges of the graph:\n"
        lines += "+------+----------------+----------------+------------+\n"
        lines += "|    No|        Vertex A|        Vertex B|    Distance|\n"
        lines += "+------+----------------+----------------+------------+\n"
        var index = 0
        for (x <- edges) {
          index += 1
          var inf1: String = index.toString
          var inf2: String = (x.srcId, x.srcAttr).toString
          var inf3: String = (x.dstId, x.dstAttr).toString
          var inf4: String = x.attr.toString
          for (i <- 1 to (6 - inf1.length)) {
            inf1 = " " + inf1
          }
          for (i <- 1 to (16 - inf2.length)) {
            inf2 = " " + inf2
          }
          for (i <- 1 to (16 - inf3.length)) {
            inf3 = " " + inf3
          }
          for (i <- 1 to (12 - inf4.length)) {
            inf4 = " " + inf4
          }
          lines += s"|$inf1|$inf2|$inf3|$inf4|\n"
        }
        lines += "+------+----------------+----------------+------------+\n"

        commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
        commentField.text = lines

      case "List all EDGES which have distance longer than ..." =>
        val r = Dialog.showInput(contents.head, "Input DISTANCE", initial = "500")
        var distance = 500
        r match {
          case Some(x) => Try(distance = x.toInt)
          case None =>
        }

        val edges = graph.triplets
          .filter(e => e.srcAttr != "nowhere" && e.dstAttr != "nowhere" && e.attr > distance)
          .sortBy(_.attr)
          .collect

        var lines = s"List all edges which have distance longer than $distance:\n"
        lines += "+------+----------------+----------------+------------+\n"
        lines += "|    No|        Vertex A|        Vertex B|    Distance|\n"
        lines += "+------+----------------+----------------+------------+\n"
        var index = 0
        for (x <- edges) {
          index += 1
          var inf1: String = index.toString
          var inf2: String = (x.srcId, x.srcAttr).toString
          var inf3: String = (x.dstId, x.dstAttr).toString
          var inf4: String = x.attr.toString
          for (i <- 1 to (6 - inf1.length)) {
            inf1 = " " + inf1
          }
          for (i <- 1 to (16 - inf2.length)) {
            inf2 = " " + inf2
          }
          for (i <- 1 to (16 - inf3.length)) {
            inf3 = " " + inf3
          }
          for (i <- 1 to (12 - inf4.length)) {
            inf4 = " " + inf4
          }
          lines += s"|$inf1|$inf2|$inf3|$inf4|\n"
        }
        lines += "+------+----------------+----------------+------------+\n"

        commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
        commentField.text = lines
    }
  }

  def createGraph():  Graph[String, Double] = {
    /* Initial */
    val conf = new SparkConf().setAppName("GuiApp").setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    /* Load data as a DataFrame */
    val spark = SparkSession.builder().getOrCreate()
    val data_df = spark
      .read
      .format("csv")
      .schema(Global.schema)
      .option("header", value = true)
      .load(filenameField.item) // hsjjshdsjhdgsjahdgjashdgjashdgjashdg
      .select("No", "DayOfWeek", "DepTime", "CRSDepTime", "ArrTime", "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "ActualElapsedTime", "CRSElapsedTime", "AirTime", "ArrDelay", "DepDelay", "Origin", "Dest", "Distance")

    /* Print data in commentField */
    val outCapture = new ByteArrayOutputStream
    Console.withOut(outCapture) {
      data_df.show()
    }
    val table_plt = new String(outCapture.toByteArray)
    commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
    commentField.text = table_plt + "Total number of items: " + data_df.count.toString

    /* Convert DataFrame to RDD[String] */
    val RDD_String = data_df.rdd.map(x => {
      x.mkString(",").replace("[", "").replace("]", "")
    })
//    RDD_String.take(5).foreach(println)

    /* Load data and parse */
    val header_line = RDD_String.first()
    val flightsRDD = RDD_String
      .filter(line => !line.contains(header_line))
      .map(_.split(","))
      .map(line => AirportGraph.parseFlight(line))

    /* Create Graph from airports, edges */
    val nowhere = "nowhere"
    val airports = flightsRDD.map(flight =>
      (flight.Origin.hashCode.toLong, flight.Origin)
    ).distinct()
    val routes = flightsRDD.map(flight =>
      ((flight.Origin, flight.Dest), flight.Distance)
    ).distinct()
    val edges = routes.map(
      route => Edge(route._1._1.hashCode.toLong, route._1._2.hashCode.toLong, route._2)
    )
    val graph = Graph(airports, edges, nowhere)
    graph
  }

  def changeText() {
    val r = Dialog.showInput(contents.head, "New label text", initial = num_defaut_label.text)
    r match {
      case Some(s) => num_defaut_label.text = s
      case None =>
    }
  }

  def showFlightsData(num_row: Int = 50): Unit = {
    /* Load data as a DataFrame */
    val spark = SparkSession.builder().getOrCreate()
    val data_df = spark
      .read
      .format("csv")
      .option("header", value = true)
      .schema(Global.schema)
      .load(filenameField.item)
      .select("No", "DayOfWeek", "DepTime", "CRSDepTime", "ArrTime", "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "ActualElapsedTime", "CRSElapsedTime", "AirTime", "ArrDelay", "DepDelay", "Origin", "Dest", "Distance")

    /* Print data in commentField */
    val outCapture = new ByteArrayOutputStream
    Console.withOut(outCapture) {
      data_df.show(num_row)
    }
    val table_plt = new String(outCapture.toByteArray)
    commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
    commentField.text = table_plt
  }

  def trainAndEvaluate() {
    if (status1.selected) {
      commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
      commentField.text = "Naive Bayes"
      println("Naive Bayes")
      applyML(Array("Naive Bayes"))
    }
    if (status2.selected) {
      commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
      if (paraTuning.selected) {
        commentField.text = "Decision Tree with Parameters tuning"
        applyML(Array("Decision Tree", "Parameters tuning"))
      }
      else {
        commentField.text = "Decision Tree without Parameters tuning"
        applyML(Array("Decision Tree", ""))
      }
    }
    if (status3.selected) {
      commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
      if (paraTuning.selected) {
        commentField.text = "Linear SVC with Parameters tuning"
        applyML(Array("Linear SVC", "Parameters tuning"))
      }
      else {
        commentField.text = "Linear SVC without Parameters tuning"
        applyML(Array("Linear SVC", ""))
      }
    }
    if (status4.selected) {
      commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
      if (paraTuning.selected) {
        commentField.text = "Random Forest with Parameters tuning"
        applyML(Array("Random Forest", "Parameters tuning"))
      }
      else {
        commentField.text = "Random Forest without Parameters tuning"
        applyML(Array("Random Forest", ""))
      }
    }
    if (status5.selected) {
      commentField.font = new Font("Monospaced", Font.TRUETYPE_FONT, 16)
      commentField.text = "Gradient-Boosted Tree"
      applyML(Array("Gradient-Boosted Tree", ""))
    }
  }

  /* ML application */
  def applyML(args: Array[String]): Unit = {
    //    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("GUIApp")
      .getOrCreate

    import spark.implicits._
    val dataFrame = spark
      .read
      .format("csv")
      .option("header", value = true)
      .schema(Global.schema)
      .load(filenameField.item)
    //.as[Global.FlightDelay]

    val dataset = dataFrame
      .select("No", "DayOfWeek", "DepTime", "CRSDepTime", "ArrTime", "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "ActualElapsedTime", "CRSElapsedTime", "AirTime", "ArrDelay", "DepDelay", "Origin", "Dest", "Distance")

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
    //    println(d)
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

    if (args(0) == "Random Forest" || args(0) == "Decision Tree")
      group1(args, dataset, balanceDataset, stringCols, numbericCols, spark)
    else
      group2(args, dataset, balanceDataset, stringCols, numbericCols, spark)
  }

  def group1(args: Array[String],
             dataset: DataFrame, balanceDataset: DataFrame,
             stringCols: Array[String],
             numbericCols: Array[String],
             spark: SparkSession
            ): Unit = {

    val (feaImp, metricsDF, confusionMatrix) = args(0) match {
      case "Decision Tree" =>
        BasedonDecisionTree.run(
          Array(args(1), "Features Important"),
          balanceDataset,
          stringCols,
          numbericCols,
          spark
        )

      case "Random Forest" =>
        BasedonRandomForest.run(
          Array(args(1), "Features Importance"),
          balanceDataset,
          stringCols,
          numbericCols,
          spark
        )
    }

    /* Print dataset */
    println(s"Flight Data:")
    dataset.show(truncate = false)
    println(s"Total mumber of rows of dataset = ${dataset.count()}\n")
    println(s"Schema of Dataset:")
    dataset.printSchema()
    println(s"Columns are used as features:")
    for (elem <- (stringCols ++ numbericCols)) print(elem + ",")
    print("\n\n")
    println("Feature Importance")
    feaImp.show(false)
    /* Print scores */
    println(s"Scores of classifier ${args(0)}:")
    metricsDF.show(truncate = false)
    println("Confusion matrix:")
    println(confusionMatrix)

    /* Print */
    val outCapture = new ByteArrayOutputStream
    Console.withOut(outCapture) {
      /* Print dataset */
      println(s"Flight Data:")
      dataset.show(truncate = false)
      println(s"Total mumber of rows of dataset = ${dataset.count()}\n")
      println(s"Schema of Dataset:")
      dataset.printSchema()
      /* Print scores */
      println(s"Scores:")
      metricsDF.show(truncate = false)
      println("Confusion matrix:")
      println(confusionMatrix)
    }
    val table_plt = new String(outCapture.toByteArray)
    commentField.text += "\n\n" + table_plt
  }

  def group2(args: Array[String],
             dataset: DataFrame, balanceDataset: DataFrame,
             stringCols: Array[String],
             numbericCols: Array[String],
             spark: SparkSession
            ): Unit = {

    val (metricsDF, confusionMatrix) = args(0) match {
      case "Naive Bayes" =>
        BasedonNaiveBayes.run(
          Array(""),
          balanceDataset,
          stringCols,
          numbericCols,
          spark
        )
      case "Linear SVC" =>
        BasedonLinearSVC.run(
          Array(args(1)), // args(1): Parameters Tuning or Not
          balanceDataset,
          stringCols,
          numbericCols,
          spark
        )
      case "Gradient-Boosted Tree" =>
        /*Run Gradient-Boost Tree*/
        BasedonGradientBoostedTree.run(
          Array(""),
          balanceDataset,
          stringCols,
          numbericCols,
          spark
        )
    }

    /* Print */
    val outCapture = new ByteArrayOutputStream
    Console.withOut(outCapture) {
      /* Print dataset */
      println(s"Flight Data:")
      dataset.show(truncate = false)
      println(s"Total mumber of rows of dataset = ${dataset.count()}\n")
      println(s"Schema of Dataset:")
      dataset.printSchema()
      println(s"Columns are used as features:")
      for (elem <- (stringCols ++ numbericCols)) print(elem + ",")
      print("\n\n")
      /* Print scores */
      println(s"Scores of classifier ${args(0)}:")
      metricsDF.show(truncate = false)
      println("Confusion matrix:")
      println(confusionMatrix)
    }
    val table_plt = new String(outCapture.toByteArray)
    commentField.text += "\n\n" + table_plt

    println(s"Flight Data:")
    dataset.show(truncate = false)
    println(s"Total mumber of rows of dataset = ${dataset.count()}\n")
    println(s"Schema of Dataset:")
    dataset.printSchema()
    println(s"Columns are used as features:")
    for (elem <- (stringCols ++ numbericCols)) print(elem + ",")
    print("\n\n")
    /* Print scores */
    println(s"Scores of classifier ${args(0)}:")
    metricsDF.show(truncate = false)
    println("Confusion matrix:")
    println(confusionMatrix)
  }
}

object GUIApp extends App {
  val ui = new UI
  ui.visible = true
}