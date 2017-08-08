import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import scala.util.Random
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}

object Read {
  def main(Args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder
      .master("local[*]")
      .appName("Titanic")
      .config("spark.sql.parquet.compression.codec", "gzip")
      .getOrCreate()

    import sparkSession.implicits._
    val Data = sparkSession.read.option("header", "true")
      .option("charset", "UTF8")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .csv("./resources/titanic.csv")
    Data
      .printSchema()

    val CleanData = Data
      .select(
        $"Survived",
        $"Pclass",
        $"Sex",
        $"Age",
        $"SibSp",
        $"Parch",
        $"Fare")
      .filter($"Age".isNotNull)
      .toDF()

    CleanData.describe().filter($"summary" === "count").show

    val Array(trainData, testData) = CleanData.randomSplit(Array(0.8, 0.2), seed = 1L)
    trainData.cache()
    testData.cache()

    val pclassOneHotEncoder = new OneHotEncoder().
      setInputCol("Pclass").
      setOutputCol("PclassVector")

    val sexIndexer = new StringIndexer().
      setInputCol("Sex").
      setOutputCol("SexIndex")

    val inputCols = Array("PclassVector", "SexIndex", "Age", "SibSp", "Parch", "Fare")
    val assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("features")

    val classifier = new LogisticRegression()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setPredictionCol("Prediction")
      .setMaxIter(100)

    val evaluator = new MulticlassClassificationEvaluator().
      setLabelCol("Survived")
      .setPredictionCol("Prediction")
      .setMetricName("accuracy")

    val pipeline = new Pipeline().setStages(Array(pclassOneHotEncoder, sexIndexer, assembler, classifier))

    /*
    # Choose the best hyper-parameters using validation set
    */

    val regP = 0.01 to 0.1 by 0.01
    val ElaP = 0.1 to 0.95 by 0.05
    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.regParam, regP)
      .addGrid(classifier.elasticNetParam, ElaP)
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val validatorModel = trainValidationSplit.fit(trainData)
    val bestModel = validatorModel.bestModel
    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    val bestLRModel = bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LogisticRegressionModel]
    println(s"Coefficients: ${bestLRModel.coefficients} Intercept: ${bestLRModel.intercept}")

    val trainPredictions =  bestModel.transform(trainData)
    val testPredictions = bestModel.transform(testData)

    val trainAccuracy = evaluator.evaluate(trainPredictions)
    println("Train Data Accuracy = " + trainAccuracy)

    val testAccuracy = evaluator.evaluate(testPredictions)
    println("Test Data Accuracy = " + testAccuracy)

    /*
    testPredictions.filter($"Survived" =!= $"Prediction")
      .select("Age", "Sex", "Pclass", "Survived", "Prediction", "probability")
      .show(truncate = false)
    */
  }
}